import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from uuid import uuid4
from pandas import DataFrame
from typing import List, Dict, Any, Optional
import io
import boto3
import pandas as pd


from data_insight_agent.ai_schema import (
    AIParsedInstruction,
    Correlation,
    Regression,
    Visuals,
    Maths,
    Quantile,
)
from data_insight_agent.utils import simple_linear_regression, returning_metadata
from data_insight_agent.config import settings


class Analysis:
    def __init__(self, context_id: str, task_id: str):
        self.context_id = context_id
        self.task_id = task_id
        self.errors = {}
        self.b2_client = boto3.client(
            "s3",
            endpoint_url=settings.B2_ENDPOINT,
            aws_access_key_id=settings.B2_KEY_ID,
            aws_secret_access_key=settings.B2_APPLICATION_KEY,
            region_name="us-east-005",
        )


    @staticmethod
    def filter_by(df: DataFrame, filters: Optional[Dict[str, Any]] = None) -> DataFrame:
        if not filters:
            return df
        for key, value in filters.items():
            if key not in df.columns:
                continue
            if isinstance(value, (list, tuple)):
                df = df[df[key].isin(value)]
            else:
                df = df[df[key] == value]
        return df

    @staticmethod
    def focus_columns(df: DataFrame, cols: Optional[List[str]] = None) -> DataFrame:
        if not cols:
            return df
        existing_cols = [col for col in cols if col in df.columns]
        return df[existing_cols]

    @staticmethod
    def fill(df: DataFrame, filling: Optional[str] = None) -> DataFrame:
        if not filling:
            return df
        if filling == "ffill":
            return df.ffill()
        elif filling == "bfill":
            return df.bfill()
        else:
            return df.fillna(filling)

    @staticmethod
    def drop(df: DataFrame, action: Optional[str] = None) -> DataFrame:
        if not action:
            return df
        if action == "drop null":
            return df.dropna()
        elif action == "drop duplicates":
            return df.drop_duplicates()
        return df

    @staticmethod
    def sorting(df: DataFrame, sort_params: Optional[List[str]] = None) -> DataFrame:
        if not sort_params:
            return df
        ascending = True
        cols = []
        for param in sort_params:
            if param.lower() == "descending":
                ascending = False
            elif param.lower() == "ascending":
                ascending = True
            elif param in df.columns:
                cols.append(param)
        if cols:
            df = df.sort_values(by=cols, ascending=ascending)
        return df

    def upload_chart_to_b2(self, fig: Figure, filename: str) -> str:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close(fig)
        self.b2_client.upload_fileobj(
            Fileobj=buffer,
            Bucket=settings.B2_BUCKET,
            Key=filename,
            ExtraArgs={"ContentType": "image/png"},
        )

        return f"{settings.B2_ENDPOINT}/{settings.B2_BUCKET}/{filename}"

    def visualize(self, df: DataFrame, chart_type: str):
        plt.figure(figsize=(8, 5))
        self.ensure_bucket_exists()

        try:
            df = df.apply(pd.to_numeric)

            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

            def safe_cat_col():
                for col in categorical_cols:
                    if df[col].nunique() <= 20:
                        return col
                return categorical_cols[0] if categorical_cols else None

            chart_handlers = {
                "bar": lambda: (
                    df.groupby(safe_cat_col())[numeric_cols[0]].mean().plot(kind="bar")
                    if safe_cat_col() and numeric_cols
                    else (
                        df[numeric_cols[0]].value_counts().plot(kind="bar")
                        if numeric_cols
                        else (_ for _ in ()).throw(
                            ValueError("No valid data for bar chart.")
                        )
                    )
                ),
                "line": lambda: (
                    df[numeric_cols].plot(kind="line")
                    if numeric_cols
                    else (_ for _ in ()).throw(
                        ValueError("No numeric columns for line chart.")
                    )
                ),
                "scatter": lambda: (
                    df.plot(kind="scatter", x=numeric_cols[0], y=numeric_cols[1])
                    if len(numeric_cols) >= 2
                    else (_ for _ in ()).throw(
                        ValueError(
                            "Scatter plot requires at least two numeric columns."
                        )
                    )
                ),
                "pie": lambda: (
                    df[safe_cat_col()]
                    .value_counts()
                    .plot(kind="pie", autopct="%1.1f%%")
                    if safe_cat_col()
                    else (
                        df[numeric_cols[0]].plot(kind="pie", autopct="%1.1f%%")
                        if numeric_cols
                        else (_ for _ in ()).throw(
                            ValueError("No valid column for pie chart.")
                        )
                    )
                ),
                "hist": lambda: (
                    df[numeric_cols].hist(bins=20)
                    if numeric_cols
                    else (_ for _ in ()).throw(
                        ValueError("No numeric columns for histogram.")
                    )
                ),
                "box": lambda: (
                    df[numeric_cols].plot(kind="box")
                    if numeric_cols
                    else (_ for _ in ()).throw(
                        ValueError("No numeric columns for box plot.")
                    )
                ),
                "heatmap": lambda: (
                    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
                    if not df.corr(numeric_only=True).empty
                    else (_ for _ in ()).throw(
                        ValueError("No numeric correlation matrix for heatmap.")
                    )
                ),
            }

            if chart_type in chart_handlers:
                chart_handlers[chart_type]()
                plt.tight_layout()
                fig = plt.gcf()
                filename = f"{self.context_id}/{self.task_id}/{uuid4()}.png"
                path = self.upload_chart_to_b2(fig, filename)
                return {chart_type: path}

        except Exception as e:
            plt.close("all")
            return f"visual_error: Visualization failed: {str(e)}"

    def handle_summary(self, df: DataFrame):
        result = df.describe(include="all").to_dict()
        return result

    def handle_math(self, df: DataFrame, math_ops: Optional[Maths] = None):
        result = {}
        ops = []
        if not math_ops:
            return result
        numeric_df = df.select_dtypes(include="number")
        try:
            for op in math_ops:
                if isinstance(op, Quantile):
                    percentiles = op.percentiles or [0.25, 0.5, 0.75]
                    result["quantile"] = numeric_df.quantile(
                        percentiles, numeric_only=True
                    ).to_dict()
                else:
                    ops.append(op)
            if ops:
                result.update(numeric_df.agg(ops).to_dict())

        except Exception as e:
            self.errors["math_error"] = f"Math operation failed: {str(e)}"
        return {"math": result}

    def handle_correlation(self, df: DataFrame, corr: Optional[Correlation] = None):
        result = {}
        try:
            method = corr[0] if isinstance(corr, list) else corr
            result["correlation"] = df.corr(numeric_only=True, method=method).to_dict()
        except Exception as e:
            self.errors["correlation_error"] = f"Correlation analysis failed: {e}"
        return result

    def handle_regression(self, df: DataFrame, regres: Optional[Regression] = None):
        result = {}
        if not regres:
            return result
        try:
            regression = simple_linear_regression(df, regres.col_x, regres.col_y)
            error = regression.get("error")
            if error:
                self.errors["regression_error"] = error
                return result
            result["regression"] = regression
        except Exception as e:
            self.errors["regression_error"] = f"Regression analysis failed: {str(e)}"
            return result

    def handle_anomaly(self, df: DataFrame):
        result = {}
        try:
            numeric_df = df.select_dtypes(include="number")
            z_scores = (numeric_df - numeric_df.mean()) / numeric_df.std()
            anomalies = numeric_df[(z_scores.abs() > 3)].dropna()
            result["zscore_anomalies"] = anomalies.to_dict()
        except Exception as e:
            self.errors["anomaly_error"] = f"Anomaly detection failed: {str(e)}"
        return result

    def handle_visualization(self, df: DataFrame, visuals: Optional[Visuals] = None):
        if not visuals:
            return None
        chart_types = visuals if isinstance(visuals, list) else [visuals]
        if chart_types:
            charts = []
            visual_error = []
            for chart_type in chart_types:
                chart_or_error = self.visualize(df, chart_type)
                if isinstance(chart_or_error, str):
                    visual_error.append(chart_or_error)
                elif isinstance(chart_or_error, dict):
                    chart = chart_or_error
                    charts.append(chart)
            if visual_error:
                self.errors["visual_error"] = visual_error
            return charts

    def analyse(self, df: DataFrame, data: AIParsedInstruction, metadata: dict):
        analyses = {}
        metadata = returning_metadata(metadata)
        metadata["status"] = "in_progress"

        if not data.intent or (data.intent == "unknown" and data.confidence == 0):
            metadata["summary"] = df.describe(include="all").to_dict()
            metadata["message"] = (
                "Low confidence or unknown intent — default summary only."
            )
            metadata["status"] = "skipped"
            analyses["metadata"] = metadata
            return analyses

        try:
            df = (
                df.pipe(self.focus_columns, data.focus_columns or [])
                .pipe(self.filter_by, data.filters or {})
                .pipe(self.fill, data.fill)
                .pipe(self.drop, data.drop)
                .pipe(self.sorting, data.sort)
            )
        except Exception as e:
            print(f"Data Preparation Error: {e}")
            self.errors["Data Preparation Error"] = e

        intent_handlers = {
            "summary": lambda: self.handle_summary(df),
            "math": lambda: (
                self.handle_math(df, data.operations.math)
                if data.operations.math
                else None
            ),
            "correlation": lambda: (
                self.handle_correlation(df, data.operations.correlation)
                if data.operations.correlation
                else None
            ),
            "regression": lambda: (
                self.handle_regression(df, data.operations.regression)
                if isinstance(data.operations.regression, Regression)
                else None
            ),
            "anomaly": lambda: (
                self.handle_anomaly(df) if "anomaly" in data.intent else None
            ),
            "visualization": lambda: (
                self.handle_visualization(df, data.operations.visualization)
                if data.operations.visualization
                else None
            ),
        }

        if data.analysis_explanation:
            text_explanation = data.analysis_explanation
            analyses["overview of the analysis"] = text_explanation

        for intent in data.intent:
            if intent in intent_handlers and intent_handlers[intent]:
                result = intent_handlers[intent]()
                if result:
                    if isinstance(result, list):
                        analyses["visuals generated"] = result
                    else:
                        analyses.update(result)
        metadata.update(
            {
                "processed_columns": df.columns.tolist(),
                "num_rows": len(df),
                "confidence": data.confidence,
                "status": "completed",
            }
        )
        analyses["metadata"] = metadata
        return analyses
