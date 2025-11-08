import matplotlib.pyplot as plt
import seaborn as sns
from uuid import uuid4
from pandas import DataFrame
from typing import List, Dict, Any, Optional
from minio import Minio
import io
import pandas as pd

from data_insight_agent.schema import (
    AIParsedInstruction,
    Correlation,
    Regression,
    Operation,
    ResponseMessagePart,
    Artifact,
)
from data_insight_agent.utils import simple_linear_regression, returning_metadata
from data_insight_agent.config import settings


class Analysis:
    def __init__(self, context_id: str, task_id: str):
        self.context_id = context_id
        self.task_id = task_id
        self.errors = {}
        self.minio_client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False,
        )

    @staticmethod
    def filter_by(df: DataFrame, filters: Dict[str, Any]) -> DataFrame:
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
    def focus_columns(df: DataFrame, cols: List[str]) -> DataFrame:
        if not cols:
            return df
        existing_cols = [col for col in cols if col in df.columns]
        return df[existing_cols]

    @staticmethod
    def fill(df: DataFrame, filling: Optional[str]) -> DataFrame:
        if not filling:
            return df
        if filling == "ffill":
            return df.ffill()
        elif filling == "bfill":
            return df.bfill()
        else:
            return df.fillna(filling)

    @staticmethod
    def drop(df: DataFrame, action: Optional[str]) -> DataFrame:
        if not action:
            return df
        if action == "drop null":
            return df.dropna()
        elif action == "drop duplicates":
            return df.drop_duplicates()
        return df

    @staticmethod
    def sorting(df: DataFrame, sort_params: Optional[List[str]]) -> DataFrame:
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

    def ensure_bucket_exists(self):
        if not self.minio_client.bucket_exists(settings.MINIO_BUCKET):
            self.minio_client.make_bucket(settings.MINIO_BUCKET)

    def visualize(self, df: DataFrame, chart_type: str):
        plt.figure(figsize=(8, 5))
        self.ensure_bucket_exists()

        try:
            df = df.apply(pd.to_numeric, errors="ignore")

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

                buffer = io.BytesIO()
                plt.savefig(buffer, format="png")
                buffer.seek(0)
                plt.close("all")

                filename = f"{self.context_id}/{self.task_id}/{uuid4()}.png"
                self.minio_client.put_object(
                    settings.MINIO_BUCKET,
                    filename,
                    buffer,
                    length=buffer.getbuffer().nbytes,
                    content_type="image/png",
                )
                path = f"{settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET}/{filename}"
                return chart_type, path

        except Exception as e:
            plt.close("all")
            return f"visual_error: Visualization failed: {str(e)}"

    def handle_summary(self, df: DataFrame):
        result = df.describe(include="all").to_dict()
        return ResponseMessagePart(kind="data", data=result)

    def handle_math(self, df: DataFrame, operations: Operation):
        result = {}
        try:
            math_ops = [op for op in operations.math if op != "quantile"]
            if math_ops:
                result.update(df.agg(math_ops).to_dict())

            if "quantile" in operations.math:
                percentiles = getattr(
                    getattr(operations, "quantile", None), "percentile", None
                ) or [0.25, 0.5, 0.75]
                result["quantile"] = df.quantile(
                    percentiles, numeric_only=True
                ).to_dict()
        except Exception as e:
            self.errors["math_error"] = f"Math operation failed: {str(e)}"
        return ResponseMessagePart(kind="data", data=result)

    def handle_relationship(self, df: DataFrame, relationships):
        result = {}
        try:
            for rel in relationships:
                if isinstance(rel, Correlation):
                    result["correlation"] = df.corr(
                        numeric_only=True, method=rel.correlation
                    ).to_dict()
                elif isinstance(rel, Regression):
                    result["regression"] = simple_linear_regression(
                        df, rel.col_x, rel.col_y
                    )
        except Exception as e:
            self.errors["relationship_error"] = (
                f"Relationship analysis failed: {str(e)}"
            )
        return ResponseMessagePart(kind="data", data=result)

    def handle_anomaly(self, df: DataFrame):
        result = {}
        try:
            numeric_df = df.select_dtypes(include="number")
            z_scores = (numeric_df - numeric_df.mean()) / numeric_df.std()
            anomalies = numeric_df[(z_scores.abs() > 3)].dropna()
            result["zscore_anomalies"] = anomalies.to_dict()
        except Exception as e:
            self.errors["anomaly_error"] = f"Anomaly detection failed: {str(e)}"
        return ResponseMessagePart(kind="data", data=result)

    def handle_visualization(self, df: DataFrame, operations: Operation):
        chart_types = []
        if getattr(operations, "visualization", None):
            chart_types = (
                operations.visualization
                if isinstance(operations.visualization, list)
                else [operations.visualization]
            )

        if chart_types:
            charts = []
            visual_error = []
            for chart_type in chart_types:
                chart_or_error = self.visualize(df, chart_type)
                if isinstance(chart_or_error, str):
                    visual_error.append(chart_or_error)
                elif isinstance(chart_or_error, tuple):
                    chart = chart_or_error
                    artifact = Artifact(
                        artifactId=str(uuid4()),
                        name=chart[0],
                        parts=ResponseMessagePart(kind="file", file_url=chart[1]),
                    )
                    charts.append(artifact)
            if visual_error:
                self.errors["visual_error"] = visual_error
            return charts

    def explanation_handler(self, explanation: str):
        return ResponseMessagePart(kind="text", text=explanation)

    def analyse(self, df: DataFrame, data: AIParsedInstruction, metadata: dict):
        metadata = returning_metadata(metadata)
        metadata["status"] = "in_progress"

        if not data.intent or data.intent == "unknown" or data.confidence == 0:
            metadata["summary"] = df.describe(include="all").to_dict()
            metadata["message"] = (
                "Low confidence or unknown intent — default summary only."
            )
            metadata["status"] = "skipped"
            return metadata, [ResponseMessagePart(kind="data", data=metadata)], []

        df = (
            df.pipe(self.focus_columns, data.focus_columns or [])
            .pipe(self.filter_by, data.filters or {})
            .pipe(self.fill, data.fill)
            .pipe(self.drop, data.drop)
            .pipe(self.sorting, data.sort)
        )

        intent_handlers = {
            "summary": lambda: self.handle_summary(df),
            "math": lambda: self.handle_math(df, data.operations),
            "relationship": lambda: self.handle_relationship(
                df, data.operations.relationship
            ),
            "anomaly": lambda: self.handle_anomaly(df),
            "visualization": lambda: self.handle_visualization(df, data.operations),
        }


        message_parts, art = [], []
        if data.analysis_explanation:
            text_explanation = self.explanation_handler(data.analysis_explanation)
            message_parts.append(text_explanation)
            
        for intent in data.intent:
            if intent in intent_handlers:
                result = intent_handlers[intent]()
                if result:
                    if isinstance(result, list):
                        art.extend(result)
                    else:
                        message_parts.append(result)

        metadata.update(
            {
                "processed_columns": df.columns.tolist(),
                "num_rows": len(df),
                "confidence": data.confidence,
                "status": "completed",
            }
        )

        return metadata, message_parts, art
