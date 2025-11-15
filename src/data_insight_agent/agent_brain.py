import pandas as pd
import numpy as np
from uuid import uuid4
import json
from datetime import datetime, timezone
from httpx import AsyncClient
from json_repair import repair_json

from data_insight_agent.rpc_schema import (
    ResponseMessagePart,
    A2AMessages,
    Artifact,
    TaskResult,
    TaskStatus,
    ResponseA2AMessage,
)
from data_insight_agent.utils import (
    is_gibberish_or_non_analytical,
    extract_json_from_text,
    validate_upload_file,
    get_text_and_file,
)
from data_insight_agent.analysis import Analysis
from data_insight_agent.prompt import get_prompt
from data_insight_agent.config import settings
from data_insight_agent.ai_schema import AIParsedInstruction


class DataInsightEngine:
    def __init__(self, ollama: AsyncClient):
        self.ollama = ollama

    async def parse_input(self, data: dict) -> dict | None:
        text, file = data.get("text"), data.get("file")
        if text:
            json_data, actual_text = extract_json_from_text(text)
            if file and json_data:
                return None
            if actual_text and not is_gibberish_or_non_analytical(actual_text):
                data["text"] = actual_text
            if json_data:
                df = pd.DataFrame(json_data)
                data["data"] = df

        if file:
            df = await validate_upload_file(file)
            if df is not None:
                data["data"] = df
        return data

    def extract_metadata(self, data: dict) -> dict:
        metadata = {}
        df = data.get("data")
        text = data.get("text")
        if text:
            metadata["text_instruction"] = text

        if df is not None and not df.empty:
            shape = df.shape
            stat = df.describe(include="all").to_dict()
            dtypes = self.get_dtypes(df)
            sample = df.head(5).to_dict(orient="records")
            total_missing = int(df.isna().sum().sum())
            percent_missing = round(
                (total_missing / (df.shape[0] * df.shape[1])) * 100, 2
            )
            memory_usage = round(df.memory_usage(deep=True).sum() / (1024**2), 2)
            metadata.update(
                {
                    "columns": dtypes,
                    "shape": shape,
                    "stat": stat,
                    "sample": sample,
                    "total_missing": total_missing,
                    "percent_missing": percent_missing,
                    "memory_usage_mb": memory_usage,
                }
            )
        return metadata

    async def data_interpreter(self, data: dict):
        prompt = get_prompt(data)

        response = await self.ollama.post(
            settings.AI_MODEL_URL,
            json={"model": settings.AI_MODEL, "prompt": prompt, "stream": False},
        )

        if response.status_code == 200:
            raw_text = response.json().get("response", "")
            if raw_text:

                try:
                    parsed_json = json.loads(repair_json(raw_text))

                    valid_body = AIParsedInstruction(**parsed_json)

                    return valid_body
                except Exception:
                    return None

    async def analyse(self, messages: A2AMessages, context_id: str, task_id: str):
        message = messages[-1] if messages else None
        if not message:
            return None, {"error": "No Message Body."}
        metadata = {
            "original_text_input": (
                message.parts[-1].text if message.parts[-1].text else None
            )
        }
        context_id = context_id or str(uuid4())
        task_id = task_id or str(uuid4())
        data = get_text_and_file(message)
        parsed_data = await self.parse_input(data)
        df = parsed_data.get("data", None)
        if df is None or df.empty:
            return None, {"error": "Failed to interpret user request."}
        metadata.update(self.extract_metadata(parsed_data))
        ai_data = await self.data_interpreter(metadata)
        if not ai_data:
            return None, {"error": "Failed to interpret user request."}
        analysis = Analysis(context_id=context_id, task_id=task_id)
        analysed_data = analysis.analyse(df, ai_data, metadata)
        errors = analysis.errors

        local_analysis = self.generate_explanation(analysed_data)

        analysed_data["local_analysis"] = local_analysis
        local_analysis_part = ResponseMessagePart(kind="text", text=local_analysis)

        response_message = ResponseA2AMessage(
            kind="message",
            role="system",
            parts=[local_analysis_part],
            messageId=str(uuid4()),
            taskId=task_id,
        )

        visuals = analysed_data.get("visuals generated", [])
        artifacts = []

        for visual in visuals:
            visual_type, path = list(visual.items())[0]
            file_part = ResponseMessagePart(kind="file", file_url=path)
            artifacts.append(
                Artifact(artifactId=str(uuid4()), name=visual_type, parts=[file_part])
            )

        messages.append(response_message)
        task_status = TaskStatus(
            state="completed", timestamp=datetime.now(tz=timezone.utc).isoformat()
        )

        result = TaskResult(
            id=str(uuid4()),
            contextId=context_id,
            status=task_status,
            artifacts=artifacts,
            history=messages,
            kind="task",
        )

        return result, errors

    def get_dtypes(self, df: pd.DataFrame):
        dtypes_info = {
            "columns": {},
            "groups": {
                "string_cols": [],
                "numerical_cols": [],
                "bool_cols": [],
                "datetime_cols": [],
            },
        }

        for column in df.columns:
            col = df[column]
            if np.issubdtype(col.dtype, np.number):
                dtype = "number"
                dtypes_info["groups"]["numerical_cols"].append(column)
            elif np.issubdtype(col.dtype, np.bool_):
                dtype = "bool"
                dtypes_info["groups"]["bool_cols"].append(column)
            else:
                try:
                    pd.to_datetime(col)
                    dtype = "datetime"
                    dtypes_info["groups"]["datetime_cols"].append(column)
                except Exception:
                    dtype = "string"
                    dtypes_info["groups"]["string_cols"].append(column)

            dtypes_info["columns"][column] = dtype

        return dtypes_info

    def generate_explanation(self, analysed_data):
        metadata = analysed_data.get("metadata") or {}
        rows = metadata.get("num_rows", "?")
        cols = len(metadata.get("processed_columns", []) or [])
        parts = [f"📊 Analyzed {rows} records across {cols} columns."]

        math_stats = analysed_data.get("math") or {}
        for stat in ["mean", "median", "mode", "min", "max", "std"]:
            stat_vals = math_stats.get(stat)
            if stat_vals:
                parts.append(f"\n📈 {stat.title()} values:")
                for col, val in (
                    stat_vals.items() if isinstance(stat_vals, dict) else []
                ):
                    try:
                        if val is None or (isinstance(val, float) and (val != val)):
                            s = "NaN"
                        else:
                            s = f"{float(val):.2f}"
                    except Exception:
                        s = str(val)
                    parts.append(f" • {col}: {s}")

        quantiles = math_stats.get("quantile")
        if quantiles:
            parts.append("\n📊 Quantiles:")
            sample_key = next(iter(quantiles), None)
            sample_val = quantiles.get(sample_key) if sample_key is not None else None
            if isinstance(sample_key, str) and isinstance(sample_val, dict):
                for col, qmap in quantiles.items():
                    if isinstance(qmap, dict):
                        for q, val in qmap.items():
                            try:
                                qf = float(q)
                                q_label = f"{qf * 100:.0f}th percentile"
                            except Exception:
                                q_label = str(q)
                            try:
                                s = f"{float(val):.2f}"
                            except Exception:
                                s = str(val)
                            parts.append(f" • {col} ({q_label}): {s}")
            else:
                for q, colsmap in quantiles.items():
                    try:
                        qf = float(q)
                        q_label = f"{qf * 100:.0f}th percentile"
                    except Exception:
                        q_label = str(q)
                    if isinstance(colsmap, dict):
                        for col, val in colsmap.items():
                            try:
                                s = f"{float(val):.2f}"
                            except Exception:
                                s = str(val)
                            parts.append(f" • {col} ({q_label}): {s}")

        corr = analysed_data.get("correlation") or {}
        if corr:
            parts.append("\n🔗 Correlations (|corr| > 0.6):")
            for col1, sub in corr.items():
                if not isinstance(sub, dict):
                    continue
                for col2, value in sub.items():
                    try:
                        corr_val = float(value)
                    except Exception:
                        continue
                    if col1 != col2 and abs(corr_val) >= 0.6:
                        parts.append(f" • {col1} ↔ {col2}: {corr_val:.2f}")

        reg = analysed_data.get("regression") or {}
        if reg:
            x_col = reg.get("x_column", "x")
            y_col = reg.get("y_column", "y")
            parts.append(f"\n📐 Regression between {x_col} and {y_col}:")
            if "equation" in reg:
                parts.append(f" • Equation: {reg.get('equation')}")
            for key in ("slope", "intercept", "r2"):
                if key in reg:
                    try:
                        parts.append(f" • {key.title()}: {float(reg[key]):.2f}")
                    except Exception:
                        parts.append(f" • {key.title()}: {reg[key]}")

        anomalies = analysed_data.get("zscore_anomalies") or {}
        total = 0
        if anomalies:
            for v in anomalies.values():
                try:
                    total += len(v)
                except Exception:
                    pass
        if total > 0:
            parts.append(f"\n⚠️ Detected {total} anomaly{'ies' if total != 1 else ''}.")

        visuals = analysed_data.get("visuals generated") or []
        if visuals:
            parts.append(
                f"\n🖼️ Generated {len(visuals)} visualization{'s' if len(visuals) != 1 else ''}."
            )

        analysis = "\n".join(parts)
        return analysis
