import pandas as pd
import numpy as np
import json
from uuid import uuid4
from datetime import datetime, timezone


from data_insight_agent.schema import AIParsedInstruction
from data_insight_agent.prompt import get_prompt
from data_insight_agent.ollama_client import get_ollama
from data_insight_agent.schema import (
    A2AMessages,
    A2AMessage,
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


class DataInsightEngine:
    @staticmethod
    async def parse_input(message: A2AMessage) -> dict | None:
        data_dict = get_text_and_file(message)
        text, file = data_dict.get("text"), data_dict.get("file")
        if text:
            json_data, actual_text = extract_json_from_text(text)
            if file and json_data:
                return None
            if actual_text and not is_gibberish_or_non_analytical(actual_text):
                data_dict["text"] = actual_text

            if json_data:
                df = pd.DataFrame(json_data)
                data_dict["data"] = df
        if file:
            df = await validate_upload_file(file)
            if df:
                data_dict["data"] = df
        return data_dict

    @staticmethod
    async def extract_metadata(message: A2AMessage) -> dict:
        metadata = {
            "original_text_input": (
                message.parts[-1].text if message.parts[-1].text else None
            )
        }
        data_dict = await DataInsightEngine.parse_input(message)
        df = data_dict.get("data")
        text = data_dict.get("text")
        if text:
            metadata["text_instruction"] = text
        if df:
            shape = df.shape
            stat = df.describe(include="all").to_dict()
            dtypes = DataInsightEngine.get_dtypes(df)
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
        return df, metadata

    @staticmethod
    async def data_ai_model(message: A2AMessage):
        df, metadata = await DataInsightEngine.extract_metadata(message)
        prompt = get_prompt(metadata)
        response = await get_ollama().post(
            "/api/generate",
            json={"model": "qwen2.5:7b", "prompt": prompt, "stream": False},
        )
        if response.status_code == 200:
            raw_text = response.json().get("response", "")
            if raw_text:
                try:
                    parsed_json = json.loads(raw_text)
                    valid_body = AIParsedInstruction(**parsed_json)
                    return df, valid_body, metadata
                except Exception:
                    return None

    @staticmethod
    async def analyse(messages: A2AMessages, context_id: str, task_id: str):
        message = messages[-1] if messages else None
        if not message:
            return None
        context_id = context_id or str(uuid4())
        task_id = task_id or str(uuid4())
        df, ai_data, metadata = await DataInsightEngine.data_ai_model(message)
        analysis = Analysis(context_id=context_id, task_id=task_id)
        updated_metadata, message_parts, artifacts = analysis.analyse(
            df, ai_data, metadata
        )
        message = ResponseA2AMessage(
            kind="message",
            role="system",
            parts=message_parts,
            messageId=str(uuid4()),
            taskId=task_id,
            metadata=updated_metadata,
        )
        messages.append(message)
        task_status = TaskStatus(
            state="completed", timestamp=datetime.now(tz=timezone.utc)
        )
        result = TaskResult(
            id=str(uuid4()),
            contextId=context_id,
            status=task_status,
            artifacts=artifacts,
            history=messages,
            kind="task",
        )
        errors = analysis.errors
        return result, errors

    @staticmethod
    def get_dtypes(df: pd.DataFrame):
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
