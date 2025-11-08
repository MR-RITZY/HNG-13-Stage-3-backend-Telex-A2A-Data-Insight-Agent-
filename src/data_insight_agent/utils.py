from functools import lru_cache
import spacy
import re
import json
import io
import pandas as pd
from fastapi import UploadFile
from data_insight_agent.schema import RequestA2AMessage


nlp = spacy.load("en_core_web_sm")

ANALYTIC_INTENTS = {
    "summary": "Summarize or describe data using descriptive statistics.",
    "math": "Compute or calculate mathematical functions such as totals, averages, or distributions.",
    "relationship": "Find correlations, regressions, or relationships between variables.",
    "anomaly": "Identify outliers or unusual patterns.",
    "visualization": "Create visual representations of data (charts, plots, trends).",
}

ANALYTICAL_KEYWORDS = {
    "summary": "describe",
    "math": [
        "sum",
        "min",
        "max",
        "mean",
        "median",
        "mode",
        "count",
        "std",
        "quantile",
        "var",
    ],
    "relationship": {
        "correlation": ["pearson", "kendall", "spearman"],
        "regression": ["x_column", "y_column"],
    },
    "anomaly": "zscore",
    "visualization": ["bar", "line", "pie", "hist", "box", "heatmap", "scatter"],
}

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}
MAX_FILE_SIZE = 20 * 1024 * 1024


@lru_cache()
def get_nlp():
    return spacy.load("en_core_web_sm")


nlp = get_nlp()


analytical_docs = [
    nlp(word) for words in ANALYTICAL_KEYWORDS.values() for word in words
]


def is_gibberish_or_non_analytical(text: str) -> bool:
    text = text.strip()
    if not text or len(text) < 3:
        return True

    alpha_ratio = sum(c.isalpha() for c in text) / len(text)
    if alpha_ratio < 0.5:
        return True

    vowel_ratio = sum(c.lower() in "aeiou" for c in text) / max(
        sum(c.isalpha() for c in text), 1
    )
    if vowel_ratio < 0.25:
        return True

    lower_text = text.lower()
    if any(kw.text in lower_text for kw in analytical_docs):
        return False

    doc = nlp(text)
    if not doc.has_vector or doc.vector_norm == 0:
        return True

    unknown_ratio = sum(1 for token in doc if not token.has_vector) / max(len(doc), 1)
    if unknown_ratio > 0.5:
        return True

    max_similarity = 0
    for token in doc:
        if token.has_vector and token.vector_norm > 0:
            for kw in analytical_docs:
                if kw.vector_norm > 0:
                    sim = (token.vector @ kw.vector) / (
                        token.vector_norm * kw.vector_norm
                    )
                    max_similarity = max(max_similarity, sim)

    return max_similarity < 0.45


def is_valid_json_data(data: dict) -> bool:
    if not isinstance(data, dict) or not data:
        return False
    lengths = []
    for value in data.values():
        if isinstance(value, dict):
            return False
        if not isinstance(value, (list, tuple)):
            return False
        lengths.append(len(value))

    return all(length == lengths[0] for length in lengths)


def extract_json_from_text(text: str):
    matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)

    if not matches:
        return None, text.strip()

    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict) and is_valid_json_data(data):
                remaining_text = text.replace(match, "").strip()
                return data, remaining_text
        except json.JSONDecodeError:
            continue

    return None, text.strip()


async def validate_upload_file(upload_file: UploadFile):
    filename = upload_file.filename
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return None
    content = await upload_file.read()
    if len(content) > MAX_FILE_SIZE:
        return None

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(io.BytesIO(content))
        elif filename.endswith(".json"):
            try:
                df = pd.read_json(io.BytesIO(content))
            except Exception:
                json_data = json.loads(content.decode("utf-8"))
                df = pd.DataFrame(json_data)
        else:
            return None
        try:
            if df.empty:
                return None
        except Exception:
            return None
        return df
    except Exception:
        return None


def get_text_and_file(A2AMessage: RequestA2AMessage):
    data_dict = {}
    for part in A2AMessage.parts:
        if part.kind == "text" and part.text:
            data_dict["text"] = part.text
        if part.kind == "file" and part.file:
            data_dict["file"] = part.file
    return data_dict


import numpy as np


def simple_linear_regression(df: pd.DataFrame, x_col: str, y_col: str):
    x = df[x_col].values
    y = df[y_col].values
    if len(x) == 0 or len(y) == 0:
        return {"error": "Empty data for regression."}
    z = np.polyfit(x, y, 1)
    slope, intercept = z
    equ = np.poly1d(z)
    return {
        "equation": equ,
        "slope": slope,
        "intercept": intercept,
    }


def returning_metadata(metadata: dict) -> dict:
        for key in ("original_text_input", "text_instruction"):
            metadata.pop(key, None)
        return metadata
