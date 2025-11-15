from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator


Corr = Literal["pearson", "kendall", "spearman"]

Correlation = Corr | List[Corr]


class Regression(BaseModel):
    col_x: str
    col_y: str


class Quantile(BaseModel):
    percentiles: Optional[float | List[float]] = None


Math = Literal["sum", "min", "max", "mean", "median", "mode", "count", "std", "var"]

Maths = List[Union[Math, Quantile]]


Charts = Literal["bar", "line", "pie", "hist", "box", "heatmap", "scatter"]
Visuals = List[Charts] | Charts


Intent = List[
    Literal[
        "summary",
        "math",
        "correlation",
        "regression",
        "anomaly",
        "visualization",
        "unknown",
    ]
]


class Operation(BaseModel):
    summary: Optional[Literal["summary"]] = None
    math: Optional[Maths] = None
    correlation: Optional[Correlation | List[Correlation]] = None
    regression: Optional[Regression] = None
    anomaly: Optional[Literal["zscore"]] = None
    visualization: Optional[Visuals] = None
    extra_data: Optional[Dict[str, Any]] = None

    @field_validator("*", mode="before")
    def normalize_empty(cls, v):
        if v in [None, "", 0, 0.0, False, [], {}]:
            return None
        return v


class AIParsedInstruction(BaseModel):
    intent: Intent = Field(
        ..., description="List of analytic tasks (intents) required for analysis."
    )
    operations: Optional[Operation] = None
    focus_columns: Optional[List[str]] = None
    group_by: Optional[List[str]] = None
    drop: Optional[Literal["drop null", "drop duplicates"]] = None
    fill: Optional[Union[Literal["ffill", "bfill"], str]] = None
    sort: Optional[List[Union[Literal["ascending", "descending"], str]]] = None
    filters: Optional[Dict[str, Any]] = None
    analysis_explanation: Optional[str] = Field(
        default=None, description="Comprehensive explanation of the analysis."
    )
    confidence: Optional[float] = 0.0

    @field_validator("*", mode="before")
    def normalize_empty(cls, v):
        if v in [None, "", 0, 0.0, False, [], {}]:
            return None
        return v
