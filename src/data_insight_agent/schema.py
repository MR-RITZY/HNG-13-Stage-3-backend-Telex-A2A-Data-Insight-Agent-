from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from uuid import uuid4
from fastapi import UploadFile

class RequestMessagePart(BaseModel):
    kind: Literal["text", "file"]
    text: Optional[str] = None
    file: Optional[UploadFile] = None

class ResponseMessagePart(BaseModel):
    kind: Literal["text", "data", "file"]
    text: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    file_url: Optional[str] = None

class RequestA2AMessage(BaseModel):
    kind: Literal["message"] = "message"
    role: Literal["user", "agent"]
    parts: List[RequestMessagePart]
    messageId: str = Field(default_factory=lambda: str(uuid4()))
    taskId: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ResponseA2AMessage(BaseModel):
    kind: Literal["message"] = "message"
    role: Literal["system"]
    parts: List[ResponseMessagePart]
    messageId: str = Field(default_factory=lambda: str(uuid4()))
    taskId: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

A2AMessage = Union[RequestA2AMessage, ResponseA2AMessage]
A2AMessages = List[A2AMessage]


class PushNotificationConfig(BaseModel):
    url: str
    token: Optional[str] = None
    authentication: Optional[Dict[str, Any]] = None

class MessageConfiguration(BaseModel):
    blocking: bool = True
    acceptedOutputModes: List[str] = ["text/plain", "image/png", "image/svg+xml"]
    pushNotificationConfig: Optional[PushNotificationConfig] = None

class MessageParams(BaseModel):
    message: RequestA2AMessage
    configuration: MessageConfiguration = Field(default_factory=MessageConfiguration)

class ExecuteParams(BaseModel):
    contextId: Optional[str] = None
    taskId: Optional[str] = None
    messages: A2AMessages

class JSONRPCRequest(BaseModel):
    jsonrpc: Literal["2.0"]
    id: str
    method: Literal["message/send", "execute"]
    params: MessageParams | ExecuteParams

class TaskStatus(BaseModel):
    state: Literal["working", "completed", "input-required", "failed"]
    timestamp: str = Field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    message: Optional[ResponseA2AMessage] = None

class Artifact(BaseModel):
    artifactId: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    parts: List[ResponseMessagePart]

class TaskResult(BaseModel):
    id: str
    contextId: str
    status: TaskStatus
    artifacts: List[Artifact] = []
    history: A2AMessages = []
    kind: Literal["task"] = "task"

class JSONRPCResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str
    result: Optional[TaskResult] = None
    error: Optional[Dict[str, Any]] = None



class Correlation(BaseModel):
    correlation: Literal["pearson", "kendall", "spearman"]


class Regression(BaseModel):
    col_x: str
    col_y: str


class Quantile(BaseModel):
    percentiles: Optional[float | List[float]] = 0.25


class Operation(BaseModel):
    summary: Optional[Literal["describe"]] = None
    math: Optional[
        List[
            Literal[
                "sum",
                "min",
                "max",
                "mean",
                "median",
                "mode",
                "count",
                "std",
                "var",
                "quantile",
            ]
        ]
    ] = None
    quantile: Optional[Quantile] = None
    relationship: Optional[List[Union[Correlation, Regression]]] = None
    anomaly: Optional[Literal["zscore"]] = None
    visualization: Optional[
        List[Literal["bar", "line", "pie", "hist", "box", "heatmap", "scatter"]]
        | Literal["bar", "line", "pie", "hist", "box", "heatmap", "scatter"]
    ] = None


class AIParsedInstruction(BaseModel):
    intent: List[
        Literal[
            "summary", "math", "relationship", "anomaly", "visualization", "unknown"
        ]
    ] = Field(
        ...,
        description="""The list of analytic tasks required to perfectly analyse and 
              give detail result for the user request.""",
    )

    operations: Optional[Operation] = None
    focus_columns: Optional[List[str]] = None
    group_by: Optional[List[str]] = None
    drop: Optional[Literal["drop null", "drop duplicates"]] = None
    fill: Optional[Literal["ffill", "bfill"] | str] = None
    sort: Optional[List[Literal["ascending", "descending"] | str]] = None
    filters: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
