from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field, model_validator
from datetime import datetime, timezone
from uuid import uuid4
from fastapi import UploadFile


class RequestMessagePart(BaseModel):
    kind: Literal["text", "file"]
    text: Optional[str] = None
    file: Optional[UploadFile] = None


class ResponseMessagePart(BaseModel):
    kind: Literal["text", "file"]
    text: Optional[str] = None
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
    configuration: Optional[MessageConfiguration] = None


class ExecuteParams(BaseModel):
    contextId: Optional[str] = None
    taskId: Optional[str] = None
    messages: A2AMessages


class JSONRPCRequest(BaseModel):
    jsonrpc: Literal["2.0"]
    id: str
    method: Literal["message/send", "execute"]
    params: MessageParams | ExecuteParams

    @model_validator(mode="after")
    def validate_json(self):
        if self.method == "message/send":
            self.params: MessageParams
        else:
            self.params: ExecuteParams
        return self


class TaskStatus(BaseModel):
    state: Literal["working", "completed", "input-required", "failed"]
    timestamp: str = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    message: Optional[ResponseA2AMessage] = None


class Artifact(BaseModel):
    artifactId: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    parts: List[ResponseMessagePart]


class TaskResult(BaseModel):
    id: str
    contextId: str
    status: TaskStatus
    artifacts: Optional[List[Artifact]] = []
    history: A2AMessages = []
    kind: Literal["task"] = "task"


class JSONRPCResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str
    result: Optional[TaskResult] = None
    error: Optional[Dict[str, Any]] = None