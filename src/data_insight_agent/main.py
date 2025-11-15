from fastapi import FastAPI, Request, status, Depends
from fastapi.responses import JSONResponse
import time
from typing import Annotated
from httpx import AsyncClient

from data_insight_agent.rpc_schema import (
    JSONRPCRequest,
    JSONRPCResponse,
    ExecuteParams,
    MessageParams,
)
from contextlib import asynccontextmanager, AsyncExitStack
from data_insight_agent.ollama_client import connect_to_ollama, get_ollama
from data_insight_agent.agent_brain import DataInsightEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(connect_to_ollama())

        yield


app = FastAPI(lifespan=lifespan)


@app.post("/telex/a2a/data-insight-agent")
async def a2a_endpoint(
    request: Request, ollama: Annotated[AsyncClient, Depends(get_ollama)]
):
    try:
        body = await request.json()
        rpc_request = JSONRPCRequest(**body)
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: jsonrpc must be '2.0' and id is required",
                },
            },
        )

    messages = []
    context_id = None
    task_id = None

    if rpc_request.method == "message/send" and isinstance(
        rpc_request.params, MessageParams
    ):
        messages = [rpc_request.params.message]
    elif rpc_request.method == "execute" and isinstance(
        rpc_request.params, ExecuteParams
    ):
        messages = rpc_request.params.messages
    else:
        messages = []

    if not messages or not any(m.parts for m in messages):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {
                    "code": -32600,
                    "message": "Invalid or No message body provided",
                },
            },
        )
    data_engine = DataInsightEngine(ollama)
    result, errors = await data_engine.analyse(messages, context_id, task_id)

    if result and errors:
        response = JSONRPCResponse(
            jsonrpc="2.0", id=rpc_request.id, result=result, error=errors
        )
    elif result:
        response = JSONRPCResponse(jsonrpc="2.0", id=rpc_request.id, result=result)
    else:
        response = JSONRPCResponse(
            jsonrpc="2.0",
            id=rpc_request.id,
            error=(
                errors
                if errors
                else {
                    "error": "System Cannot Parse Your Request --- No Message Body or Unclear Request"
                }
            ),
        )
    return response
