from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic.errors import PydanticInvalidForJsonSchema



from data_insight_agent.schema import JSONRPCRequest, JSONRPCResponse
from contextlib import asynccontextmanager, AsyncExitStack
from data_insight_agent.ollama_client import connect_to_ollama
from data_insight_agent.agent_brain import DataInsightEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(connect_to_ollama())

        yield


app = FastAPI(lifespan=lifespan)


@app.post("telex/a2a/data-insight-agent")
async def a2a_endpoint(request: Request):
    try:
        body = request.json()
        rpc_request = JSONRPCRequest(body)
    except (PydanticInvalidForJsonSchema, Exception):
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

    if rpc_request.method == "message/send":
        messages = [rpc_request.params.message]
    elif rpc_request.method == "execute":
        messages = rpc_request.params.messages
        context_id = rpc_request.params.contextId
        task_id = rpc_request.params.taskId
    result, errors = await DataInsightEngine.analyse(messages, context_id, task_id)

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
