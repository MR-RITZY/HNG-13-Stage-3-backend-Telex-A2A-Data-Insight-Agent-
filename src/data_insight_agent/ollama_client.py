from httpx import AsyncClient, Timeout
from contextlib import asynccontextmanager

from data_insight_agent.config import settings


ollama_client: AsyncClient


@asynccontextmanager
async def connect_to_ollama():
    global ollama_client
    ollama_client = AsyncClient(
        base_url=settings.OLLAMA_URL, timeout=Timeout(300.0, connect=5.0)
    )
    try:
        print("Testing connection to Ollama...")
        response = await ollama_client.post(
            settings.AI_MODEL_URL,
            json={"model": settings.AI_MODEL, "prompt": "ping", "stream": False},
        )
        response.raise_for_status()
        print("✅ Ollama connected successfully!")
        yield
    except Exception as e:
        print(f"Ollama connection failed: {e}")
        yield
    finally:
        await ollama_client.aclose()


def get_ollama() -> AsyncClient:
    return ollama_client
