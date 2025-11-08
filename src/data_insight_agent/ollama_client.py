from httpx import AsyncClient
from contextlib import asynccontextmanager

from data_insight_agent.config import settings


ollama_client: AsyncClient

@asynccontextmanager
async def connect_to_ollama():
        global ollama_client
        ollama_client = AsyncClient(base_url=settings.OLLAMA_URL)

        yield
        await AsyncClient.aclose()

async def get_ollama():
        return ollama_client