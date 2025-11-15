from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    OLLAMA_URL: str
    AI_MODEL: str
    AI_MODEL_URL: str
    B2_KEY_ID: str
    B2_APPLICATION_KEY: str
    B2_BUCKET: str
    B2_ENDPOINT: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()