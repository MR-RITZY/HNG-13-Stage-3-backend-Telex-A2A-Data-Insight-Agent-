from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    OLLAMA_URL: str
    AI_MODEL: str
    AI_MODEL_URL: str
    MINIO_ENDPOINT: str 
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET:str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()