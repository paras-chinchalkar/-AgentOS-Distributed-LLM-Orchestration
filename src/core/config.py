from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os
load_dotenv()
_groq_key = os.getenv("GROQ_API_KEY")
if _groq_key:
    os.environ["groq_api_key"] = _groq_key


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    groq_api_key: str = "your-groq-api-key"
    groq_model: str = "llama-3.3-70b-versatile"

    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_stream_name: str = "agent_tasks"
    redis_consumer_group: str = "worker_group"

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/agentdb"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = int(os.getenv("PORT", "8000"))  # Railway sets PORT env var
    log_level: str = "INFO"


settings = Settings()
