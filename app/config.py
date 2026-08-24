from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    postgres_dsn: str = "postgresql://dnd_rag:dnd_rag_local_dev@localhost:5432/dnd_rag"

    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "qwen2.5-14b-awq"

    embedding_model_name: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    chunk_target_tokens: int = 500
    chunk_max_tokens: int = 600
    default_top_k: int = 5


settings = Settings()
