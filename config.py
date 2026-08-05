"""Centralized configuration loaded from environment variables / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings sourced from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM provider
    llm_provider: str = "ollama"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    groq_api_key: str = ""
    groq_model: str = "llama3-70b-8192"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"

    # Vector store
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "medassist_drug_labels"
    chroma_host: str = ""
    chroma_port: int = 8200

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Ingestion
    openfda_api_key: str = ""
    openfda_limit: int = 100

    # RAG
    retrieval_top_k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 50

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Frontend
    api_base_url: str = "http://localhost:8000"


settings = Settings()
