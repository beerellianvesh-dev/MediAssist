"""Pydantic request/response models for the MedAssist RAG API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The medical question to answer.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve.")
    provider: str = Field(default="ollama", description="LLM provider: ollama, groq, openai, anthropic.")


class SourceCitation(BaseModel):
    drug_name: str
    section_name: str
    source_url: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    confidence: float = Field(..., ge=0.0, le=1.0)
    response_time_ms: float
    model_used: str


class HealthResponse(BaseModel):
    status: str
    vector_db_count: int
    model_status: str


class SourcesResponse(BaseModel):
    drugs: list[str]
    total_documents: int


class StatsResponse(BaseModel):
    collection_name: str
    document_count: int
