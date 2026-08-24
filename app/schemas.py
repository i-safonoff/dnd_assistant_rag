from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_id: int
    filename: str
    chunk_count: int
    low_text_pages: list[int] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class Citation(BaseModel):
    document_id: int
    chunk_id: int
    filename: str
    page_start: int | None
    page_end: int | None
    section_heading: str | None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


class HealthResponse(BaseModel):
    vllm: str
    postgres: str
