from datetime import datetime

from pydantic import BaseModel


class Document(BaseModel):
    id: int
    filename: str
    title: str | None
    sha256: str
    page_count: int | None
    ingested_at: datetime


class Chunk(BaseModel):
    id: int
    document_id: int
    chunk_index: int
    chunk_type: str
    page_start: int | None
    page_end: int | None
    section_heading: str | None
    content: str
    token_count: int | None
