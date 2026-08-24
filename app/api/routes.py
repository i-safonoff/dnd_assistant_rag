import tempfile
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, UploadFile
from psycopg.rows import dict_row

from app.config import settings
from app.db.session import pool
from app.ingestion.chunk import chunk_blocks
from app.ingestion.extract import extract_pdf
from app.rag.embedding import embed_texts
from app.rag.generator import generate_answer
from app.rag.retriever import search
from app.schemas import Citation, HealthResponse, IngestResponse, QueryRequest, QueryResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    postgres_status = "ok"
    try:
        with pool.connection() as conn:
            conn.execute("SELECT 1")
    except Exception as exc:
        postgres_status = f"error: {exc}"

    vllm_status = "ok"
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{settings.vllm_base_url}/models")
            resp.raise_for_status()
    except Exception as exc:
        vllm_status = f"error: {exc}"

    return HealthResponse(vllm=vllm_status, postgres=postgres_status)


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        sha256, page_count, blocks, low_text_pages = extract_pdf(tmp_path)
        chunks = chunk_blocks(blocks)

        with pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT id FROM documents WHERE sha256 = %s", (sha256,))
                existing = cur.fetchone()
                if existing:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Document already ingested (id={existing['id']})",
                    )

                cur.execute(
                    "INSERT INTO documents (filename, sha256, page_count) "
                    "VALUES (%s, %s, %s) RETURNING id",
                    (file.filename, sha256, page_count),
                )
                document_id = cur.fetchone()["id"]

                if chunks:
                    embeddings = embed_texts([c.content for c in chunks])
                    for chunk, embedding in zip(chunks, embeddings):
                        cur.execute(
                            """
                            INSERT INTO chunks
                                (document_id, chunk_index, chunk_type, page_start, page_end,
                                 section_heading, content, token_count, embedding)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                document_id,
                                chunk.chunk_index,
                                chunk.chunk_type,
                                chunk.page_start,
                                chunk.page_end,
                                chunk.section_heading,
                                chunk.content,
                                chunk.token_count,
                                embedding,
                            ),
                        )
            conn.commit()

        return IngestResponse(
            document_id=document_id,
            filename=file.filename,
            chunk_count=len(chunks),
            low_text_pages=low_text_pages,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    top_k = request.top_k or settings.default_top_k
    chunks = search(request.question, top_k)
    if not chunks:
        return QueryResponse(answer="No ingested documents to search yet.", citations=[])

    answer = generate_answer(request.question, chunks)
    citations = [
        Citation(
            document_id=c["document_id"],
            chunk_id=c["chunk_id"],
            filename=c["filename"],
            page_start=c["page_start"],
            page_end=c["page_end"],
            section_heading=c["section_heading"],
        )
        for c in chunks
    ]
    return QueryResponse(answer=answer, citations=citations)
