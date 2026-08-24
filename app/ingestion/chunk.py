"""Token-aware chunking: pack whole blocks (never split mid-block) up to a token budget,
starting a new chunk on a table or a section-heading change. Every chunk's stored text is
prefixed with its section heading path instead of using sliding-window overlap."""

from dataclasses import dataclass

from transformers import AutoTokenizer

from app.config import settings
from app.ingestion.extract import ExtractedBlock

_tokenizer = None


def _get_tokenizer() -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model_name)
    return _tokenizer


def _count_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text, add_special_tokens=False))


@dataclass
class Chunk:
    chunk_index: int
    chunk_type: str
    page_start: int
    page_end: int
    section_heading: str | None
    content: str
    token_count: int


def chunk_blocks(blocks: list[ExtractedBlock]) -> list[Chunk]:
    chunks: list[Chunk] = []

    heading: str | None = None
    texts: list[str] = []
    page_start: int | None = None
    page_end: int | None = None
    tokens = 0

    def flush(chunk_type: str = "prose") -> None:
        nonlocal texts, page_start, page_end, tokens
        if not texts or page_start is None:
            return
        prefix = f"{heading}\n\n" if heading else ""
        content = prefix + "\n\n".join(texts)
        chunks.append(
            Chunk(
                chunk_index=len(chunks),
                chunk_type=chunk_type,
                page_start=page_start,
                page_end=page_end,
                section_heading=heading,
                content=content,
                token_count=_count_tokens(content),
            )
        )
        texts = []
        page_start = None
        page_end = None
        tokens = 0

    for block in blocks:
        block_tokens = _count_tokens(block.text)

        if block.block_type == "table":
            flush()
            heading = block.heading_path or heading
            texts = [block.text]
            page_start = page_end = block.page
            flush(chunk_type="table")
            continue

        heading_changed = bool(texts) and block.heading_path != heading
        would_overflow = bool(texts) and tokens + block_tokens > settings.chunk_max_tokens

        if heading_changed or would_overflow:
            flush()

        heading = block.heading_path
        texts.append(block.text)
        page_start = page_start if page_start is not None else block.page
        page_end = block.page
        tokens += block_tokens

        if tokens >= settings.chunk_target_tokens:
            flush()

    flush()
    return chunks
