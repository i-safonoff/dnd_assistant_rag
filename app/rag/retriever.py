from psycopg.rows import dict_row

from app.db.session import pool
from app.rag.embedding import embed_query

_SEARCH_SQL = """
    SELECT c.id AS chunk_id, c.document_id, c.page_start, c.page_end,
           c.section_heading, c.content, c.chunk_type,
           d.filename,
           1 - (c.embedding <=> %s) AS score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    ORDER BY c.embedding <=> %s
    LIMIT %s
"""


def search(question: str, top_k: int) -> list[dict]:
    query_vec = embed_query(question)
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(_SEARCH_SQL, (query_vec, query_vec, top_k))
            return cur.fetchall()
