from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg_pool import ConnectionPool

from app.config import settings


def _configure(conn: Connection) -> None:
    register_vector(conn)


pool = ConnectionPool(
    conninfo=settings.postgres_dsn,
    min_size=1,
    max_size=5,
    open=False,
    configure=_configure,
)


def open_pool() -> None:
    pool.open(wait=True)


def close_pool() -> None:
    pool.close()
