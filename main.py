import os
from functools import lru_cache
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from psycopg import connect
from sentence_transformers import SentenceTransformer
from models.request_models import QueryRequest
from models.response_models import QueryResponse

load_dotenv()

app = FastAPI(title="Sage300 RAG API")


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    # CPU-friendly model for fast retrieval embeddings
    return SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")


def get_db_connection():
    return connect(
        host=os.getenv("POSTGRES_HOST", "db"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "sage300"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASS", "postgres"),
    )


def to_pgvector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_docs(payload: QueryRequest) -> QueryResponse:
    text = payload.query.strip()
    if not text:
        raise HTTPException(status_code=400, detail="query must not be empty")

    model = get_embedding_model()
    query_embedding = model.encode([text], normalize_embeddings=True)[0].tolist()
    vector_literal = to_pgvector_literal(query_embedding)

    sql = """
        SELECT content
        FROM sage300_docs
        ORDER BY embedding <=> %s::vector
        LIMIT 5;
    """

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (vector_literal,))
                rows = cur.fetchall()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"database error: {exc}") from exc

    matches = [row[0] for row in rows]
    return QueryResponse(query=text, matches=matches)
