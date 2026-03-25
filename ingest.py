import argparse
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from sentence_transformers import SentenceTransformer


def load_documents(docs_dir: Path):
    documents = []
    supported_text_suffixes = {".txt", ".md", ".rst"}

    for path in docs_dir.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            documents.extend(PyPDFLoader(str(path)).load())
        elif suffix in supported_text_suffixes:
            documents.extend(TextLoader(str(path), encoding="utf-8").load())

    return documents


def escape_sql_text(value: str) -> str:
    return value.replace("'", "''")


def to_pgvector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


def build_sql(chunks: list[str], vectors: list[list[float]]) -> str:
    statements = [
        "CREATE EXTENSION IF NOT EXISTS vector;",
        """
CREATE TABLE IF NOT EXISTS sage300_docs (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL
);
        """.strip(),
        "TRUNCATE TABLE sage300_docs;",
    ]

    for chunk, vector in zip(chunks, vectors):
        escaped_chunk = escape_sql_text(chunk)
        vector_literal = to_pgvector_literal(vector)
        statements.append(
            "INSERT INTO sage300_docs (content, embedding) "
            f"VALUES ('{escaped_chunk}', '{vector_literal}'::vector);"
        )

    return "\n".join(statements) + "\n"


def run_ingestion(docs_path: str, output_sql: str, chunk_size: int, chunk_overlap: int):
    docs_dir = Path(docs_path)
    if not docs_dir.exists():
        raise FileNotFoundError(f"documents path does not exist: {docs_dir}")

    documents = load_documents(docs_dir)
    if not documents:
        raise RuntimeError(
            f"no supported documents found in {docs_dir} (.pdf, .txt, .md, .rst)"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = splitter.split_documents(documents)
    chunks = [doc.page_content.strip() for doc in split_docs if doc.page_content.strip()]
    if not chunks:
        raise RuntimeError("no non-empty chunks produced from documents")

    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    embeddings = model.encode(chunks, normalize_embeddings=True).tolist()

    sql = build_sql(chunks, embeddings)
    output_file = Path(output_sql)
    os.makedirs(output_file.parent, exist_ok=True)
    output_file.write_text(sql, encoding="utf-8")

    print(f"Loaded docs: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Wrote SQL to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Sage 300 docs into SQL for pgvector")
    parser.add_argument(
        "--docs-path",
        default="training_documents",
        help="Folder containing .pdf/.txt/.md/.rst documentation files",
    )
    parser.add_argument(
        "--output-sql",
        default="db/init/02_sage300_docs.sql",
        help="Path to generated SQL file",
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)

    args = parser.parse_args()
    run_ingestion(
        docs_path=args.docs_path,
        output_sql=args.output_sql,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
