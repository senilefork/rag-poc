import argparse
import os
import shutil
from pathlib import Path

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from markdown_nodes import build_nodes

DB_PATH = os.environ.get("DB_PATH", "./local_chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "avino_history")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")


def reset_db(path: str) -> None:
    """Delete the contents of the Chroma persistence dir, including orphaned segments.

    `client.delete_collection` only drops the live collection; HNSW segment
    directories from earlier seeds stay on disk unreferenced. Wiping the store is
    the only way to guarantee no stale vectors survive.

    Children are removed but the directory itself is kept, because DB_PATH is a
    bind-mounted volume in docker-compose and a mount point cannot be unlinked.
    """
    db_dir = Path(path).expanduser().resolve()
    if db_dir == Path(db_dir.anchor) or db_dir == Path.home():
        raise SystemExit(f"refusing to delete {db_dir}")
    if not db_dir.exists():
        return
    if not db_dir.is_dir():
        raise SystemExit(f"DB_PATH is not a directory: {db_dir}")
    if not (db_dir / "chroma.sqlite3").exists() and any(db_dir.iterdir()):
        raise SystemExit(f"{db_dir} is not a Chroma store; refusing to delete")

    for child in db_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    print(f"Cleared stale Chroma store at {db_dir}")


def main(pdf_path: str, *, keep: bool = False) -> None:
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    nodes = build_nodes(pdf_path)

    if keep:
        print(f"Keeping existing store at {DB_PATH}")
    else:
        reset_db(DB_PATH)

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=Settings.embed_model,
    )
    print(f"Seeded {len(nodes)} nodes into collection '{COLLECTION_NAME}' at {DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the Chroma store from a PDF.")
    parser.add_argument("pdf_path", help="path to the PDF to parse")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="append to the existing store instead of deleting it first",
    )
    args = parser.parse_args()
    main(args.pdf_path, keep=args.keep)