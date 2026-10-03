import json
import os
import sys
import urllib.request

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

DB_PATH = os.environ.get("DB_PATH", "./local_chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "avino_history")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5-1.5b-instruct")
TOP_K = int(os.environ.get("TOP_K", "5"))

PROMPT_TEMPLATE = (
    "You are a helpful assistant. Answer the question using only the context "
    "provided below. If the context does not contain the answer, say so.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def retrieve(query_text: str):
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=Settings.embed_model
    )
    return index.as_retriever(similarity_top_k=TOP_K).retrieve(query_text)


def generate(prompt: str) -> str:
    body = json.dumps(
        {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{LLM_BASE_URL}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def main(question: str) -> None:
    nodes = retrieve(question)
    context = "\n\n".join(
        f"[{i + 1}] ({node.metadata.get('section', '?')}) {node.get_content()}"
        for i, node in enumerate(nodes)
    )
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    print("---------------- answer ----------------")
    print(generate(prompt))
    print("---------------- sources ----------------")
    for i, node in enumerate(nodes, 1):
        print(f"[{i}] {node.metadata}")
        print(node.get_content()[:300])
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python query.py "your question"')
    main(sys.argv[1])