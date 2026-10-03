# Avino History — Local RAG (all Docker, all open source)

Ask questions about `avino_history.pdf` via local retrieval + a local LLM. No cloud,
no host-side package installs — everything runs in Docker.

```
avino_history.pdf ──> docling (pypdfium backend) ──> DoclingDocument
                                                     │
                            text chunks (300 words / 50 overlap) + whole-table nodes
                                                     │
                                 HuggingFace BGE embeddings
                                                     │
                                                   ChromaDB
                                                     │
  question ──> top-k retrieval ──> llama.cpp server (Qwen2.5-1.5B) ──> answer + sources
```

## Prerequisites

- Docker with Docker Compose v2

## Run

```bash
docker compose build

docker compose run --rm app python seed.py pdfs/avino_history.pdf

docker compose run --rm app python query.py "When was the Avino deposit discovered and by whom?"

docker compose run --rm app python main.py pdfs/avino_history.pdf
```

Seeding deletes `local_chroma_db/` and rebuilds it, so a re-seed never mixes in
vectors from an older parse. Pass `--keep` to append to the existing store
instead, which is what you want when adding a PDF incrementally without paying
to re-embed the ones already there.

The `llm` service starts automatically when you run the app service; it exposes the
model on `http://llm:8080`. Embeddings and Chroma run in-process in the `app` container.

## Tests

```bash
docker compose run --rm app pytest
```

`test_markdown_nodes.py` covers the word-overlap splitter and validates the parsed
Avino table against `questions.json` — every cell value across all six periods. That
check is deliberate: these PDFs contain no extractable text, so every table value is
OCR output and can silently corrupt.

## Files

| File | Role |
|------|------|
| `markdown_nodes.py` | Module: PDF → docling → overlapping text chunks + whole-table nodes, with `source`/`page`/`type`/`section`/`chunk_id` metadata |
| `seed.py` | Nodes → embeddings → Chroma (collection `avino_history`; deletes the old store first unless `--keep`) |
| `query.py` | Retrieve top-k from Chroma, ask the llama.cpp container, print answer + sources |
| `main.py` | Debug: prints the parsed nodes for a given PDF |
| `test_markdown_nodes.py` | Splitter unit tests + table values validated against `questions.json` |
| `Dockerfile` | App image (parsing, embeddings, Chroma, retrieval) |
| `Dockerfile.llm` | llama.cpp server image with the Qwen GGUF baked in |
| `docker-compose.yml` | `app` + `llm` services, shared Chroma volume, network |

## Configuration (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `./local_chroma_db` | Chroma persistence location (volume-mounted) |
| `COLLECTION_NAME` | `avino_history` | Chroma collection name |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `LLM_BASE_URL` | `http://llm:8080/v1` | llama.cpp server endpoint |
| `LLM_MODEL` | `qwen2.5-1.5b-instruct` | Model name sent to the server |
| `TOP_K` | `5` | Retrieved context chunks |

## Swap the LLM

Edit `Dockerfile.llm` to fetch a different GGUF and update the model path + `LLM_MODEL`
in `docker-compose.yml`, then `docker compose build llm`.
