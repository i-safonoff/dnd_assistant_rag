# D&D Assistant RAG

![Python](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![pgvector](https://img.shields.io/badge/vector%20store-pgvector-336791?logo=postgresql&logoColor=white)
![vLLM](https://img.shields.io/badge/serving-vLLM-orange)
![Docker](https://img.shields.io/badge/docker-compose-2496ED?logo=docker&logoColor=white)

Local, fully offline RAG system over D&D rulebooks/campaign PDFs, answering questions with a
locally-hosted Qwen model. No calls to hosted LLM APIs.

## Architecture

- **FastAPI app** (`app/`) — runs on the host via Poetry. Extracts and chunks PDFs, embeds text
  with `BAAI/bge-m3` (in-process, CPU), stores chunks + vectors in Postgres, retrieves top-k
  matches, and asks a locally-served Qwen model to answer using only those excerpts.
- **Postgres + pgvector** (Docker) — vector store, browsable with any Postgres client (DBeaver:
  connect to `localhost:5432`, db `dnd_rag`, user `dnd_rag`).
- **vLLM** (Docker) — serves `Qwen/Qwen2.5-14B-Instruct-AWQ` behind an OpenAI-compatible API on
  `localhost:8000`.

vLLM alone uses ~15.6GB of the 16GB VRAM budget at the configured `--gpu-memory-utilization=0.85`
(the legacy V1 model runner it's forced onto — see below — has more overhead than expected), so
the embedding model runs on CPU instead of fighting it for GPU memory. Ingestion is a one-off
batch step and query-time embedding is a single short string, so CPU is fine here.

**Known env quirk:** on Windows/WSL2, vLLM's default "Model Runner V2" crashes with
`RuntimeError: UVA is not available` (CUDA Unified Virtual Addressing isn't available under
WSL2's pinned-memory restrictions). `docker-compose.yml` sets `VLLM_USE_V2_MODEL_RUNNER=0` to
force the older, WSL2-compatible runner.

## Prerequisites

- Docker Desktop with the WSL2 backend and GPU support (`docker run --rm --gpus all
  nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` should print your GPU).
- Poetry, and Python 3.13 available on the system (the venv is pinned to 3.13, not whatever
  newer Python may be default on PATH — ML wheel coverage lags behind new CPython releases).

## Setup

```bash
poetry env use "C:\Users\<you>\AppData\Local\Programs\Python\Python313\python.exe"
poetry install

cp .env.example .env   # adjust if you change ports/passwords

docker compose up -d postgres vllm
# first vllm start downloads the model (~9GB) — check progress with:
docker compose logs -f vllm
```

Once `docker compose logs vllm` shows the server is up and `curl http://localhost:8000/v1/models`
responds, start the API (see port note below).

## Usage

vLLM already occupies port 8000, so run the FastAPI app on 8001:
```bash
poetry run uvicorn app.main:app --reload --port 8001
```

**Ingest a PDF:**
```bash
curl -F "file=@data/pdfs/players-handbook.pdf" http://localhost:8001/ingest
```

**Ask a question:**
```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What happens on a critical hit?"}'
```

**Health check:** `GET /health` — reports Postgres and vLLM connectivity.

Interactive docs: `http://localhost:8001/docs`.

## Known limitations

- PDF extraction (`app/ingestion/extract.py`) uses heuristics for column detection (page
  horizontal midpoint) and headings (relative font size). Works well for typical two-column
  rulebook layouts; irregular layouts may chunk imperfectly.
- No OCR — scanned/image-only PDFs will yield near-empty pages (flagged in the `/ingest`
  response as `low_text_pages`).
- No schema migrations — the Postgres schema is a single init SQL file
  (`docker/postgres/init/01_init.sql`), applied once on first container start. Schema changes
  require recreating the `pgdata` volume or a manual `ALTER`.

## Legacy

`legacy_qwen27b_model/` (originally `triton/models/Qwen3.6-27B-ModelScope/`) contains a
previously-downloaded 27B model that turned out to be too large for a 16GB GPU and too new an
architecture for reliable serving-engine support. It's unreferenced by anything in this repo
(and gitignored) — left in place since it's ~54GB and expensive to re-download; delete manually
if you want the disk space back.

Renamed from `triton/` because that name collided with the `triton` PyPI package (PyTorch's GPU
kernel compiler): Python resolves a same-named local directory in the project root before
site-packages, so every `torch.compile`/dynamo code path silently imported our model folder
instead of the real library and crashed with `AttributeError: module 'triton' has no attribute
'language'`. Avoid naming any top-level project directory after an installed package.
