# Modular RAG Pipeline

A modular, end-to-end Retrieval-Augmented Generation (RAG) system that ingests multi-document corpora (JSONL/TXT directories, single JSONL files, or PDFs), persists dense embeddings with ChromaDB, builds a sparse BM25 index, performs hybrid retrieval with Reciprocal Rank Fusion (RRF), re-ranks with a cross-encoder, and generates grounded answers with citations using an LLM.

## Highlights
- Modular architecture: ingestion, retrieval, reranking, generation are separated into focused modules.
- Persistent vector store: ChromaDB with SentenceTransformer embeddings.
- Sparse retrieval: in-memory BM25 with on-disk cache.
- Hybrid retrieval: dense + BM25 fused via RRF.
- Reranking: cross-encoder selection with threshold-based abstention.
- Answering: grounded, citation-style responses; abstains when evidence is insufficient.

## Repository structure (key files)
- [rag.py](rag.py) — Backward-compatible CLI shim that delegates to the rag/ package.
- [requirements.txt](requirements.txt) — Python dependencies.
- [data/2wikimqa.jsonl](data/2wikimqa.jsonl) — Example dataset.
- [rag/cli.py](rag/cli.py) — CLI runner (build and query modes).
- [rag/config.py](rag/config.py) — Immutable configuration (replaces runtime globals).
- [rag/constants.py](rag/constants.py) — Shared constants (e.g., abstention message).
- [rag/ingestion/parser.py](rag/ingestion/parser.py) — 2Wiki passage splitter + generic record parser.
- [rag/ingestion/chunking.py](rag/ingestion/chunking.py) — Sentence chunker builder for PDFs.
- [rag/ingestion/persist.py](rag/ingestion/persist.py) — Ingestion orchestration and persistence (Chroma + BM25 cache).
- [rag/retrieval/bm25.py](rag/retrieval/bm25.py) — BM25, tokenization, cache loader.
- [rag/retrieval/expansion.py](rag/retrieval/expansion.py) — LLM-based multi-query expansion.
- [rag/retrieval/fusion.py](rag/retrieval/fusion.py) — RRF + hybrid retrieval orchestration.
- [rag/rerank/cross_encoder.py](rag/rerank/cross_encoder.py) — Cross-encoder reranking and selection.
- [rag/generation/answer.py](rag/generation/answer.py) — Answer synthesis with citations.

## Prerequisites
- Python 3.10–3.12 recommended.
- An OpenAI API key for query expansion and answer generation.
- Internet access on first run to download transformer models and weights.

## Setup
1) Create and activate a virtual environment (Windows PowerShell shown; adapt for your shell/OS):
   - `python -m venv .venv`
   - `.\\.venv\\Scripts\\Activate.ps1`
2) Install dependencies:
   - `pip install --upgrade pip`
   - `pip install -r requirements.txt`
3) Configure your OpenAI API key:
   - Windows (PowerShell): `$env:OPENAI_API_KEY="sk-...your_key..."`
   - macOS/Linux (bash/zsh): `export OPENAI_API_KEY="sk-...your_key..."`
   - Or create a .env file at root directory with `OPENAI_API_KEY=Your-openai-api-key`

## Usage
Build persistent index and BM25 cache (do this once per collection):
- `python rag.py --build --path data/2wikimqa.jsonl --collection demo`

Ask a question using the existing index:
- `python rag.py --path data/2wikimqa.jsonl --collection demo --query "Where did Helena Carroll's father study?"`

Notes:
- `--path` should point to the same corpus used to build. If the BM25 cache is missing, the system will rebuild it on the fly.
- For PDFs, sentence-level chunking is automatically enabled; for JSONL passages (e.g., 2Wiki), each passage is a single chunk.

## How it works (module interactions)

```mermaid
flowchart TB
  subgraph CLI
    CLI[rag/cli.py]
  end

  subgraph Ingestion
    PARSER[rag/ingestion/parser.py]
    CHUNK[rag/ingestion/chunking.py]
    PERSIST[rag/ingestion/persist.py]
  end

  subgraph Retrieval
    BM25[rag/retrieval/bm25.py]
    EXP[rag/retrieval/expansion.py]
    FUSION[rag/retrieval/fusion.py]
  end

  subgraph Rerank
    XENC[rag/rerank/cross_encoder.py]
  end

  subgraph Generation
    ANS[rag/generation/answer.py]
  end

  CLI -->|--build| PERSIST
  CLI -->|query| EXP --> FUSION
  FUSION --> BM25
  FUSION --> XENC
  XENC --> ANS

  PARSER --> PERSIST
  CHUNK --> PERSIST
```

- Build mode:
  - [rag/cli.py](rag/cli.py) calls [rag/ingestion/persist.py](rag/ingestion/persist.py) to:
    - Parse/normalize documents via [rag/ingestion/parser.py](rag/ingestion/parser.py)
    - Chunk PDFs via [rag/ingestion/chunking.py](rag/ingestion/chunking.py)
    - Persist to Chroma and build BM25, caching tokens/doc_ids to a pickle
- Query mode:
  - [rag/cli.py](rag/cli.py) loads Chroma and the BM25 cache from [rag/retrieval/bm25.py](rag/retrieval/bm25.py)
  - Expands the user query via [rag/retrieval/expansion.py](rag/retrieval/expansion.py)
  - Performs dense + BM25 retrieval and RRF fusion via [rag/retrieval/fusion.py](rag/retrieval/fusion.py)
  - Reranks candidates via [rag/rerank/cross_encoder.py](rag/rerank/cross_encoder.py)
  - Generates a grounded answer via [rag/generation/answer.py](rag/generation/answer.py), using text-only context and inline citations

## What gets generated
- Persistent Chroma directory: `chroma_<collection>/` (e.g., `chroma_demo/`)
  - Stores the dense index (embeddings) and metadata
- BM25 cache file: `bm25_<collection>.pkl` (e.g., `bm25_demo.pkl`)
  - Contains tokenized chunk texts and aligned chunk IDs
- Console output:
  - Answer grounded with inline bracketed citations
  - Selected citations and rerank scores
  - Timing breakdown for ingest, expand, retrieve, rerank, generate

## Troubleshooting
- First run is slow:
  - `sentence-transformers` and `transformers` will download models to your HF cache
  - Chroma will initialize a persistent local store under `chroma_<collection>/`
- Missing `OPENAI_API_KEY`:
  - Multi-query expansion and answer generation require a valid key. Set the env var before running.
- Rebuilding indices:
  - Delete `chroma_<collection>/` and `bm25_<collection>.pkl` to fully rebuild; or re-run with `--build` to refresh.
- Windows “package shadowing”:
  - The [rag.py](rag.py) shim ensures imports resolve to the `rag/` package while preserving the `python rag.py` UX.
