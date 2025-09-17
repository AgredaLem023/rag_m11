from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # Persistence
    persist_dir: Path
    bm25_cache: Path
    collection_name: str

    # Ingestion/chunking
    use_chunker: bool = True
    add_batch_size: int = 1024  # affects how often we call .add on Chroma

    # Retrieval sizes and fusion
    k_dense: int = 15
    k_bm25: int = 15
    rrf_k: int = 60

    # Reranking
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 5
    rerank_threshold: float = 0.2

    # LLMs
    expand_model: str = "gpt-5-mini"
    answer_model: str = "gpt-5-mini"