from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, cast

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder

from rag.config import Config
from rag.constants import ABSTAIN_MSG
from rag.generation.answer import generate_answer
from rag.ingestion.persist import ingest_corpus
from rag.retrieval.bm25 import load_bm25_from_cache, SimpleBM25
from rag.retrieval.expansion import generate_multi_query
from rag.retrieval.fusion import hybrid_retrieve


def build_mode(cfg: Config, dataset_path: str) -> None:
    t0 = time.perf_counter()
    collection, bm25, chunk_id_to_text, chunk_id_to_meta = ingest_corpus(
        dataset_path=dataset_path,
        persist_dir=cfg.persist_dir,
        bm25_cache=cfg.bm25_cache,
        collection_name=cfg.collection_name,
        batch_size=cfg.add_batch_size,
        use_chunker=cfg.use_chunker,
    )
    t_ingest = time.perf_counter() - t0
    print(
        f"Build done. Latency - ingest: {t_ingest:.3f}s | chunks={len(chunk_id_to_text)} | "
        f"chroma_dir={cfg.persist_dir} | bm25_cache={cfg.bm25_cache}"
    )


def query_mode(
    cfg: Config,
    dataset_path: str,
    query: str,
    openai_client: OpenAI,
) -> None:
    t_ingest_start = time.perf_counter()
    embedding_function = SentenceTransformerEmbeddingFunction()
    chroma_client = chromadb.PersistentClient(path=str(cfg.persist_dir))
    collection = chroma_client.get_or_create_collection(
        cfg.collection_name, embedding_function=embedding_function  # type: ignore
    )

    bm25 = load_bm25_from_cache(cfg.bm25_cache)
    chunk_id_to_text: Dict[str, str] = {}
    chunk_id_to_meta: Dict[str, Dict[str, Any]] = {}

    # If BM25 is missing (first run without --build), build it now (slow once)
    if bm25 is None:
        collection, bm25, chunk_id_to_text, chunk_id_to_meta = ingest_corpus(
            dataset_path=dataset_path,
            persist_dir=cfg.persist_dir,
            bm25_cache=cfg.bm25_cache,
            collection_name=cfg.collection_name,
            batch_size=cfg.add_batch_size,
            use_chunker=cfg.use_chunker,
        )

    t_ingest = time.perf_counter() - t_ingest_start

    # Reranker
    cross_encoder = CrossEncoder(cfg.cross_encoder_model)

    # Query
    original_query = query

    # Multi-query expansion
    t1 = time.perf_counter()
    generated_queries = generate_multi_query(openai_client, original_query, model=cfg.expand_model)
    queries = [original_query] + generated_queries
    t_expand = time.perf_counter() - t1

    # Retrieval (dense + BM25)
    t2 = time.perf_counter()
    merged, _ = hybrid_retrieve(
        queries,
        collection,
        cast(SimpleBM25, bm25),
        k_dense=cfg.k_dense,
        k_bm25=cfg.k_bm25,
        rrf_k=cfg.rrf_k,
    )
    candidate_ids = [cid for cid, _ in merged]
    t_retrieve = time.perf_counter() - t2

    if not candidate_ids:
        print("Answer:")
        print(ABSTAIN_MSG)
        print(
            f"Latency - ingest: {t_ingest:.3f}s | expand: {t_expand:.3f}s | retrieve: {t_retrieve:.3f}s | "
            f"rerank: 0.000s | generate: 0.000s"
        )
        return

    # If we didn't preload texts/metas, fetch them from Chroma (batched)
    if not chunk_id_to_text:
        selected = collection.get(ids=candidate_ids)
        id_to_doc = dict(zip(selected.get("ids", []), selected.get("documents", [])))  # type: ignore
        id_to_meta = dict(zip(selected.get("ids", []), selected.get("metadatas", [])))  # type: ignore
        chunk_id_to_text = cast(Dict[str, str], id_to_doc)
        chunk_id_to_meta = cast(Dict[str, Dict[str, Any]], id_to_meta)

    # Rerank and select
    from rag.rerank.cross_encoder import rerank_and_select

    t3 = time.perf_counter()
    selected_texts, selected_metas, selected_scores, abstain_rerank = rerank_and_select(
        cross_encoder,
        original_query,
        candidate_ids,
        chunk_id_to_text,
        chunk_id_to_meta,
        top_n=cfg.rerank_top_n,
        rerank_threshold=cfg.rerank_threshold,
    )
    t_rerank = time.perf_counter() - t3

    if abstain_rerank:
        print("Answer:")
        print(ABSTAIN_MSG)
        print(
            f"Latency - ingest: {t_ingest:.3f}s | expand: {t_expand:.3f}s | retrieve: {t_retrieve:.3f}s | "
            f"rerank: {t_rerank:.3f}s | generate: 0.000s"
        )
        return

    # Generate
    t4 = time.perf_counter()
    answer = generate_answer(
        openai_client,
        original_query,
        selected_texts,
        selected_metas,
        model=cfg.answer_model,
    )
    t_generate = time.perf_counter() - t4

    print("Answer:")
    print(answer)
    print("Citations (selected):")
    for m, s in zip(selected_metas, selected_scores):
        print(f"- {m.get('title') or m.get('doc_id')} (score={s:.3f})")
    print(
        f"Latency - ingest: {t_ingest:.3f}s | expand: {t_expand:.3f}s | retrieve: {t_retrieve:.3f}s | "
        f"rerank: {t_rerank:.3f}s | generate: {t_generate:.3f}s"
    )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build", action="store_true", help="Build persistent index from --path and cache BM25 (slow once, fast later)"
    )
    parser.add_argument("--path", type=str, required=True, help="Path to corpus (JSONL for 2WikiMultihopQA or PDF)")
    parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="Chroma collection name (use a different one per corpus)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Where did Helena Carroll's father study?",
        help="Query to ask in query mode",
    )
    args = parser.parse_args()

    # Corpus-aware settings
    is_pdf = args.path.lower().endswith(".pdf")

    cfg = Config(
        persist_dir=Path(f"./chroma_{args.collection}"),
        bm25_cache=Path(f"./bm25_{args.collection}.pkl"),
        collection_name=args.collection,
        use_chunker=is_pdf,
    )

    # OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if args.build:
        build_mode(cfg, args.path)
        return

    query_mode(cfg, dataset_path=args.path, query=args.query, openai_client=client)