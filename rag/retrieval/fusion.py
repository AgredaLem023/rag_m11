from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .bm25 import SimpleBM25, simple_tokenize


def rrf_merge(rank_lists: List[List[str]], k: int = 60, top_n: int = 50) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion across multiple ranked lists of candidate ids.
    """
    scores: Dict[str, float] = {}
    for rl in rank_lists:
        for rank, cid in enumerate(rl, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def hybrid_retrieve(
    queries: List[str],
    collection,
    bm25: SimpleBM25,
    k_dense: int = 15,
    k_bm25: int = 15,
    rrf_k: int = 60,
) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """
    Return list of (chunk_id, rrf_score) and raw retrieval info for diagnostics.
    Combines dense retrieval from Chroma with BM25, then fuses via RRF.
    """
    rank_lists: List[List[str]] = []
    info: Dict[str, Any] = {"dense": [], "bm25": []}

    dense = collection.query(query_texts=queries, n_results=k_dense)
    for lst in dense.get("ids", []):
        ids_list = [cid for cid in lst]
        rank_lists.append(ids_list)
        info["dense"].append(ids_list)

    for q in queries:
        q_tokens = simple_tokenize(q)
        top = bm25.top_n(q_tokens, k_bm25)
        cids = [bm25.doc_ids[i] for (i, _) in top]
        rank_lists.append(cids)
        info["bm25"].append(cids)

    merged = rrf_merge(rank_lists, k=rrf_k, top_n=max(k_dense, k_bm25))
    return merged, info