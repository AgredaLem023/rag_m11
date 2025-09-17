from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import CrossEncoder


def rerank_and_select(
    cross_encoder: CrossEncoder,
    query: str,
    candidate_ids: List[str],
    chunk_id_to_text: Dict[str, str],
    chunk_id_to_meta: Dict[str, Dict[str, Any]],
    top_n: int = 5,
    rerank_threshold: float = 0.2,
) -> Tuple[List[str], List[Dict[str, Any]], List[float], bool]:
    """
    Return (selected_texts, selected_metas, selected_scores, abstain_flag).

    Scores (higher=better) are produced by the cross-encoder for (query, text) pairs.
    We select the top_n by score and abstain when the best score is below threshold.
    """
    pairs: List[List[str]] = []
    per_id_text: List[Tuple[str, str]] = []

    for cid in candidate_ids:
        t = chunk_id_to_text.get(cid, "")
        if not t:
            continue
        pairs.append([query, t])
        per_id_text.append((cid, t))

    if not pairs:
        return [], [], [], True

    scores = cross_encoder.predict(pairs)
    order = np.argsort(scores)[::-1][:top_n]

    selected_texts: List[str] = []
    selected_metas: List[Dict[str, Any]] = []
    selected_scores: List[float] = []

    for idx in order:
        cid, t = per_id_text[int(idx)]
        selected_texts.append(t)
        selected_metas.append(chunk_id_to_meta.get(cid, {}))
        selected_scores.append(float(scores[int(idx)]))

    abstain = (len(selected_scores) == 0) or (max(selected_scores) < rerank_threshold)
    return selected_texts, selected_metas, selected_scores, abstain