from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pickle



# Tokenization
_token_re = re.compile(r"[A-Za-z0-9_]+")


def simple_tokenize(text: str) -> List[str]:
    return [t.lower() for t in _token_re.findall(text)]



# BM25 core
def math_log_idf(N: int, df: int) -> float:
    """
    BM25 IDF with +1 smoothing to avoid negative idf for very common terms.
    """
    return float(np.log((N - df + 0.5) / (df + 0.5) + 1.0))


class SimpleBM25:
    """
    Minimal BM25 implementation over tokenized documents.

    Attributes:
        corpus_tokens: List of token lists per document
        N:            Number of documents
        k1, b:        BM25 hyperparameters
        doc_freq:     Document frequency per term
        avgdl:        Average document length
        doc_len:      Document lengths
        term_freqs:   Term frequencies per document
        idf:          Precomputed IDF per term
        doc_ids:      Optional mapping of internal index -> external chunk ids
    """

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus_tokens = corpus_tokens
        self.N = len(corpus_tokens)
        self.k1 = k1
        self.b = b
        self.doc_freq: Dict[str, int] = {}
        self.avgdl = 0.0
        self.doc_len = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_len) / self.N if self.N > 0 else 0.0
        self.term_freqs: List[Dict[str, int]] = []

        for tokens in corpus_tokens:
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)
            for t in tf.keys():
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1

        # Precompute IDF
        self.idf: Dict[str, float] = {}
        for t, df in self.doc_freq.items():
            self.idf[t] = math_log_idf(self.N, df)

        # mapping of BM25 internal index -> external chunk ids
        self.doc_ids: List[str] = []

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.N
        # Use set(query_tokens) to avoid repeated terms dominating
        for qi in set(query_tokens):
            if qi not in self.idf:
                continue
            idf = self.idf[qi]
            for idx, tf in enumerate(self.term_freqs):
                f = tf.get(qi, 0)
                if f == 0:
                    continue
                dl = self.doc_len[idx] if self.doc_len[idx] > 0 else 1
                denom = f + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl if self.avgdl > 0 else 1)))
                score = idf * (f * (self.k1 + 1)) / denom
                scores[idx] += float(score)
        return scores

    def top_n(self, query_tokens: List[str], n: int) -> List[Tuple[int, float]]:
        scores = self.get_scores(query_tokens)
        idxs = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[int(i)])) for i in idxs if scores[int(i)] > 0.0]



# Cache I/O
def load_bm25_from_cache(cache_path: Path) -> Optional[SimpleBM25]:
    """
    Load a SimpleBM25 from a pickle cache created during ingestion.

    The cache is expected to store:
      - "tokens": List[List[str]] -> tokenized chunk texts
      - "doc_ids": List[str]      -> external chunk ids aligned to tokens
    """
    if not cache_path or not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as fp:
            data = pickle.load(fp)
        tokens = data.get("tokens")
        doc_ids = data.get("doc_ids")
        if not isinstance(tokens, list) or not isinstance(doc_ids, list):
            return None
        bm25 = SimpleBM25(tokens)
        bm25.doc_ids = doc_ids
        return bm25
    except (OSError, pickle.UnpicklingError, KeyError, TypeError, ValueError):
        return None