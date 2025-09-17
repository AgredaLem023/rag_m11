import os
import re
import time
import json
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union, cast

import numpy as np
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import CrossEncoder
from chonkie import SentenceChunker
from transformers import AutoTokenizer
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from dotenv import load_dotenv

# -----------------------------
# Globals determined at runtime (set in main)
# -----------------------------
PERSIST_DIR: Optional[Path] = None     # set per collection
BM25_CACHE: Optional[Path] = None      # set per collection
USE_CHUNKER: bool = True               # PDFs: True; 2Wiki JSONL: False
ADD_BATCH_SIZE: int = 1024             # fewer .add calls -> faster

ABSTAIN_MSG = ("I'm sorry, I can't answer that query based on the information I have access to. "
               "My knowledge is limited, and I couldn't find a relevant passage.  Please try rephrasing your question or asking about a different topic.")

# -----------------------------
# 2WikiMultihopQA passage splitter
# -----------------------------
# Matches lines like: "Passage 1: Some Wikipedia Title"
_PASSAGE_HEADER = re.compile(r"Passage\s+(\d+):\s*([^\n]+)\s*\n", re.IGNORECASE)

def _split_2wiki_context(example_id: str, context_str: str) -> List[Dict[str, Any]]:
    """
    Split a 2WikiMultihopQA 'context' mega-string into individual passages.
    Returns list of dicts: {doc_id, title, source, text}
    """
    parts: List[Dict[str, Any]] = []
    headers = list(_PASSAGE_HEADER.finditer(context_str))

    for i, m in enumerate(headers):
        pnum = m.group(1)
        title = (m.group(2) or "").strip()
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(context_str)
        body = context_str[start:end].strip()
        if not body:
            continue
        doc_id = f"{example_id}::passage_{pnum}"
        parts.append({
            "doc_id": doc_id,
            "title": title or f"passage_{pnum}",
            "source": f"2wikimqa::{example_id}::{pnum}",
            "text": body
        })

    # Fallback: no headers found -> keep entire context as one doc
    if not parts and context_str.strip():
        parts.append({
            "doc_id": f"{example_id}::context",
            "title": "context",
            "source": f"2wikimqa::{example_id}",
            "text": context_str.strip()
        })
    return parts

# -----------------------------
# Simple BM25 implementation
# -----------------------------
class SimpleBM25:
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
                scores[idx] += score
        return scores

    def top_n(self, query_tokens: List[str], n: int) -> List[Tuple[int, float]]:
        scores = self.get_scores(query_tokens)
        idxs = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[int(i)])) for i in idxs if scores[int(i)] > 0.0]

def math_log_idf(N: int, df: int) -> float:
    # BM25 IDF with +1 smoothing to avoid negative idf for very common terms
    return np.log((N - df + 0.5) / (df + 0.5) + 1.0)

_token_re = re.compile(r"[A-Za-z0-9_]+")
def simple_tokenize(text: str) -> List[str]:
    return [t.lower() for t in _token_re.findall(text)]

def build_chunker() -> SentenceChunker:
    hf_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return SentenceChunker(
        tokenizer_or_token_counter=hf_tok,
        chunk_size=512,
        chunk_overlap=50,
        min_sentences_per_chunk=1,
    )

# -----------------------------
# Generic record parser (includes 'context' as last-resort key)
# -----------------------------
def _parse_record(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = None
    for key in ["text", "passage", "content", "document", "ctx", "context"]:
        if key in obj and isinstance(obj[key], str) and obj[key].strip():
            text = obj[key].strip()
            break
    if not text:
        return None
    doc_id = None
    for key in ["id", "doc_id", "page_id", "pid", "_id"]:
        if key in obj:
            doc_id = str(obj[key])
            break
    if not doc_id:
        doc_id = str(abs(hash(text)) % (10**12))
    title = None
    for key in ["title", "doc_title", "name", "page_title"]:
        if key in obj and isinstance(obj[key], str):
            title = obj[key].strip()
            break
    if not title:
        title = f"doc_{doc_id}"
    source = None
    for key in ["source", "url", "wiki_url"]:
        if key in obj and isinstance(obj[key], str):
            source = obj[key].strip()
            break
    return {"doc_id": doc_id, "title": title, "source": source, "text": text}

# -----------------------------
# Ingestion (JSONL + PDF, persistent Chroma & BM25 cache)
# -----------------------------
def ingest_corpus(
    dataset_path: Optional[str],
    collection_name: str = "multidoc-collection",
    batch_size: int = 256,
    use_chunker: bool = True,
) -> Tuple[Any, SimpleBM25, Dict[str, str], Dict[str, Dict[str, Any]]]:
    """
    Ingest multi-document corpus (dir of JSONL/TXT, a JSONL file, or a PDF),
    optionally chunk text, store in Chroma, and build an in-memory BM25 over chunks.
    Uses persistent Chroma (PERSIST_DIR) and caches BM25 (BM25_CACHE) if set.
    """
    assert PERSIST_DIR is not None, "PERSIST_DIR must be set before calling ingest_corpus"
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    embedding_function = SentenceTransformerEmbeddingFunction()
    collection = client.get_or_create_collection(collection_name, embedding_function=embedding_function)  # type: ignore

    # Build chunker only if requested (PDFs)
    chunker = build_chunker() if use_chunker else None

    path = dataset_path or ""
    docs: List[Dict[str, Any]] = []

    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for fname in files:
                fpath = os.path.join(root, fname)
                if fname.lower().endswith(".jsonl"):
                    with open(fpath, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            if isinstance(obj.get("context"), str) and isinstance(obj.get("input"), str):
                                ex_id = str(obj.get("_id") or abs(hash(line)) % (10**12))
                                context_docs = _split_2wiki_context(ex_id, obj["context"])
                                docs.extend(context_docs)
                            else:
                                rec = _parse_record(obj)
                                if rec:
                                    docs.append(rec)
                elif fname.lower().endswith(".txt"):
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            text = f.read().strip()
                        if text:
                            doc_id = os.path.splitext(fname)[0]
                            docs.append({"doc_id": doc_id, "title": doc_id, "source": fpath, "text": text})
                    except Exception:
                        pass

    elif os.path.isfile(path) and path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                if isinstance(obj.get("context"), str) and isinstance(obj.get("input"), str):
                    ex_id = str(obj.get("_id") or abs(hash(line)) % (10**12))
                    context_docs = _split_2wiki_context(ex_id, obj["context"])
                    docs.extend(context_docs)
                else:
                    rec = _parse_record(obj)
                    if rec:
                        docs.append(rec)

    elif os.path.isfile(path) and path.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            pages = [(p.extract_text() or "").strip() for p in reader.pages]
            text = "\n\n".join([t for t in pages if t])
            if text:
                docs.append({"doc_id": "pdf_fallback", "title": os.path.basename(path), "source": path, "text": text})
        except Exception:
            pass

    else:
        raise FileNotFoundError(f"Dataset path not found or unsupported: {path}")

    # Chunk & index
    chunk_id_to_text: Dict[str, str] = {}
    chunk_id_to_meta: Dict[str, Dict[str, Any]] = {}
    bm25_tokens: List[List[str]] = []
    bm25_chunk_ids: List[str] = []

    batch_docs: List[str] = []
    batch_ids: List[str] = []
    batch_metas: List[Dict[str, Union[str, int, float, bool, None]]] = []

    for doc in docs:
        doc_id = doc["doc_id"]
        title = doc.get("title") or ""
        source = doc.get("source") or ""
        text = doc["text"]

        # For PDFs: sentence chunking; For 2Wiki: 1 chunk per passage
        if use_chunker and chunker is not None:
            chunks = chunker.chunk(text)
            chunk_texts = [ch.text for ch in chunks]
        else:
            chunk_texts = [text]

        for i, ctext in enumerate(chunk_texts):
            cid = f"{doc_id}::chunk_{i}"
            meta: Dict[str, Union[str, int, float, bool, None]] = {
                "doc_id": doc_id, "title": title, "source": source, "chunk_index": i
            }
            chunk_id_to_text[cid] = ctext
            chunk_id_to_meta[cid] = dict(meta)
            batch_ids.append(cid)
            batch_docs.append(ctext)
            batch_metas.append(meta)

            toks = simple_tokenize(ctext)
            bm25_tokens.append(toks)
            bm25_chunk_ids.append(cid)

            if len(batch_ids) >= batch_size:
                collection.add(ids=batch_ids, documents=batch_docs, metadatas=cast(Any, batch_metas))
                batch_docs, batch_ids, batch_metas = [], [], []

    if batch_ids:
        collection.add(ids=batch_ids, documents=batch_docs, metadatas=cast(Any, batch_metas))

    bm25 = SimpleBM25(bm25_tokens)
    bm25.doc_ids = bm25_chunk_ids  # type: ignore

    # Persist BM25 cache
    if BM25_CACHE is not None:
        try:
            with open(BM25_CACHE, "wb") as fp:
                pickle.dump({"tokens": bm25_tokens, "doc_ids": bm25_chunk_ids}, fp, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    return collection, bm25, chunk_id_to_text, chunk_id_to_meta

def load_bm25_from_cache() -> Optional[SimpleBM25]:
    if BM25_CACHE is None or not BM25_CACHE.exists():
        return None
    try:
        with open(BM25_CACHE, "rb") as fp:
            data = pickle.load(fp)
        bm25 = SimpleBM25(data["tokens"])
        bm25.doc_ids = data["doc_ids"]
        return bm25
    except Exception:
        return None

# -----------------------------
# Retrieval / Rerank / Answer
# -----------------------------
def generate_multi_query(client: OpenAI, query: str, model: str = "gpt-5-mini") -> List[str]:
    prompt = (
        "You are a knowledgeable research assistant.\n"
        "For the given question, propose up to five short, diverse, non-overlapping related questions that could help retrieve relevant evidence.\n"
        "Return each question on its own line without numbering."
    )
    messages: List[ChatCompletionMessageParam] = cast(List[ChatCompletionMessageParam], [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ])
    resp = client.chat.completions.create(model=model, messages=messages)
    content = resp.choices[0].message.content or ""
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    return lines[:5]

def rrf_merge(rank_lists: List[List[str]], k: int = 60, top_n: int = 50) -> List[Tuple[str, float]]:
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
    """Return list of (chunk_id, rrf_score) and raw retrieval info for diagnostics."""
    rank_lists: List[List[str]] = []
    info: Dict[str, Any] = {"dense": [], "bm25": []}

    dense = collection.query(query_texts=queries, n_results=k_dense)
    for lst in dense.get("ids", []):
        rank_lists.append([cid for cid in lst])
        info["dense"].append(lst)

    for q in queries:
        q_tokens = simple_tokenize(q)
        top = bm25.top_n(q_tokens, k_bm25)
        cids = [bm25.doc_ids[i] for (i, _) in top]
        rank_lists.append(cids)
        info["bm25"].append(cids)

    merged = rrf_merge(rank_lists, k=rrf_k, top_n=max(k_dense, k_bm25))
    return merged, info

def rerank_and_select(
    cross_encoder: CrossEncoder,
    query: str,
    candidate_ids: List[str],
    chunk_id_to_text: Dict[str, str],
    chunk_id_to_meta: Dict[str, Dict[str, Any]],
    top_n: int = 5,
    rerank_threshold: float = 0.2,
) -> Tuple[List[str], List[Dict[str, Any]], List[float], bool]:
    """Return (selected_texts, selected_metas, selected_scores, abstain_flag)."""
    pairs = []
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

def generate_answer(
    client: OpenAI,
    query: str,
    contexts: List[str],
    metas: List[Dict[str, Any]],
    model: str = "gpt-5-mini",
) -> str:
    if not contexts:
        return ABSTAIN_MSG
    context_blocks = []
    for i, (ctx, m) in enumerate(zip(contexts, metas), start=1):
        cite = m.get("title") or m.get("doc_id") or f"chunk_{i}"
        key = f"[{cite}]"
        context_blocks.append(f"{key} {ctx}")
    joined_context = "\n\n".join(context_blocks)
    system_prompt = (
        "You are a careful, evidence-grounded assistant. "
        "Use ONLY the provided context snippets to answer. "
        f"If the context is insufficient to answer, reply exactly: {ABSTAIN_MSG} "
        "Cite evidence after each sentence using the provided citation keys in square brackets. "
        "Do not use prior knowledge."
    )
    user_prompt = (
        f"Question: {query}\n\n"
        f"Context snippets with citations:\n{joined_context}\n\n"
        "Provide a concise answer grounded strictly on the snippets."
    )
    messages: List[ChatCompletionMessageParam] = cast(List[ChatCompletionMessageParam], [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])
    resp = client.chat.completions.create(model=model, messages=messages)
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        return ABSTAIN_MSG
    low = content.lower()
    if "no answer" in low or "i don't know" in low:
        return ABSTAIN_MSG
    return content

# -----------------------------
# CLI entry
# -----------------------------
def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true",
                        help="Build persistent index from --path and cache BM25 (slow once, fast later)")
    parser.add_argument("--path", type=str, required=True,
                        help="Path to corpus (JSONL for 2WikiMultihopQA or PDF)")
    parser.add_argument("--collection", type=str, required=True,
                        help="Chroma collection name (use a different one per corpus)")
    parser.add_argument("--query", type=str, default="Where did Helena Carroll's father study?",
                        help="Query to ask in query mode")
    args = parser.parse_args()

    # Corpus-aware settings
    is_pdf = args.path.lower().endswith(".pdf")
    global USE_CHUNKER, PERSIST_DIR, BM25_CACHE
    USE_CHUNKER = is_pdf  # PDFs -> chunk sentences; 2Wiki JSONL -> single-chunk passages
    PERSIST_DIR = Path(f"./chroma_{args.collection}")
    BM25_CACHE = Path(f"./bm25_{args.collection}.pkl")

    # OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # BUILD MODE: ingest once, persist Chroma & BM25
    if args.build:
        t0 = time.perf_counter()
        collection, bm25, chunk_id_to_text, chunk_id_to_meta = ingest_corpus(
            args.path,
            collection_name=args.collection,
            batch_size=ADD_BATCH_SIZE,
            use_chunker=USE_CHUNKER,
        )
        t_ingest = time.perf_counter() - t0
        print(f"Build done. Latency - ingest: {t_ingest:.3f}s | chunks={len(chunk_id_to_text)} | "
              f"chroma_dir={PERSIST_DIR} | bm25_cache={BM25_CACHE}")
        return

    # QUERY MODE: reuse existing persistent Chroma & BM25 cache
    t_ingest_start = time.perf_counter()
    embedding_function = SentenceTransformerEmbeddingFunction()
    chroma_client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    collection = chroma_client.get_or_create_collection(args.collection, embedding_function=embedding_function) # type: ignore

    bm25 = load_bm25_from_cache()
    chunk_id_to_text: Dict[str, str] = {}
    chunk_id_to_meta: Dict[str, Dict[str, Any]] = {}

    # If BM25 is missing (first run without --build), build it now (slow once)
    if bm25 is None:
        collection, bm25, chunk_id_to_text, chunk_id_to_meta = ingest_corpus(
            args.path,
            collection_name=args.collection,
            batch_size=ADD_BATCH_SIZE,
            use_chunker=USE_CHUNKER,
        )

    t_ingest = time.perf_counter() - t_ingest_start

    # Reranker
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Query
    original_query = args.query

    # Multi-query expansion
    t1 = time.perf_counter()
    generated_queries = generate_multi_query(client, original_query)
    queries = [original_query] + generated_queries
    t_expand = time.perf_counter() - t1

    # Retrieval (dense + BM25)
    t2 = time.perf_counter()
    merged, _ = hybrid_retrieve(queries, collection, bm25, k_dense=15, k_bm25=15, rrf_k=60)
    candidate_ids = [cid for cid, _ in merged]
    t_retrieve = time.perf_counter() - t2

    if not candidate_ids:
        print("Answer:")
        print(ABSTAIN_MSG)
        print(f"Latency - ingest: {t_ingest:.3f}s | expand: {t_expand:.3f}s | retrieve: {t_retrieve:.3f}s | "
              f"rerank: 0.000s | generate: 0.000s")
        return

    # If we didn't preload texts/metas, fetch them from Chroma (batched)
    if not chunk_id_to_text:
        selected = collection.get(ids=candidate_ids)
        id_to_doc = dict(zip(selected.get("ids", []), selected.get("documents", []))) # type: ignore
        id_to_meta = dict(zip(selected.get("ids", []), selected.get("metadatas", []))) # type: ignore
        chunk_id_to_text = id_to_doc # type: ignore
        chunk_id_to_meta = id_to_meta # type: ignore

    # Rerank and select
    t3 = time.perf_counter()
    selected_texts, selected_metas, selected_scores, abstain_rerank = rerank_and_select(
        cross_encoder, original_query, candidate_ids, chunk_id_to_text, chunk_id_to_meta, top_n=5, rerank_threshold=0.2
    )
    t_rerank = time.perf_counter() - t3

    if abstain_rerank:
        print("Answer:")
        print(ABSTAIN_MSG)
        print(f"Latency - ingest: {t_ingest:.3f}s | expand: {t_expand:.3f}s | retrieve: {t_retrieve:.3f}s | "
              f"rerank: {t_rerank:.3f}s | generate: 0.000s")
        return

    # Generate
    t4 = time.perf_counter()
    answer = generate_answer(client, original_query, selected_texts, selected_metas, model="gpt-5-mini")
    t_generate = time.perf_counter() - t4

    print("Answer:")
    print(answer)
    print("Citations (selected):")
    for m, s in zip(selected_metas, selected_scores):
        print(f"- {m.get('title') or m.get('doc_id')} (score={s:.3f})")
    print(f"Latency - ingest: {t_ingest:.3f}s | expand: {t_expand:.3f}s | retrieve: {t_retrieve:.3f}s | "
          f"rerank: {t_rerank:.3f}s | generate: {t_generate:.3f}s")


if __name__ == "__main__":
    main()
