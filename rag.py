import os
import re
import time
import json
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

ABSTAIN_MSG = ("I'm sorry, I can't answer that query based on the information I have access to. "
               "My knowledge is limited, and I couldn't find a relevant passage.  Please try rephrasing your question or asking about a different topic.")

# Simple BM25 implementation (fallback without external dependencies)
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
        # Track mapping of BM25 internal index to external chunk ids
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

def _parse_record(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Flexible field extraction
    text = None
    for key in ["text", "passage", "content", "document", "ctx"]:
        if key in obj and isinstance(obj[key], str) and obj[key].strip():
            text = obj[key].strip()
            break
    if not text:
        return None
    doc_id = None
    for key in ["id", "doc_id", "page_id", "pid"]:
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

def ingest_corpus(
    dataset_path: Optional[str],
    collection_name: str = "multidoc-collection",
    batch_size: int = 256,
) -> Tuple[Any, SimpleBM25, Dict[str, str], Dict[str, Dict[str, Any]]]:
    """
    Ingest multi-document corpus (directory of JSONL/TXT or a JSONL file), chunk with Chonkie,
    store in Chroma with metadata, and build an in-memory BM25 index over chunks.
    """
    # Determine dataset path with fallbacks
    preferred = dataset_path or "data/wWikiMultihopQA"
    fallback_jsonl = "data/2wikimqa.jsonl"
    fallback_pdf = "data/microsoft-annual-report.pdf"
    path = preferred if os.path.exists(preferred) else (fallback_jsonl if os.path.exists(fallback_jsonl) else fallback_pdf)

    embedding_function = SentenceTransformerEmbeddingFunction()
    client = chromadb.Client()
    collection = client.get_or_create_collection(collection_name, embedding_function=embedding_function)  # type: ignore

    chunker = build_chunker()

    docs: List[Dict[str, Any]] = []
    if os.path.isdir(path):
        # Ingest all .jsonl and .txt files in directory
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
                rec = _parse_record(obj)
                if rec:
                    docs.append(rec)
    else:
        # Fallback: single PDF
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            pages = [(p.extract_text() or "").strip() for p in reader.pages]
            text = "\n\n".join([t for t in pages if t])
            if text:
                docs.append({"doc_id": "pdf_fallback", "title": os.path.basename(path), "source": path, "text": text})
        except Exception:
            pass

    # Chunk and index
    chunk_id_to_text: Dict[str, str] = {}
    chunk_id_to_meta: Dict[str, Dict[str, Any]] = {}
    bm25_tokens: List[List[str]] = []
    bm25_chunk_ids: List[str] = []

    # Optionally clear existing collection? We'll append to allow re-runs building up; to start clean, uncomment:
    # if collection.count() > 0:
    #     collection.delete(where={})

    batch_docs: List[str] = []
    batch_ids: List[str] = []
    batch_metas: List[Dict[str, Union[str, int, float, bool, None]]] = []

    for doc in docs:
        doc_id = doc["doc_id"]
        title = doc.get("title") or ""
        source = doc.get("source") or ""
        text = doc["text"]
        # Chunk
        chunks = chunker.chunk(text)
        for i, ch in enumerate(chunks):
            cid = f"{doc_id}::chunk_{i}"
            ctext = ch.text
            meta: Dict[str, Union[str, int, float, bool, None]] = {"doc_id": doc_id, "title": title, "source": source, "chunk_index": i}  # type: ignore[assignment]
            chunk_id_to_text[cid] = ctext
            chunk_id_to_meta[cid] = dict(meta)
            batch_ids.append(cid)
            batch_docs.append(ctext)
            batch_metas.append(meta)
            # BM25
            toks = simple_tokenize(ctext)
            bm25_tokens.append(toks)
            bm25_chunk_ids.append(cid)
            # Flush in batches
            if len(batch_ids) >= batch_size:
                collection.add(ids=batch_ids, documents=batch_docs, metadatas=cast(Any, batch_metas))
                batch_docs, batch_ids, batch_metas = [], [], []
    # Final flush
    if batch_ids:
        collection.add(ids=batch_ids, documents=batch_docs, metadatas=cast(Any, batch_metas))

    # Build BM25 index
    bm25 = SimpleBM25(bm25_tokens)
    # Attach doc id mapping
    bm25.doc_ids = bm25_chunk_ids  # type: ignore

    return collection, bm25, chunk_id_to_text, chunk_id_to_meta

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

    # Dense retrieval via Chroma
    dense = collection.query(query_texts=queries, n_results=k_dense)  # rely on default ids in response
    # dense["ids"] is List[List[str]]
    for lst in dense.get("ids", []):
        rank_lists.append([cid for cid in lst])
        info["dense"].append(lst)

    # BM25 retrieval
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
    # Build grounded context with citations
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

def main():
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Ingest multi-doc corpus
    t0 = time.perf_counter()
    dataset_path = "data\2wikimqa.jsonl"
    collection, bm25, chunk_id_to_text, chunk_id_to_meta = ingest_corpus(dataset_path)
    t_ingest = time.perf_counter() - t0

    # Reranker
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Example query (replace with user input or evaluation loop)
    original_query = "Where did Helena Carroll's father study?"

    # Multi-query expansion
    t1 = time.perf_counter()
    generated_queries = generate_multi_query(client, original_query)  # may be empty
    queries = [original_query] + generated_queries
    t_expand = time.perf_counter() - t1

    # Hybrid retrieval (dense + BM25 with RRF)
    t2 = time.perf_counter()
    merged, _ = hybrid_retrieve(queries, collection, bm25, k_dense=15, k_bm25=15, rrf_k=60)
    candidate_ids = [cid for cid, _ in merged]
    t_retrieve = time.perf_counter() - t2

    # Abstain if nothing retrieved
    if not candidate_ids:
        print("Answer:")
        print(ABSTAIN_MSG)
        print(f"Latency - ingest: {t_ingest:.3f}s | expand: {t_expand:.3f}s | retrieve: {t_retrieve:.3f}s | rerank: 0.000s | generate: 0.000s")
        return

    # Rerank and select top-k
    t3 = time.perf_counter()
    selected_texts, selected_metas, selected_scores, abstain_rerank = rerank_and_select(
        cross_encoder, original_query, candidate_ids, chunk_id_to_text, chunk_id_to_meta, top_n=5, rerank_threshold=0.2
    )
    t_rerank = time.perf_counter() - t3

    if abstain_rerank:
        print("Answer:")
        print(ABSTAIN_MSG)
        print(f"Latency - ingest: {t_ingest:.3f}s | expand: {t_expand:.3f}s | retrieve: {t_retrieve:.3f}s | rerank: {t_rerank:.3f}s | generate: 0.000s")
        return

    # Generation with grounded prompt and abstention rule in instructions
    t4 = time.perf_counter()
    answer = generate_answer(client, original_query, selected_texts, selected_metas, model="gpt-5-mini")
    t_generate = time.perf_counter() - t4

    print("Answer:")
    print(answer)
    print("Citations (selected):")
    cites = []
    for m, s in zip(selected_metas, selected_scores):
        cites.append(f"- {m.get('title') or m.get('doc_id')} (score={s:.3f})")
    print("\n".join(cites))
    print(f"Latency - ingest: {t_ingest:.3f}s | expand: {t_expand:.3f}s | retrieve: {t_retrieve:.3f}s | rerank: {t_rerank:.3f}s | generate: {t_generate:.3f}s")

if __name__ == "__main__":
    main()
