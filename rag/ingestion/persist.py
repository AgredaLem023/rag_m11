from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import chromadb
import numpy as np
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .chunking import build_chunker
from .parser import parse_record, split_2wiki_context
from rag.retrieval.bm25 import SimpleBM25, simple_tokenize


def ingest_corpus(
    dataset_path: Optional[str],
    persist_dir: Path,
    bm25_cache: Optional[Path],
    collection_name: str = "multidoc-collection",
    batch_size: int = 256,
    use_chunker: bool = True,
) -> Tuple[Any, SimpleBM25, Dict[str, str], Dict[str, Dict[str, Any]]]:
    """
    Ingest multi-document corpus (dir of JSONL/TXT, a JSONL file, or a PDF),
    optionally chunk text, store in Chroma, and build an in-memory BM25 over chunks.
    Uses persistent Chroma (persist_dir) and caches BM25 (bm25_cache) if set.
    """
    client = chromadb.PersistentClient(path=str(persist_dir))
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
                                context_docs = split_2wiki_context(ex_id, obj["context"])
                                docs.extend(context_docs)
                            else:
                                rec = parse_record(obj)
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
                    context_docs = split_2wiki_context(ex_id, obj["context"])
                    docs.extend(context_docs)
                else:
                    rec = parse_record(obj)
                    if rec:
                        docs.append(rec)

    elif os.path.isfile(path) and path.lower().endswith(".pdf"):
        # Light-weight PDF text extraction using pypdf
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
                "doc_id": doc_id,
                "title": title,
                "source": source,
                "chunk_index": i,
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
    if bm25_cache is not None:
        try:
            with open(bm25_cache, "wb") as fp:
                pickle.dump({"tokens": bm25_tokens, "doc_ids": bm25_chunk_ids}, fp, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    return collection, bm25, chunk_id_to_text, chunk_id_to_meta