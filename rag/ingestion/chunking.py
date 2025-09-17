from __future__ import annotations

from chonkie import SentenceChunker
from transformers import AutoTokenizer


def build_chunker() -> SentenceChunker:
    """
    Build a sentence-aware chunker using the MiniLM tokenizer.
    """
    hf_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return SentenceChunker(
        tokenizer_or_token_counter=hf_tok,
        chunk_size=512,
        chunk_overlap=50,
        min_sentences_per_chunk=1,
    )