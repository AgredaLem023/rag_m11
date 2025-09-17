from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Matches lines like: "Passage 1: Some Wikipedia Title"
_PASSAGE_HEADER = re.compile(r"Passage\s+(\d+):\s*([^\n]+)\s*\n", re.IGNORECASE)


def split_2wiki_context(example_id: str, context_str: str) -> List[Dict[str, Any]]:
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
        parts.append(
            {
                "doc_id": doc_id,
                "title": title or f"passage_{pnum}",
                "source": f"2wikimqa::{example_id}::{pnum}",
                "text": body,
            }
        )

    # Fallback: no headers found -> keep entire context as one doc
    if not parts and context_str.strip():
        parts.append(
            {
                "doc_id": f"{example_id}::context",
                "title": "context",
                "source": f"2wikimqa::{example_id}",
                "text": context_str.strip(),
            }
        )
    return parts


def parse_record(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generic record parser that extracts text/title/id/source from varied schemas.
    Includes 'context' as a last-resort key for text.
    """
    text: Optional[str] = None
    for key in ["text", "passage", "content", "document", "ctx", "context"]:
        if key in obj and isinstance(obj[key], str) and obj[key].strip():
            text = obj[key].strip()
            break
    if not text:
        return None

    doc_id: Optional[str] = None
    for key in ["id", "doc_id", "page_id", "pid", "_id"]:
        if key in obj:
            doc_id = str(obj[key])
            break
    if not doc_id:
        doc_id = str(abs(hash(text)) % (10**12))

    title: Optional[str] = None
    for key in ["title", "doc_title", "name", "page_title"]:
        if key in obj and isinstance(obj[key], str):
            title = obj[key].strip()
            break
    if not title:
        title = f"doc_{doc_id}"

    source: Optional[str] = None
    for key in ["source", "url", "wiki_url"]:
        if key in obj and isinstance(obj[key], str):
            source = obj[key].strip()
            break

    return {"doc_id": doc_id, "title": title, "source": source, "text": text}