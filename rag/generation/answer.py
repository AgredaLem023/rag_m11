from __future__ import annotations

from typing import Any, Dict, List, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from rag.constants import ABSTAIN_MSG


def generate_answer(
    client: OpenAI,
    query: str,
    contexts: List[str],
    metas: List[Dict[str, Any]],
    model: str = "gpt-5-mini",
) -> str:
    """
    Compose a grounded answer using ONLY the provided context snippets.
    Applies strict abstention when context is insufficient.
    """
    if not contexts:
        return ABSTAIN_MSG

    context_blocks: List[str] = []
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
    messages: List[ChatCompletionMessageParam] = cast(
        List[ChatCompletionMessageParam],
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    resp = client.chat.completions.create(model=model, messages=messages)
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        return ABSTAIN_MSG

    low = content.lower()
    if "no answer" in low or "i don't know" in low:
        return ABSTAIN_MSG
    return content