from __future__ import annotations

from typing import List, cast
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


def generate_multi_query(client: OpenAI, query: str, model: str = "gpt-5-mini") -> List[str]:
    """
    Use an LLM to propose up to five short, diverse, non-overlapping related questions
    that could help retrieve relevant evidence. Returns each on its own line.
    """
    prompt = (
        "You are a knowledgeable research assistant.\n"
        "For the given question, propose up to five short, diverse, non-overlapping related questions that could help retrieve relevant evidence.\n"
        "Return each question on its own line without numbering."
    )
    messages: List[ChatCompletionMessageParam] = cast(
        List[ChatCompletionMessageParam],
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
    )
    resp = client.chat.completions.create(model=model, messages=messages)
    content = resp.choices[0].message.content or ""
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    return lines[:5]