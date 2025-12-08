from __future__ import annotations
import os
from groq import Groq
import requests
from typing import List, Optional, Union
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
INSTRUCTIONS = (
    "Answer ONLY using the provided CONTEXT. "
    "If the answer is not in the context, reply: \"I don't know.\" "
    "Cite page/table numbers when available. "
    "Never add external knowledge."
)
def _extract_llama_text(content: Union[str, List]) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                pieces.append(block.get("text", "").strip())
        return " ".join(pieces).strip()
    return ""
def _clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(text.split()).strip()
#I am just defining both claude and llama answer functions here just incase if i end up with out of credits in either service. You can use either of them based on your preference. But make sure you replace the existing calls in other files(frontend.py) accordingly.
def answer_with_claude(
    context_text: str,
    question: str,
    model: str = "claude-3-haiku-20240307",
    max_tokens: int = 1024
) -> str:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": INSTRUCTIONS,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{context_text}\n\n"
                    f"QUESTION:\n{question}"
                )
            }
        ]
    }
    try:
        resp = requests.post(
            CLAUDE_API_URL,
            headers={
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"[Claude error] {e}"
    parts = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            txt = block.get("text", "").strip()
            if txt:
                parts.append(txt)
    return "\n".join(parts) if parts else "No clear answer found."
def answer_with_llama(
    context_text: str,
    question: str,
    model: str = "llama-3.1-8b-instant",
    max_tokens: int = 1024
) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        return f"[Groq init error] {e}"
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context_text}\n\n"
                f"QUESTION:\n{question}"
            )
        }
    ]
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception as e:
        return f"[Groq API error] {e}"
    if (
        not chat_completion or
        not chat_completion.choices or
        not chat_completion.choices[0].message
    ):
        return "No response from LLaMA."
    raw = chat_completion.choices[0].message.content
    text = _extract_llama_text(raw)
    return _clean_text(text)