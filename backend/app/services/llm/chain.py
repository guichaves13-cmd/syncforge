"""4-tier LLM chain — Groq → Gemini → Cerebras → OpenRouter → DeepSeek.

Each provider returns text. We try them in order, with quick retry on 429."""
from __future__ import annotations
import os
import time
from dataclasses import dataclass

import requests


@dataclass
class LLMKeys:
    groq: str = ""
    gemini: str = ""
    cerebras: str = ""
    openrouter: str = ""
    deepseek: str = ""

    @classmethod
    def from_env(cls) -> "LLMKeys":
        return cls(
            groq=os.getenv("GROQ_API_KEY", ""),
            gemini=os.getenv("GEMINI_API_KEY", ""),
            cerebras=os.getenv("CEREBRAS_API_KEY", ""),
            openrouter=os.getenv("OPENROUTER_API_KEY", ""),
            deepseek=os.getenv("DEEPSEEK_API_KEY", ""),
        )


def call_chain(prompt: str, keys: LLMKeys, *, max_tokens: int = 1500,
               temperature: float = 0.7, json_mode: bool = False) -> str | None:
    """Returns text from the first provider that responds; None if all fail."""
    for fn in (_groq, _gemini, _cerebras, _openrouter, _deepseek):
        try:
            out = fn(prompt, keys, max_tokens, temperature, json_mode)
            if out:
                return out
        except Exception:
            continue
    return None


# ─── Provider impls ──────────────────────────────────────────────────────

def _groq(prompt, k: LLMKeys, max_tokens, temp, json_mode):
    if not k.groq:
        return None
    body = {"model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temp}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {k.groq}",
                 "Content-Type": "application/json"},
        json=body, timeout=30,
    )
    if r.status_code == 429:
        return None
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _gemini(prompt, k: LLMKeys, max_tokens, temp, json_mode):
    if not k.gemini:
        return None
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=k.gemini)
    cfg = types.GenerateContentConfig(
        temperature=temp, max_output_tokens=max_tokens,
        response_mime_type="application/json" if json_mode else "text/plain",
    )
    resp = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt, config=cfg,
    )
    return resp.text


def _cerebras(prompt, k: LLMKeys, max_tokens, temp, json_mode):
    if not k.cerebras:
        return None
    body = {"model": "llama-3.3-70b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temp}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(
        "https://api.cerebras.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {k.cerebras}",
                 "Content-Type": "application/json"},
        json=body, timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _openrouter(prompt, k: LLMKeys, max_tokens, temp, json_mode):
    if not k.openrouter:
        return None
    body = {"model": "meta-llama/llama-3.3-70b-instruct:free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temp}
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {k.openrouter}"},
        json=body, timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _deepseek(prompt, k: LLMKeys, max_tokens, temp, json_mode):
    if not k.deepseek:
        return None
    body = {"model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temp}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {k.deepseek}"},
        json=body, timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
