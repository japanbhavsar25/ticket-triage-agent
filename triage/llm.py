"""Groq LLM factory (Groq exposes an OpenAI-compatible API)."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
# Groq list price for openai/gpt-oss-120b, USD per 1M tokens (console.groq.com/docs/models, 2026-10-25).
PRICE_PER_M = {"input": 0.15, "output": 0.60}


def make_llm(model: str | None = None) -> ChatOpenAI:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set (put it in .env or the host's secrets)")
    return ChatOpenAI(
        model=model or os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        base_url=GROQ_BASE_URL,
        api_key=api_key,
        temperature=0,
        reasoning_effort="low",
        max_retries=8,
        timeout=60,
    )


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * PRICE_PER_M["input"] + output_tokens * PRICE_PER_M["output"]) / 1_000_000
