"""A scripted fake LLM so tests never call Groq."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage.schemas import Draft, Triage  # noqa: E402


class _Structured:
    def __init__(self, fake, schema):
        self.fake, self.schema = fake, schema

    def invoke(self, messages):
        self.fake.seen.append((self.schema.__name__, messages))
        value = self.fake.structured[self.schema.__name__].pop(0)
        raw = AIMessage(content="", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        return {"raw": raw, "parsed": value, "parsing_error": None}


class _Tools:
    def __init__(self, fake):
        self.fake = fake

    def invoke(self, messages):
        self.fake.seen.append(("tools", messages))
        return self.fake.tool_turns.pop(0)


class FakeLLM:
    def __init__(self, triage: Triage, tool_turns: list[AIMessage], drafts: list[Draft]):
        self.structured = {"Triage": [triage], "Draft": list(drafts)}
        self.tool_turns = list(tool_turns)
        self.seen: list = []

    def bind_tools(self, tools):
        return _Tools(self)

    def with_structured_output(self, schema, **kwargs):
        return _Structured(self, schema)


def tool_turn(*calls: tuple[str, dict]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": n, "args": a, "id": f"call_{i}"} for i, (n, a) in enumerate(calls)],
    )


DONE = AIMessage(content="done")


@pytest.fixture
def store_data():
    return {
        "customers": [
            {
                "customer_id": "C-1",
                "name": "Ann Lee",
                "email": "ann@example.com",
                "plan": "pro",
                "account_status": "active",
                "member_since": "2024-01-01",
            },
        ],
        "orders": [
            {
                "order_id": "ORD-1",
                "email": "ann@example.com",
                "product": "Mouse",
                "total": 20.0,
                "status": "delivered",
                "order_date": "2026-09-01",
                "delivered_date": "2026-09-05",
                "carrier": "UPS",
                "refunded_amount": 0.0,
            },
            {
                "order_id": "ORD-2",
                "email": "ann@example.com",
                "product": "Headphones",
                "total": 199.0,
                "status": "delivered",
                "order_date": "2026-09-01",
                "delivered_date": "2026-09-05",
                "carrier": "UPS",
                "refunded_amount": 0.0,
            },
            {
                "order_id": "ORD-3",
                "email": "ann@example.com",
                "product": "Lamp",
                "total": 30.0,
                "status": "shipped",
                "order_date": "2026-09-01",
                "delivered_date": None,
                "carrier": "DHL",
                "refunded_amount": 0.0,
            },
        ],
    }
