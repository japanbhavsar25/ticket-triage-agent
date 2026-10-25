import httpx
import openai
from langchain_core.runnables import RunnableLambda

from triage.llm import FallbackLLM, cost_usd


def _rate_limited(_):
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    raise openai.RateLimitError("tokens per day", response=httpx.Response(429, request=req), body=None)


class _Model:
    def __init__(self, fn, name):
        self.fn, self.model_name = fn, name

    def bind_tools(self, tools):
        return RunnableLambda(self.fn)

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(self.fn)


def test_fallback_used_when_primary_is_rate_limited():
    llm = FallbackLLM(_Model(_rate_limited, "big"), _Model(lambda _: "backup answer", "small"))
    assert llm.bind_tools([]).invoke("hi") == "backup answer"
    assert llm.with_structured_output(dict).invoke("hi") == "backup answer"


def test_primary_used_when_healthy():
    llm = FallbackLLM(_Model(lambda _: "primary", "big"), _Model(lambda _: "backup", "small"))
    assert llm.bind_tools([]).invoke("hi") == "primary"


def test_cost():
    assert cost_usd(1_000_000, 1_000_000) == 0.75
