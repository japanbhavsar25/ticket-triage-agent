"""The LangGraph triage agent.

    classify -> agent <-> tools -> draft
                  |        ^
                  +-> approval   (human-in-the-loop pause for refunds over ₹5,000)

The approval rule lives in code (route_after_agent), not in the prompt, so the
LLM cannot skip it.
"""

from __future__ import annotations

import json
import operator
import re
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from . import prompts
from .schemas import Draft, Triage
from .store import Store

APPROVAL_THRESHOLD = 5000.0  # Indian rupees (INR)
MAX_AGENT_TURNS = 4

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up an order by id. Returns status, product, total, refundable_amount.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "e.g. ORD-1001"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": "Look up a customer by email. Returns name, plan, account_status, order_ids.",
            "parameters": {
                "type": "object",
                "properties": {"email": {"type": "string"}},
                "required": ["email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Refund an amount in Indian rupees (INR) on a delivered order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
                "required": ["order_id", "amount"],
            },
        },
    },
]


class TicketState(TypedDict, total=False):
    ticket: str
    triage: dict
    messages: Annotated[list[AnyMessage], add_messages]
    tool_log: Annotated[list[dict], operator.add]
    approvals: Annotated[list[dict], operator.add]
    llm_calls: Annotated[list[dict], operator.add]
    turns: int
    draft: dict


def _usage(node: str, msg: Any, seconds: float) -> dict:
    meta = getattr(msg, "usage_metadata", None) or {}
    return {
        "node": node,
        "model": (getattr(msg, "response_metadata", None) or {}).get("model_name"),
        "input_tokens": meta.get("input_tokens", 0),
        "output_tokens": meta.get("output_tokens", 0),
        "seconds": round(seconds, 3),
    }


def _needs_approval(call: dict) -> bool:
    if call["name"] != "issue_refund":
        return False
    try:
        return float(call["args"].get("amount", 0)) > APPROVAL_THRESHOLD
    except (TypeError, ValueError):
        return True  # unparseable amount: let a human look at it


def build_graph(llm, store: Store | None = None, checkpointer=None):
    store = store or Store()
    tool_llm = llm.bind_tools(TOOLS)
    classify_llm = llm.with_structured_output(Triage, method="json_schema", include_raw=True)
    draft_llm = llm.with_structured_output(Draft, method="json_schema", include_raw=True)

    def classify(state: TicketState):
        t0 = time.perf_counter()
        out = classify_llm.invoke([SystemMessage(prompts.CLASSIFY), HumanMessage(state["ticket"])])
        triage: Triage = out["parsed"]
        if triage is None:
            raise ValueError(f"could not parse triage output: {out.get('parsing_error')}")
        triage_json = triage.model_dump_json()
        return {
            "triage": triage.model_dump(),
            "messages": [
                SystemMessage(prompts.AGENT.format(triage=triage_json)),
                HumanMessage(state["ticket"]),
            ],
            "llm_calls": [_usage("classify", out["raw"], time.perf_counter() - t0)],
            "turns": 0,
        }

    def agent(state: TicketState):
        t0 = time.perf_counter()
        msg = tool_llm.invoke(state["messages"])
        return {
            "messages": [msg],
            "turns": state.get("turns", 0) + 1,
            "llm_calls": [_usage("agent", msg, time.perf_counter() - t0)],
        }

    def route_after_agent(state: TicketState) -> str:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        if not calls:
            return "draft"
        if state.get("turns", 0) > MAX_AGENT_TURNS:
            return "draft"
        if any(_needs_approval(c) for c in calls):
            return "approval"
        return "tools"

    def approval(state: TicketState):
        # interrupt() pauses the graph here; the UI / eval resumes it with
        # Command(resume={"approved": bool, "note": str}).
        decisions = []
        for call in state["messages"][-1].tool_calls:
            if not _needs_approval(call):
                continue
            answer = interrupt(
                {
                    "type": "refund_approval",
                    "tool_call_id": call["id"],
                    "order_id": call["args"].get("order_id"),
                    "amount": call["args"].get("amount"),
                    "threshold": APPROVAL_THRESHOLD,
                    "ticket_summary": state["triage"].get("summary"),
                }
            )
            decisions.append(
                {
                    "tool_call_id": call["id"],
                    "order_id": call["args"].get("order_id"),
                    "amount": call["args"].get("amount"),
                    "approved": bool(answer.get("approved")),
                    "note": answer.get("note", ""),
                }
            )
        return {"approvals": decisions}

    def tools(state: TicketState):
        approved = {a["tool_call_id"]: a for a in state.get("approvals", [])}
        messages, log = [], []
        for call in state["messages"][-1].tool_calls:
            name, args = call["name"], call["args"]
            if name == "get_order":
                result = store.get_order(args.get("order_id", ""))
            elif name == "get_customer":
                result = store.get_customer(args.get("email", ""))
            elif name == "issue_refund":
                decision = approved.get(call["id"])
                if _needs_approval(call) and not (decision and decision["approved"]):
                    result = {
                        "success": False,
                        "error": "rejected by human reviewer",
                        "note": decision["note"] if decision else "",
                    }
                else:
                    result = store.issue_refund(args.get("order_id", ""), args.get("amount", 0))
            else:
                result = {"error": f"unknown tool {name}"}
            log.append({"name": name, "args": args, "result": result})
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=call["id"]))
        return {"messages": messages, "tool_log": log}

    def draft(state: TicketState):
        tool_log = state.get("tool_log", [])
        messages = [
            SystemMessage(prompts.DRAFT_SYSTEM.format(outcome=refund_outcome(tool_log))),
            HumanMessage(
                prompts.DRAFT_USER.format(
                    ticket=state["ticket"],
                    triage=json.dumps(state["triage"]),
                    tool_log=json.dumps(tool_log, indent=1),
                )
            ),
        ]
        calls, result = [], None
        for _ in range(2):  # one retry if the self-check fails
            t0 = time.perf_counter()
            out = draft_llm.invoke(messages)
            calls.append(_usage("draft", out["raw"], time.perf_counter() - t0))
            parsed: Draft | None = out["parsed"]
            result = parsed.model_dump() if parsed else {"reply": "(could not draft a reply)", "cited_facts": []}
            problem = check_draft(result["reply"], tool_log)
            if problem is None:
                break
            messages.append(HumanMessage(f"Your reply is wrong: {problem} Rewrite it."))
        return {"draft": result, "llm_calls": calls}

    g = StateGraph(TicketState)
    g.add_node("classify", classify)
    g.add_node("agent", agent)
    g.add_node("approval", approval)
    g.add_node("tools", tools)
    g.add_node("draft", draft)
    g.add_edge(START, "classify")
    g.add_edge("classify", "agent")
    g.add_conditional_edges("agent", route_after_agent, ["tools", "approval", "draft"])
    g.add_edge("approval", "tools")
    g.add_edge("tools", "agent")
    g.add_edge("draft", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def pending_interrupt(graph, config) -> dict | None:
    """Return the approval request the graph is paused on, or None."""
    snap = graph.get_state(config)
    for task in snap.tasks:
        for intr in task.interrupts:
            return intr.value
    return None


def run_ticket(graph, ticket: str, thread_id: str, approve=lambda req: True) -> dict:
    """Run a ticket to the end, answering every approval pause with approve(request).

    Used by the CLI and the eval; the Streamlit app resumes manually instead.
    """
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke({"ticket": ticket}, config)
    pauses = []
    while (req := pending_interrupt(graph, config)) is not None:
        ok = approve(req)
        pauses.append({**req, "approved": ok})
        graph.invoke(Command(resume={"approved": ok, "note": "eval/cli decision"}), config)
    state = graph.get_state(config).values
    return {**state, "pauses": pauses}


_REFUND_CLAIM = re.compile(
    r"\b(issued|processed|initiated|approved|credited|refunded)\b[^.!?]*\brefund"
    r"|\brefund\w*\b[^.!?]*\b(has been|was|is being|have been|issued|processed|initiated|credited)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\b(not|unable|cannot|no)\b|n't\b", re.IGNORECASE)


def check_draft(reply: str, tool_log: list[dict]) -> str | None:
    """Cheap self-check: flag a reply that claims a refund the tools never issued."""
    issued = any(e["name"] == "issue_refund" and e["result"].get("success") for e in tool_log)
    if issued:
        return None
    for sentence in re.split(r"(?<=[.!?])\s+", reply):
        if _REFUND_CLAIM.search(sentence) and not _NEGATION.search(sentence):
            return f'it claims a refund was made ("{sentence.strip()}") but no refund was issued.'
    return None


def refund_outcome(tool_log: list[dict]) -> str:
    """Plain-English refund outcome, computed in code so the draft can't invent one."""
    lines = []
    for entry in tool_log:
        if entry["name"] != "issue_refund":
            continue
        r = entry["result"]
        if r.get("success"):
            lines.append(f"Refund {r['refund_id']} of ₹{r['amount']:,.2f} was issued on {r['order_id']}.")
        else:
            lines.append(f"A refund on {entry['args'].get('order_id')} was NOT issued: {r.get('error')}.")
    return " ".join(lines) or "No refund was issued for this ticket. Do not say or imply that one was."
