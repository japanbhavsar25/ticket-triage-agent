"""Streamlit demo: paste a ticket -> classification, tool calls, approval step, draft reply."""

from __future__ import annotations

import json
import os
import uuid

import streamlit as st
from langgraph.types import Command

from triage.graph import APPROVAL_THRESHOLD, build_graph, pending_interrupt
from triage.llm import cost_usd, make_demo_llm
from triage.store import Store

MAX_RUNS_PER_SESSION = 15  # protects the shared free-tier Groq key
MAX_TICKET_CHARS = 1500

EXAMPLES = {
    "Big refund (needs approval)": "Order ORD-1002: the E-Reader arrived damaged. Requesting a refund.",
    "Small refund (auto)": "Hi, the USB-C Charger from order ORD-1001 stopped working after two days. "
    "I'd like a full refund please.",
    "Refund not allowed (not delivered)": "Can I get a refund on order ORD-1004? I changed my mind.",
    "Shipping": "Where is my order ORD-1013? It's been over a week.",
    "Account (urgent)": "Someone logged into my account and changed my address! Email: priya.singh@example.com. "
    "Please lock it NOW.",
    "Other": "Do you offer student discounts?",
}

st.set_page_config(page_title="Ticket Triage Agent", page_icon="🎫", layout="wide")

# Streamlit Community Cloud keeps secrets in st.secrets; locally we use .env.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
except Exception:  # no secrets.toml locally
    pass

ss = st.session_state
if "graph" not in ss:
    ss.store = Store()  # each browser session gets its own fake database
    ss.graph = build_graph(make_demo_llm(), store=ss.store)
    ss.runs = 0
    ss.thread = None


def config():
    return {"configurable": {"thread_id": ss.thread}}


st.title("🎫 Support-Ticket Triage Agent")
st.caption(
    "A LangGraph agent on Groq (`openai/gpt-oss-120b`, falling back to `gpt-oss-20b` when the free daily "
    "quota runs out). It classifies the ticket, looks up a fake "
    f"order/customer database with tools, **pauses for your approval before any refund over "
    f"${APPROVAL_THRESHOLD:.0f}**, then drafts a reply that cites the facts it used."
)

with st.sidebar:
    st.subheader("Try an example")
    choice = st.radio("Example tickets", list(EXAMPLES), label_visibility="collapsed")
    st.divider()
    st.subheader("Fake database")
    st.caption("Orders ORD-1001 … ORD-1060, customers like aarav.shah@example.com. Data is made up.")
    with st.expander("Browse orders"):
        st.dataframe(
            [
                {k: o[k] for k in ("order_id", "product", "total", "status", "refunded_amount")}
                for o in ss.store.data["orders"]
            ],
            hide_index=True,
        )
    if ss.store.refunds:
        st.write("Refunds issued this session:", ss.store.refunds)
    st.caption(f"Runs this session: {ss.runs}/{MAX_RUNS_PER_SESSION}")

ticket = st.text_area("Customer ticket", EXAMPLES[choice], height=110, max_chars=MAX_TICKET_CHARS)
if st.button("Triage ticket", type="primary", disabled=ss.runs >= MAX_RUNS_PER_SESSION):
    ss.runs += 1
    ss.thread = str(uuid.uuid4())
    with st.spinner("Classifying and calling tools…"):
        try:
            ss.graph.invoke({"ticket": ticket}, config())
        except Exception as exc:
            st.error(
                f"LLM call failed ({type(exc).__name__}). This demo runs on Groq's free tier, which has "
                "per-minute and per-day limits; please try again later."
            )
            ss.thread = None

if ss.thread:
    state = ss.graph.get_state(config()).values
    request = pending_interrupt(ss.graph, config())

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("1 · Classification")
        tri = state.get("triage", {})
        if tri:
            a, b = st.columns(2)
            a.metric("Category", tri["category"])
            b.metric("Urgency", tri["urgency"])
            st.write(tri["summary"])
            st.json({k: tri[k] for k in ("order_id", "email", "product", "refund_amount_requested")})

        st.subheader("2 · Tool calls")
        if not state.get("tool_log"):
            st.caption("No tools called yet." if request else "No tools needed.")
        for e in state.get("tool_log", []):
            args = ", ".join(f"{k}={v!r}" for k, v in e["args"].items())
            with st.expander(f"`{e['name']}({args})`", expanded=e["name"] == "issue_refund"):
                st.json(e["result"])

    with col2:
        st.subheader("3 · Human approval")
        if request:
            st.warning(
                f"The agent wants to refund **\\${float(request['amount']):.2f}** on **{request['order_id']}** "
                f"(over the \\${request['threshold']:.0f} limit). Approve?"
            )
            note = st.text_input("Note for the record (optional)")
            ok, no = st.columns(2)
            decision = None
            if ok.button("✅ Approve refund", use_container_width=True):
                decision = True
            if no.button("❌ Reject", use_container_width=True):
                decision = False
            if decision is not None:
                with st.spinner("Resuming the agent…"):
                    ss.graph.invoke(Command(resume={"approved": decision, "note": note}), config())
                st.rerun()
        elif state.get("approvals"):
            for a in state["approvals"]:
                verdict = "approved ✅" if a["approved"] else "rejected ❌"
                st.success(f"Refund of \\${float(a['amount']):.2f} on {a['order_id']} was {verdict} by you.")
        else:
            st.caption(f"Not needed (no refund over ${APPROVAL_THRESHOLD:.0f}).")

        st.subheader("4 · Draft reply")
        draft = state.get("draft")
        if draft:
            st.markdown(draft["reply"].replace("$", "\\$"))  # "$" would start LaTeX
            if draft["cited_facts"]:
                st.markdown("**Facts cited**")
                for i, f in enumerate(draft["cited_facts"], 1):
                    st.markdown(f"[{i}] `{f}`")
        elif request:
            st.caption("Waiting for your decision…")

    calls = state.get("llm_calls", [])
    if calls:
        tin = sum(c["input_tokens"] for c in calls)
        tout = sum(c["output_tokens"] for c in calls)
        secs = sum(c["seconds"] for c in calls)
        st.caption(
            f"Model: {', '.join(sorted({c.get('model') or '?' for c in calls}))} · "
            f"{len(calls)} LLM calls · {tin + tout:,} tokens · {secs:.1f}s of LLM time · "
            f"≈${sum(cost_usd(c['input_tokens'], c['output_tokens'], c.get('model') or '') for c in calls):.5f} "
            "at Groq list price"
        )
        with st.expander("Raw LLM call log"):
            st.code(json.dumps(calls, indent=1))
