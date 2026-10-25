from conftest import DONE, FakeLLM, tool_turn
from langgraph.types import Command

from triage.graph import build_graph, check_draft, pending_interrupt, refund_outcome, run_ticket
from triage.schemas import Draft, Triage
from triage.store import Store

REFUND = Triage(category="refund", urgency="medium", order_id="ORD-2", summary="wants refund")
OK_DRAFT = Draft(reply="Done [1].", cited_facts=["get_order(ORD-2): status = delivered"])


def make(store_data, triage, turns, drafts=(OK_DRAFT,)):
    fake = FakeLLM(triage, turns, list(drafts))
    store = Store(store_data)
    return build_graph(fake, store=store), store, fake


def test_small_refund_runs_without_pause(store_data):
    turns = [
        tool_turn(("get_order", {"order_id": "ORD-1"})),
        tool_turn(("issue_refund", {"order_id": "ORD-1", "amount": 20.0})),
        DONE,
    ]
    graph, store, _ = make(store_data, REFUND, turns)
    out = run_ticket(graph, "refund ORD-1", "t1")
    assert out["pauses"] == []
    assert store.refunds[0]["amount"] == 20.0
    assert [e["name"] for e in out["tool_log"]] == ["get_order", "issue_refund"]
    assert out["draft"]["reply"] == "Done [1]."


def test_big_refund_pauses_and_approval_issues_it(store_data):
    turns = [
        tool_turn(("get_order", {"order_id": "ORD-2"})),
        tool_turn(("issue_refund", {"order_id": "ORD-2", "amount": 199.0})),
        DONE,
    ]
    graph, store, _ = make(store_data, REFUND, turns)
    config = {"configurable": {"thread_id": "t2"}}
    graph.invoke({"ticket": "refund ORD-2"}, config)

    req = pending_interrupt(graph, config)
    assert req["type"] == "refund_approval" and req["amount"] == 199.0
    assert store.refunds == []  # nothing happens until a human decides

    graph.invoke(Command(resume={"approved": True}), config)
    assert pending_interrupt(graph, config) is None
    assert store.refunds[0]["order_id"] == "ORD-2"


def test_big_refund_rejected_by_human(store_data):
    turns = [tool_turn(("issue_refund", {"order_id": "ORD-2", "amount": 199.0})), DONE]
    graph, store, _ = make(store_data, REFUND, turns)
    out = run_ticket(graph, "refund ORD-2", "t3", approve=lambda req: False)
    assert out["pauses"][0]["approved"] is False
    assert store.refunds == []
    assert out["tool_log"][0]["result"]["error"] == "rejected by human reviewer"


def test_draft_self_check_retries_false_refund_claim(store_data):
    triage = Triage(category="refund", urgency="low", order_id="ORD-3", summary="x")
    turns = [tool_turn(("get_order", {"order_id": "ORD-3"})), DONE]
    bad = Draft(reply="Your refund has been processed.", cited_facts=[])
    good = Draft(reply="Your order has not been delivered yet, so we can't refund it yet.", cited_facts=[])
    graph, _, fake = make(store_data, triage, turns, drafts=[bad, good])
    out = run_ticket(graph, "refund ORD-3", "t4")
    assert out["draft"]["reply"] == good.reply
    assert sum(1 for c in out["llm_calls"] if c["node"] == "draft") == 2


def test_agent_turn_limit_stops_loops(store_data):
    looping = [tool_turn(("get_order", {"order_id": "ORD-1"})) for _ in range(10)]
    graph, _, _ = make(store_data, REFUND, looping)
    out = run_ticket(graph, "loop", "t5")
    assert "draft" in out
    assert len(out["tool_log"]) <= 5


def test_refund_outcome_and_check_draft():
    assert "No refund" in refund_outcome([])
    log = [
        {
            "name": "issue_refund",
            "args": {"order_id": "ORD-1"},
            "result": {"success": True, "refund_id": "RF-0001", "order_id": "ORD-1", "amount": 20.0},
        }
    ]
    assert "RF-0001" in refund_outcome(log)
    assert check_draft("I've processed your refund.", log) is None
    assert check_draft("I've processed your refund.", []) is not None
    assert check_draft("We cannot issue a refund until it is delivered.", []) is None
