"""Run the agent on the labelled tickets and report metrics.

    python eval/run_eval.py              # all tickets
    python eval/run_eval.py --limit 10   # quick smoke run

Every refund approval request is auto-approved here; the metric checks that the
agent paused exactly when it should (refund > $50), not what the human decided.
Writes eval/results/results.jsonl (per ticket) and eval/results/summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from triage.graph import build_graph, run_ticket  # noqa: E402
from triage.llm import cost_usd, make_llm  # noqa: E402
from triage.store import Store  # noqa: E402

TOKENS_PER_MINUTE = 8000  # Groq free-tier limit we measured from the x-ratelimit headers


def norm_call(call: dict) -> tuple:
    args = call["args"]
    if call["name"] == "get_order":
        return ("get_order", str(args.get("order_id", "")).strip().upper())
    if call["name"] == "get_customer":
        return ("get_customer", str(args.get("email", "")).strip().lower())
    if call["name"] == "issue_refund":
        try:
            amount = round(float(args.get("amount", 0)), 2)
        except (TypeError, ValueError):
            amount = None
        return ("issue_refund", str(args.get("order_id", "")).strip().upper(), amount)
    return (call["name"], json.dumps(args, sort_keys=True))


def same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return str(a).strip().lower() == str(b).strip().lower()


def score(t: dict, state: dict, error: str | None, latency: float) -> dict:
    triage = state.get("triage", {})
    calls = state.get("llm_calls", [])
    got_tools = Counter(norm_call(c) for c in state.get("tool_log", []))
    want_tools = Counter(norm_call(c) for c in t["expected_tools"])
    paused = bool(state.get("pauses"))
    return {
        "id": t["id"],
        "error": error,
        "category_ok": triage.get("category") == t["category"],
        "urgency_ok": triage.get("urgency") == t["urgency"],
        "order_id_ok": same(triage.get("order_id"), t["order_id"]),
        "email_ok": same(triage.get("email"), t["email"]),
        "tools_ok": got_tools == want_tools,
        "approval_ok": paused == t["needs_approval"],
        "paused": paused,
        "needs_approval": t["needs_approval"],
        "latency_s": round(latency, 2),
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "llm_calls": len(calls),
        "expected": {"category": t["category"], "urgency": t["urgency"], "tools": t["expected_tools"]},
        "got": {
            "category": triage.get("category"),
            "urgency": triage.get("urgency"),
            "tools": [{"name": c["name"], "args": c["args"]} for c in state.get("tool_log", [])],
        },
        "reply": state.get("draft", {}).get("reply"),
    }


def summarize(rows: list[dict], model: str) -> dict:
    """Metrics over the tickets that completed. Tickets that errored (e.g. the
    free-tier daily quota ran out) are reported separately, not counted as wrong."""
    ok = [r for r in rows if not r["error"]]
    n = len(ok)
    lat = sorted(r["latency_s"] for r in ok)

    def pct(key: str) -> float | None:
        return round(100 * sum(r[key] for r in ok) / n, 1) if n else None

    approval_rows = [r for r in ok if r["needs_approval"]]
    avg_in = statistics.mean(r["input_tokens"] for r in ok) if ok else 0
    avg_out = statistics.mean(r["output_tokens"] for r in ok) if ok else 0
    return {
        "model": model,
        "tickets_total": len(rows),
        "tickets_scored": n,
        "tickets_errored": len(rows) - n,
        "routing_accuracy_pct": pct("category_ok"),
        "urgency_accuracy_pct": pct("urgency_ok"),
        "order_id_extraction_pct": pct("order_id_ok"),
        "email_extraction_pct": pct("email_ok"),
        "tool_call_accuracy_pct": pct("tools_ok"),
        "approval_gate_correct_pct": pct("approval_ok"),
        "approval_needed_tickets": len(approval_rows),
        "approval_needed_and_paused": sum(r["paused"] for r in approval_rows),
        "paused_without_need": sum(r["paused"] and not r["needs_approval"] for r in ok),
        "avg_latency_s": round(statistics.mean(lat), 2) if lat else None,
        "p50_latency_s": lat[len(lat) // 2] if lat else None,
        "p95_latency_s": lat[math.ceil(len(lat) * 0.95) - 1] if lat else None,
        "avg_llm_calls": round(statistics.mean(r["llm_calls"] for r in ok), 2) if ok else None,
        "avg_input_tokens": round(avg_in),
        "avg_output_tokens": round(avg_out),
        "avg_cost_usd_at_list_price": round(cost_usd(avg_in, avg_out), 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model")
    ap.add_argument(
        "--rerun-errors",
        action="store_true",
        help="re-run only the tickets that errored in results.jsonl and merge them in",
    )
    args = ap.parse_args()

    tickets = [json.loads(line) for line in (ROOT / "eval" / "tickets.jsonl").read_text().splitlines()]
    if args.limit:
        tickets = tickets[: args.limit]
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(exist_ok=True)
    results_path = out_dir / "results.jsonl"

    previous: dict[str, dict] = {}
    if args.rerun_errors:
        previous = {r["id"]: r for r in map(json.loads, results_path.read_text().splitlines())}
        tickets = [t for t in tickets if t["id"] not in previous or previous[t["id"]]["error"]]
        print(f"re-running {len(tickets)} errored tickets")

    llm = make_llm(args.model)
    window: deque[tuple[float, int]] = deque()  # (timestamp, tokens) for client-side pacing
    new_rows: dict[str, dict] = {}
    quota_gone = False

    for i, t in enumerate(tickets, 1):
        if quota_gone:
            new_rows[t["id"]] = score(t, {}, "skipped: daily token quota exhausted", 0.0)
            continue
        # Pace ourselves under the free-tier tokens-per-minute limit. Waiting time is
        # NOT counted in latency.
        while True:
            now = time.time()
            while window and now - window[0][0] > 60:
                window.popleft()
            if sum(n for _, n in window) < TOKENS_PER_MINUTE - 4000:
                break
            time.sleep(2)

        graph = build_graph(llm, store=Store())
        t0 = time.perf_counter()
        error = None
        try:
            state = run_ticket(graph, t["ticket"], thread_id=t["id"])
        except Exception as exc:  # record the ticket as errored, keep going
            state, error = {}, f"{type(exc).__name__}: {exc}"
            quota_gone = "tokens per day" in str(exc)
        latency = time.perf_counter() - t0
        row = score(t, state, error, latency)
        window.append((time.time(), row["input_tokens"] + row["output_tokens"]))
        new_rows[t["id"]] = row
        flags = "".join("." if row[k] else "x" for k in ("category_ok", "tools_ok", "approval_ok"))
        tok = row["input_tokens"] + row["output_tokens"]
        print(f"[{i}/{len(tickets)}] {t['id']} {flags} {latency:.1f}s {tok} tok {(error or '')[:120]}", flush=True)

    merged = {**previous, **new_rows}
    rows = sorted(merged.values(), key=lambda r: r["id"])
    with results_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary = summarize(rows, llm.model_name)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
