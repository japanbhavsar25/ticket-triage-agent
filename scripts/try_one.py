"""Run one ticket from the command line:  python scripts/try_one.py "ticket text" """

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage.graph import build_graph, run_ticket  # noqa: E402
from triage.llm import make_llm  # noqa: E402

ticket = sys.argv[1] if len(sys.argv) > 1 else "Order ORD-1004 arrived broken, please refund it."
graph = build_graph(make_llm())
out = run_ticket(graph, ticket, "cli")
print(json.dumps({k: out[k] for k in ("triage", "tool_log", "pauses", "draft", "llm_calls")}, indent=1, default=str))
