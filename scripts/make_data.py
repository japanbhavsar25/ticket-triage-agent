"""Generate the fake store (triage/data/store.json) and the labelled eval set
(eval/tickets.jsonl).

Tickets are built from hand-written templates, so every label (category,
urgency, expected tool calls, whether a refund needs approval) is known by
construction instead of being guessed by an LLM. Run:  python scripts/make_data.py
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rng = random.Random(42)

FIRST = [
    "Aarav",
    "Maya",
    "Liam",
    "Sofia",
    "Noah",
    "Priya",
    "Ethan",
    "Zara",
    "Omar",
    "Chloe",
    "Kenji",
    "Isla",
    "Mateo",
    "Ananya",
    "Lucas",
    "Fatima",
    "Ravi",
    "Emma",
    "Diego",
    "Hana",
    "Arjun",
    "Nora",
    "Felix",
    "Leah",
    "Sam",
]
LAST = [
    "Shah",
    "Patel",
    "Chen",
    "Garcia",
    "Kim",
    "Singh",
    "Brown",
    "Ali",
    "Rossi",
    "Mehta",
    "Tanaka",
    "Nguyen",
    "Lopez",
    "Iyer",
    "Muller",
    "Khan",
    "Rao",
    "Wilson",
    "Silva",
    "Sato",
    "Desai",
    "Olsen",
    "Weber",
    "Cohen",
    "Park",
]
PRODUCTS = {
    "USB-C Charger": 19.99,
    "Wireless Mouse": 24.50,
    "Phone Case": 14.99,
    "Bluetooth Speaker": 49.00,
    "Mechanical Keyboard": 89.99,
    "Noise-Cancelling Headphones": 199.00,
    "Smart Watch": 149.50,
    "Webcam HD": 59.99,
    "Laptop Stand": 34.00,
    "Portable SSD 1TB": 109.00,
    "Fitness Tracker": 45.00,
    "Desk Lamp": 29.95,
    "Robot Vacuum": 279.00,
    "E-Reader": 129.99,
}


def make_store() -> dict:
    customers = []
    for i, (first, last) in enumerate(zip(FIRST, LAST, strict=True)):
        customers.append(
            {
                "customer_id": f"C-{200 + i}",
                "name": f"{first} {last}",
                "email": f"{first.lower()}.{last.lower()}@example.com",
                "plan": rng.choice(["free", "plus", "plus", "pro"]),
                "account_status": "locked" if i % 9 == 4 else "active",
                "member_since": str(date(2022, 1, 1) + timedelta(days=rng.randint(0, 1200))),
            }
        )
    orders = []
    statuses = ["delivered"] * 5 + ["shipped", "shipped", "processing", "cancelled"]
    for i in range(60):
        cust = customers[i % len(customers)]
        product = rng.choice(list(PRODUCTS))
        status = rng.choice(statuses)
        ordered = date(2026, 9, 1) + timedelta(days=rng.randint(0, 45))
        total = PRODUCTS[product]
        orders.append(
            {
                "order_id": f"ORD-{1001 + i}",
                "email": cust["email"],
                "product": product,
                "total": total,
                "status": status,
                "order_date": str(ordered),
                "delivered_date": str(ordered + timedelta(days=4)) if status == "delivered" else None,
                "carrier": rng.choice(["UPS", "FedEx", "DHL"]) if status in ("shipped", "delivered") else None,
                "refunded_amount": total if (status == "delivered" and i % 11 == 3) else 0.0,
            }
        )
    return {"customers": customers, "orders": orders}


def call(name: str, **args) -> dict:
    return {"name": name, "args": args}


def make_tickets(store: dict) -> list[dict]:
    orders = store["orders"]
    customers = store["customers"]
    by_status = {
        s: [o for o in orders if o["status"] == s] for s in ("delivered", "shipped", "processing", "cancelled")
    }
    deliv_open = [o for o in by_status["delivered"] if o["refunded_amount"] == 0]
    deliv_refunded = [o for o in by_status["delivered"] if o["refunded_amount"] > 0]
    small = [o for o in deliv_open if o["total"] <= 50]
    big = [o for o in deliv_open if o["total"] > 50]
    not_delivered = by_status["shipped"] + by_status["processing"]
    tickets: list[dict] = []

    def add(text, category, urgency, calls, order_id=None, email=None, product=None, refund_amount=None):
        tickets.append(
            {
                "id": f"T{len(tickets) + 1:03d}",
                "ticket": text,
                "category": category,
                "urgency": urgency,
                "order_id": order_id,
                "email": email,
                "product": product,
                "expected_tools": calls,
                "refund_amount": refund_amount,
                "needs_approval": refund_amount is not None and refund_amount > 50,
            }
        )

    # ---- refund: full refund of a delivered order (small and large amounts) ----
    full_tpl = [
        ("Hi, the {p} from order {o} stopped working after two days. I'd like a full refund please.", "medium"),
        ("I want my money back for order {o}. The {p} is not what was advertised.", "medium"),
        ("Please refund order {o}, I returned the {p} last week and haven't heard anything.", "medium"),
        ("Order {o}: the {p} arrived damaged. Requesting a refund.", "medium"),
        ("URGENT - I need a refund for {o} today, the {p} is defective and I can't afford to wait.", "high"),
    ]
    for i, o in enumerate(small[:6] + big[:8]):
        tpl, urg = full_tpl[i % len(full_tpl)]
        add(
            tpl.format(o=o["order_id"], p=o["product"]),
            "refund",
            urg,
            [
                call("get_order", order_id=o["order_id"]),
                call("issue_refund", order_id=o["order_id"], amount=o["total"]),
            ],
            order_id=o["order_id"],
            product=o["product"],
            refund_amount=o["total"],
        )

    # ---- refund: partial amount stated ----
    for o in big[8:11] + small[6:8]:
        amt = round(o["total"] / 2, 2) if o["total"] > 50 else 10.0
        add(
            f"Order {o['order_id']} - the {o['product']} is missing a part. Could you refund ${amt:.2f} of the price? I'll keep it.",
            "refund",
            "low",
            [call("get_order", order_id=o["order_id"]), call("issue_refund", order_id=o["order_id"], amount=amt)],
            order_id=o["order_id"],
            product=o["product"],
            refund_amount=amt,
        )

    # ---- refund with email included ----
    for o in big[11:13] + small[8:9]:
        add(
            f"Hello, my email is {o['email']}. I'd like a refund for order {o['order_id']} ({o['product']}), it doesn't fit my needs.",
            "refund",
            "medium",
            [
                call("get_order", order_id=o["order_id"]),
                call("get_customer", email=o["email"]),
                call("issue_refund", order_id=o["order_id"], amount=o["total"]),
            ],
            order_id=o["order_id"],
            email=o["email"],
            product=o["product"],
            refund_amount=o["total"],
        )

    # ---- refund that must NOT be issued (not delivered / already refunded / unknown order / too much) ----
    for o in not_delivered[:3]:
        add(
            f"Can I get a refund on order {o['order_id']}? I changed my mind about the {o['product']}.",
            "refund",
            "low",
            [call("get_order", order_id=o["order_id"])],
            order_id=o["order_id"],
            product=o["product"],
        )
    for o in deliv_refunded[:2]:
        add(
            f"I still want a refund for {o['order_id']}, the {o['product']} broke.",
            "refund",
            "medium",
            [call("get_order", order_id=o["order_id"])],
            order_id=o["order_id"],
            product=o["product"],
        )
    for oid in ["ORD-9001", "ORD-7777"]:
        add(
            f"Refund my order {oid} immediately, the item never worked.",
            "refund",
            "high",
            [call("get_order", order_id=oid)],
            order_id=oid,
        )
    for o in small[9:11]:
        add(
            f"Refund ${o['total'] + 100:.2f} for order {o['order_id']} for all the trouble the {o['product']} caused me.",
            "refund",
            "medium",
            [call("get_order", order_id=o["order_id"])],
            order_id=o["order_id"],
            product=o["product"],
        )

    # ---- billing ----
    billing_order = [
        ("I was charged twice for order {o}! Two identical charges on my card. Please fix this now.", "high"),
        ("Can you send me an invoice for order {o}? I need it for my company expenses.", "low"),
        ("The amount on my statement for {o} doesn't match the price of the {p}. Why?", "medium"),
        ("Do you charge sales tax on order {o}? I can't see it on the receipt.", "low"),
    ]
    for i, o in enumerate(orders[5:17]):
        tpl, urg = billing_order[i % len(billing_order)]
        add(
            tpl.format(o=o["order_id"], p=o["product"]),
            "billing",
            urg,
            [call("get_order", order_id=o["order_id"])],
            order_id=o["order_id"],
            product=o["product"] if "{p}" in tpl else None,
        )
    for c in customers[3:6]:
        add(
            f"My plan renewal charge looks wrong. Account email {c['email']}. Was I billed for the wrong plan?",
            "billing",
            "medium",
            [call("get_customer", email=c["email"])],
            email=c["email"],
        )

    # ---- shipping ----
    ship_tpl = [
        ("Where is my order {o}? It's been over a week.", "medium"),
        ("Has order {o} shipped yet? Just checking, no rush.", "low"),
        (
            "My {p} (order {o}) was supposed to arrive for my daughter's birthday tomorrow, it's not here. Please help ASAP!",
            "high",
        ),
        ("Tracking for {o} hasn't updated in days. Which carrier is it with?", "medium"),
    ]
    for i, o in enumerate(by_status["shipped"][3:] + by_status["processing"][1:] + by_status["delivered"][-4:]):
        if i >= 18:
            break
        tpl, urg = ship_tpl[i % len(ship_tpl)]
        add(
            tpl.format(o=o["order_id"], p=o["product"]),
            "shipping",
            urg,
            [call("get_order", order_id=o["order_id"])],
            order_id=o["order_id"],
            product=o["product"] if "{p}" in tpl else None,
        )

    # ---- technical ----
    tech_tpl = [
        ("My {p} won't turn on anymore. I tried charging it overnight. Order {o}.", "medium", True),
        ("How do I pair the {p} with my laptop? The manual is unclear.", "low", False),
        ("The {p} keeps disconnecting every few minutes. Is there a firmware update?", "medium", False),
        ("Your app crashes every time I open the order history page. I can't see anything!", "medium", False),
        ("The {p} from order {o} overheated and started smoking. This is dangerous!", "high", True),
    ]
    tech_orders = orders[20:35]
    for i, o in enumerate(tech_orders):
        tpl, urg, has_order = tech_tpl[i % len(tech_tpl)]
        has_product = "{p}" in tpl
        add(
            tpl.format(o=o["order_id"], p=o["product"]),
            "technical",
            urg,
            [call("get_order", order_id=o["order_id"])] if has_order else [],
            order_id=o["order_id"] if has_order else None,
            product=o["product"] if has_product else None,
        )

    # ---- account ----
    acct_tpl = [
        ("I forgot my password and the reset email never arrives. My email is {e}.", "medium"),
        ("Someone logged into my account and changed my address! Email: {e}. Please lock it NOW.", "high"),
        ("Please change the email on my account from {e} to a new one, how do I do that?", "low"),
        (
            "My account {e} says it is locked. I can't log in at all and I need to place an order for work today.",
            "high",
        ),
        ("How do I delete my account? It's registered under {e}.", "low"),
    ]
    for i, c in enumerate(customers[8:23]):
        tpl, urg = acct_tpl[i % len(acct_tpl)]
        add(tpl.format(e=c["email"]), "account", urg, [call("get_customer", email=c["email"])], email=c["email"])

    # ---- other ----
    other = [
        ("Do you have any job openings in your customer support team?", "low"),
        ("I just wanted to say your delivery driver was super friendly. Thanks!", "low"),
        ("Are you planning to open a store in Toronto?", "low"),
        ("I'm a journalist writing about e-commerce, can I interview someone from your team?", "low"),
        ("Do you offer student discounts?", "low"),
        ("We'd like to partner with you as a wholesale supplier of phone accessories. Who should we contact?", "low"),
        ("What are your customer service hours on weekends?", "low"),
        ("Is your packaging recyclable?", "low"),
    ]
    for text, urg in other:
        add(text, "other", urg, [])

    return tickets


def main() -> None:
    store = make_store()
    (ROOT / "triage" / "data" / "store.json").write_text(json.dumps(store, indent=2), encoding="utf-8")
    tickets = make_tickets(store)
    with (ROOT / "eval" / "tickets.jsonl").open("w", encoding="utf-8") as f:
        for t in tickets:
            f.write(json.dumps(t) + "\n")
    counts: dict[str, int] = {}
    for t in tickets:
        counts[t["category"]] = counts.get(t["category"], 0) + 1
    print(f"{len(tickets)} tickets", counts, "approval-needed:", sum(t["needs_approval"] for t in tickets))


if __name__ == "__main__":
    main()
