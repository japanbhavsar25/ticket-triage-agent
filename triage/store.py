"""A tiny fake customer/order database backed by a JSON file.

Each Store instance works on its own in-memory copy, so refunds issued during
one run (or one eval ticket) never leak into another.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "store.json"


def _load(path: Path = DATA_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Store:
    def __init__(self, data: dict | None = None):
        self.data = copy.deepcopy(data if data is not None else _load())
        self._orders = {o["order_id"].upper(): o for o in self.data["orders"]}
        self._customers = {c["email"].lower(): c for c in self.data["customers"]}
        self.refunds: list[dict] = []

    def get_order(self, order_id: str) -> dict:
        order = self._orders.get(str(order_id).strip().upper())
        if order is None:
            return {"found": False, "order_id": order_id}
        refundable = round(order["total"] - order["refunded_amount"], 2)
        return {"found": True, **order, "refundable_amount": refundable}

    def get_customer(self, email: str) -> dict:
        customer = self._customers.get(str(email).strip().lower())
        if customer is None:
            return {"found": False, "email": email}
        orders = [o["order_id"] for o in self.data["orders"] if o["email"] == customer["email"]]
        return {"found": True, **customer, "order_ids": orders}

    def issue_refund(self, order_id: str, amount: float) -> dict:
        order = self._orders.get(str(order_id).strip().upper())
        if order is None:
            return {"success": False, "error": f"order {order_id} not found"}
        amount = round(float(amount), 2)
        refundable = round(order["total"] - order["refunded_amount"], 2)
        if order["status"] != "delivered":
            return {"success": False, "error": f"order status is {order['status']}, not delivered"}
        if amount <= 0 or amount > refundable:
            return {"success": False, "error": f"amount {amount} exceeds refundable {refundable}"}
        order["refunded_amount"] = round(order["refunded_amount"] + amount, 2)
        refund = {
            "success": True,
            "refund_id": f"RF-{len(self.refunds) + 1:04d}",
            "order_id": order["order_id"],
            "amount": amount,
        }
        self.refunds.append(refund)
        return refund
