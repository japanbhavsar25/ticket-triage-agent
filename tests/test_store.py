from triage.store import Store


def test_get_order_found_and_missing(store_data):
    s = Store(store_data)
    assert s.get_order("ord-1")["refundable_amount"] == 20.0
    assert s.get_order("ORD-404") == {"found": False, "order_id": "ORD-404"}


def test_get_customer_lists_orders(store_data):
    c = Store(store_data).get_customer("ANN@example.com ")
    assert c["found"] and c["order_ids"] == ["ORD-1", "ORD-2", "ORD-3"]


def test_refund_rules(store_data):
    s = Store(store_data)
    assert s.issue_refund("ORD-1", 20)["success"]
    assert not s.issue_refund("ORD-1", 1)["success"]  # nothing left to refund
    assert not s.issue_refund("ORD-3", 5)["success"]  # not delivered
    assert not s.issue_refund("ORD-2", 500)["success"]  # more than the total


def test_stores_do_not_share_state(store_data):
    a, b = Store(store_data), Store(store_data)
    a.issue_refund("ORD-1", 20)
    assert b.get_order("ORD-1")["refundable_amount"] == 20.0
