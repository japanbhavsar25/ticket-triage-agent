from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["billing", "refund", "shipping", "technical", "account", "other"]
Urgency = Literal["low", "medium", "high"]


class Triage(BaseModel):
    """Classification and entities extracted from one support ticket."""

    category: Category
    urgency: Urgency
    order_id: str | None = Field(None, description="Order id exactly as written, e.g. ORD-1001")
    email: str | None = Field(None, description="Customer email address if present")
    product: str | None = Field(None, description="Product name if mentioned")
    refund_amount_requested: float | None = Field(
        None, description="Dollar amount the customer explicitly asks to have refunded, if stated"
    )
    summary: str = Field(description="One-sentence summary of the customer's problem")


class Draft(BaseModel):
    """The reply to send to the customer, plus the facts it relies on."""

    reply: str = Field(description="Reply to the customer. Cite facts inline as [1], [2], ...")
    cited_facts: list[str] = Field(
        description="Numbered facts used in the reply, each copied from a tool result, "
        "e.g. 'get_order(ORD-1001): status = delivered'"
    )
