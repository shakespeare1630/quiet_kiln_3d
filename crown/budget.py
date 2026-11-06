"""Exact competition accounting and conservative pre-call reservations."""

import json
from decimal import Decimal
from pathlib import Path

PRICING = json.loads(Path(__file__).with_name("pricing.json").read_text(), parse_float=Decimal)


class BudgetStop(Exception):
    """No room to start another call; keep a previously validated candidate."""


class EvidenceUnavailable(RuntimeError):
    """Usage cannot be established; let the live runner recover the call."""


class BudgetExceeded(RuntimeError):
    """Actual usage exceeded the committed limit; do not submit the result."""


def credits(model: str, usage: dict) -> Decimal:
    for kind in ("input_tokens", "output_tokens"):
        if type(usage.get(kind)) is not int or usage[kind] < 0:
            raise EvidenceUnavailable(f"Missing or invalid {kind}")
    if "total_tokens" in usage and usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
        raise EvidenceUnavailable("Inconsistent total token usage")
    rates = PRICING["models"][model]
    prefix = "long_" if usage["input_tokens"] > PRICING["long_context_above_input_tokens"] else ""
    return sum(
        Decimal(str(rates[f"{prefix}{kind}_credits_per_token"])) * usage[f"{kind}_tokens"]
        for kind in ("input", "output")
    )


def reservation(
    model: str,
    instructions: str,
    content: list[dict],
    output_tokens: int,
    *,
    image_tokens: int = 16_384,
    framing_tokens: int = 2_048,
) -> Decimal:
    # All image inputs are resized to <=1024px by media.py. Reserve 16K tokens
    # per image by default and one token per UTF-8 text byte plus framing headroom.
    # Experimental estimates can be calibrated separately; actual spend still
    # settles every response and overrun always aborts.
    # This is deliberately conservative; actual provider usage settles the call.
    input_tokens = len(instructions.encode()) + framing_tokens
    for item in content:
        if item["type"] == "input_text":
            input_tokens += len(item["text"].encode()) + 64
        elif item["type"] == "input_image":
            input_tokens += image_tokens
        else:
            raise ValueError("Unaccounted input type")
    return credits(model, {"input_tokens": input_tokens, "output_tokens": output_tokens})


class Budget:
    def __init__(self, max_credits: int, max_calls: int):
        self.max_credits = Decimal(max_credits)
        self.max_calls = max_calls
        self.charges: dict[int, Decimal] = {}

    @property
    def spent(self) -> Decimal:
        return sum(self.charges.values(), Decimal(0))

    def check(self, expected: Decimal, *, calls: int = 1) -> None:
        if len(self.charges) + calls > self.max_calls or self.spent + expected > self.max_credits:
            raise BudgetStop

    def settle(self, request_id: int, model: str, usage: dict) -> None:
        amount = credits(model, usage)
        if request_id in self.charges:
            if self.charges[request_id] != amount:
                raise EvidenceUnavailable("A request ID was reused for a different charge")
            return
        self.charges[request_id] = amount
        if self.spent > self.max_credits or len(self.charges) > self.max_calls:
            raise BudgetExceeded("Actual captured usage exceeded this miner's budget")
