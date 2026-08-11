"""Context budget estimation helpers for model calls."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from langchain_core.messages import MessageLikeRepresentation
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool


ContextBudgetAction = Literal["proceed", "warn", "compact", "hard_limit"]


@dataclass(frozen=True)
class ContextBudgetResult:
    """Estimated context usage for one model invocation."""

    estimated_input_tokens: int
    message_tokens: int
    tool_schema_tokens: int
    reserved_output_tokens: int
    safety_margin_tokens: int
    context_window: int
    utilization: float
    action: ContextBudgetAction
    largest_messages: list[dict[str, Any]]

    def as_metadata(self, call_name: str) -> dict[str, Any]:
        """Return JSON-serializable metadata for tracing."""
        return {
            "context_budget_call": call_name,
            "context_budget_estimated_input_tokens": self.estimated_input_tokens,
            "context_budget_message_tokens": self.message_tokens,
            "context_budget_tool_schema_tokens": self.tool_schema_tokens,
            "context_budget_reserved_output_tokens": self.reserved_output_tokens,
            "context_budget_safety_margin_tokens": self.safety_margin_tokens,
            "context_budget_context_window": self.context_window,
            "context_budget_utilization": round(self.utilization, 6),
            "context_budget_action": self.action,
            "context_budget_largest_messages": self.largest_messages,
        }


def estimate_context_budget(
    messages: Sequence[MessageLikeRepresentation],
    *,
    context_window: int,
    reserved_output_tokens: int,
    warning_ratio: float = 0.70,
    compaction_ratio: float = 0.80,
    hard_limit_ratio: float = 0.90,
    safety_margin_ratio: float = 0.05,
    chars_per_token: float = 2.0,
    tools: list[BaseTool | dict[str, Any]] | None = None,
    largest_message_count: int = 3,
) -> ContextBudgetResult:
    """Estimate context use without changing or rejecting the model input."""
    if context_window <= 0:
        raise ValueError("context_window must be positive")
    if reserved_output_tokens < 0:
        raise ValueError("reserved_output_tokens cannot be negative")
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    if not 0 <= warning_ratio <= compaction_ratio <= hard_limit_ratio <= 1:
        raise ValueError(
            "context ratios must satisfy 0 <= warning <= compaction <= hard <= 1"
        )
    if not 0 <= safety_margin_ratio < 1:
        raise ValueError("safety_margin_ratio must satisfy 0 <= ratio < 1")

    message_tokens = count_tokens_approximately(
        messages,
        chars_per_token=chars_per_token,
    )
    tool_schema_tokens = (
        count_tokens_approximately(
            [],
            chars_per_token=chars_per_token,
            tools=tools,
        )
        if tools
        else 0
    )
    estimated_input_tokens = message_tokens + tool_schema_tokens
    safety_margin_tokens = math.ceil(context_window * safety_margin_ratio)
    utilization = (
        estimated_input_tokens
        + reserved_output_tokens
        + safety_margin_tokens
    ) / context_window

    if utilization >= hard_limit_ratio:
        action: ContextBudgetAction = "hard_limit"
    elif utilization >= compaction_ratio:
        action = "compact"
    elif utilization >= warning_ratio:
        action = "warn"
    else:
        action = "proceed"

    per_message_tokens = []
    for index, message in enumerate(messages):
        tokens = count_tokens_approximately(
            [message],
            chars_per_token=chars_per_token,
        )
        message_type = getattr(message, "type", type(message).__name__)
        message_name = getattr(message, "name", None)
        per_message_tokens.append(
            {
                "index": index,
                "type": str(message_type),
                "name": message_name,
                "estimated_tokens": tokens,
            }
        )

    largest_messages = sorted(
        per_message_tokens,
        key=lambda item: item["estimated_tokens"],
        reverse=True,
    )[:largest_message_count]

    return ContextBudgetResult(
        estimated_input_tokens=estimated_input_tokens,
        message_tokens=message_tokens,
        tool_schema_tokens=tool_schema_tokens,
        reserved_output_tokens=reserved_output_tokens,
        safety_margin_tokens=safety_margin_tokens,
        context_window=context_window,
        utilization=utilization,
        action=action,
        largest_messages=largest_messages,
    )


def log_context_budget(call_name: str, budget: ContextBudgetResult) -> None:
    """Log the prediction while leaving model behavior unchanged."""
    log = logging.warning if budget.action != "proceed" else logging.info
    log(
        "Context budget for %s: action=%s utilization=%.3f input=%s "
        "reserved_output=%s safety_margin=%s window=%s",
        call_name,
        budget.action,
        budget.utilization,
        budget.estimated_input_tokens,
        budget.reserved_output_tokens,
        budget.safety_margin_tokens,
        budget.context_window,
    )
