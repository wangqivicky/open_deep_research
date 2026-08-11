import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from open_deep_research.context_manager import estimate_context_budget


@tool
def sample_tool(query: str) -> str:
    """Search for a short query."""
    return query


def test_estimate_context_budget_counts_messages_tools_and_output_reserve():
    result = estimate_context_budget(
        [SystemMessage(content="instructions"), HumanMessage(content="x" * 200)],
        tools=[sample_tool],
        context_window=200,
        reserved_output_tokens=20,
        safety_margin_ratio=0.05,
        warning_ratio=0.30,
        compaction_ratio=0.60,
        hard_limit_ratio=0.90,
        chars_per_token=2.0,
    )

    assert result.message_tokens > 0
    assert result.tool_schema_tokens > 0
    assert result.estimated_input_tokens == (
        result.message_tokens + result.tool_schema_tokens
    )
    assert result.safety_margin_tokens == 10
    assert result.action in {"warn", "compact", "hard_limit"}
    assert result.largest_messages[0]["index"] == 1


def test_context_budget_metadata_is_flat_and_trace_friendly():
    result = estimate_context_budget(
        [HumanMessage(content="hello")],
        context_window=1_000,
        reserved_output_tokens=100,
    )

    metadata = result.as_metadata("researcher")

    assert metadata["context_budget_call"] == "researcher"
    assert metadata["context_budget_action"] == "proceed"
    assert (
        metadata["context_budget_estimated_input_tokens"]
        == result.estimated_input_tokens
    )
    assert isinstance(metadata["context_budget_largest_messages"], list)


def test_context_budget_rejects_invalid_threshold_order():
    with pytest.raises(ValueError, match="context ratios"):
        estimate_context_budget(
            [HumanMessage(content="hello")],
            context_window=1_000,
            reserved_output_tokens=100,
            warning_ratio=0.9,
            compaction_ratio=0.8,
            hard_limit_ratio=0.7,
        )
