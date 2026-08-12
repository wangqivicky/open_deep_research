"""Tests for resilient, non-JSON webpage summary formatting."""

import asyncio

from langchain_core.messages import AIMessage

from open_deep_research.utils import _format_webpage_summary, summarize_webpage


class FakeSummaryModel:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    async def ainvoke(self, messages):
        if self.error:
            raise self.error
        return self.response


def test_formats_tagged_response_and_ignores_reasoning_blocks():
    response = AIMessage(
        content=[
            {"type": "reasoning", "summary": ["internal reasoning"]},
            {
                "type": "text",
                "text": (
                    "<summary>Verified company facts.</summary>\n"
                    "<key_excerpts>\n- Founded in 2013\n- Capital: 30m\n"
                    "</key_excerpts>"
                ),
            },
        ]
    )

    formatted = _format_webpage_summary(response)

    assert "Verified company facts." in formatted
    assert "Founded in 2013" in formatted
    assert "internal reasoning" not in formatted
    assert formatted.count("<summary>") == 1
    assert formatted.count("<key_excerpts>") == 1


def test_malformed_legacy_json_becomes_safe_summary_instead_of_raising():
    malformed = AIMessage(
        content=(
            '{"summary":"Company facts","key_excerpts":"first",'
            '"second","third"}'
        )
    )

    formatted = _format_webpage_summary(malformed)

    assert malformed.content in formatted
    assert "<key_excerpts>\n\n</key_excerpts>" in formatted


def test_missing_closing_tag_is_recovered_locally():
    response = AIMessage(
        content=(
            "<summary>Useful summary without a closing tag\n"
            "<key_excerpts>- supporting excerpt</key_excerpts>"
        )
    )

    formatted = _format_webpage_summary(response)

    assert "Useful summary without a closing tag" in formatted
    assert "supporting excerpt" in formatted


def test_model_failure_keeps_existing_raw_content_fallback():
    raw_content = "original webpage content"
    model = FakeSummaryModel(error=RuntimeError("gateway unavailable"))

    result = asyncio.run(summarize_webpage(model, raw_content))

    assert result == raw_content
