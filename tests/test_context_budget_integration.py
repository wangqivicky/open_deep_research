import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import open_deep_research.deep_researcher as deep_researcher


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.configs = []
        self.invoke_count = 0
        self.invocations = []

    def bind_tools(self, tools):
        return self

    def with_retry(self, **kwargs):
        return self

    def with_config(self, config):
        self.configs.append(config)
        return self

    async def ainvoke(self, messages):
        self.invoke_count += 1
        self.invocations.append(list(messages))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def _budget_calls(model):
    return [
        config["metadata"]["context_budget_call"]
        for config in model.configs
        if "metadata" in config and "context_budget_call" in config["metadata"]
    ]


def test_supervisor_attaches_context_budget_metadata(monkeypatch):
    model = FakeModel(AIMessage(content="continue"))
    monkeypatch.setattr(deep_researcher, "configurable_model", model)

    asyncio.run(
        deep_researcher.supervisor(
            {"supervisor_messages": [HumanMessage(content="research the topic")]},
            {"configurable": {}},
        )
    )

    assert _budget_calls(model) == ["supervisor"]


def test_supervisor_compacts_active_view_but_retains_full_state(monkeypatch):
    model = FakeModel(AIMessage(content="continue"))
    monkeypatch.setattr(deep_researcher, "configurable_model", model)
    archived = []

    async def fake_summarize(messages, existing_summary, configurable, config):
        archived.extend(messages)
        return "Earlier research established finding A from https://example.com/a."

    monkeypatch.setattr(
        deep_researcher,
        "_summarize_supervisor_history",
        fake_summarize,
    )
    messages = [
        HumanMessage(content="system instructions placeholder"),
        HumanMessage(content="research brief"),
        AIMessage(content="delegate first research round"),
        ToolMessage(
            content="old completed finding that should leave the active window",
            name="ConductResearch",
            tool_call_id="research-1",
        ),
        AIMessage(content="plan the latest research round"),
        ToolMessage(
            content="latest completed finding that should remain active",
            name="ConductResearch",
            tool_call_id="research-2",
        ),
    ]

    result = asyncio.run(
        deep_researcher.supervisor(
            {"supervisor_messages": messages},
            {
                "configurable": {
                    "supervisor_model_context_window": 12000,
                }
            },
        )
    )

    assert len(messages) == 6
    assert [message.content for message in archived] == [
        "delegate first research round",
        "old completed finding that should leave the active window",
    ]
    active_contents = [message.content for message in model.invocations[-1]]
    assert any("Earlier research established" in content for content in active_contents)
    assert "old completed finding that should leave the active window" not in active_contents
    assert "latest completed finding that should remain active" in active_contents
    assert result.update["supervisor_compacted_message_count"] == 4
    assert result.update["supervisor_progress_summary"].startswith("Earlier research")
    supervisor_metadata = next(
        config["metadata"]
        for config in reversed(model.configs)
        if config.get("metadata", {}).get("context_budget_call") == "supervisor"
    )
    assert supervisor_metadata["context_budget_compaction_triggered"] is True
    assert supervisor_metadata["context_budget_full_message_count"] == 6
    assert supervisor_metadata["context_budget_active_message_count"] == 5


def test_researcher_attaches_context_budget_metadata(monkeypatch):
    model = FakeModel(AIMessage(content="done"))
    monkeypatch.setattr(deep_researcher, "configurable_model", model)

    async def fake_get_all_tools(config):
        return [deep_researcher.think_tool]

    monkeypatch.setattr(deep_researcher, "get_all_tools", fake_get_all_tools)

    asyncio.run(
        deep_researcher.researcher(
            {"researcher_messages": [HumanMessage(content="find evidence")]},
            {"configurable": {}},
        )
    )

    assert _budget_calls(model) == ["researcher"]


def test_researcher_failure_with_existing_search_results_falls_back_to_compression(
    monkeypatch,
):
    model = FakeModel(RuntimeError("provider overloaded"))
    monkeypatch.setattr(deep_researcher, "configurable_model", model)

    async def fake_get_all_tools(config):
        return [deep_researcher.think_tool]

    monkeypatch.setattr(deep_researcher, "get_all_tools", fake_get_all_tools)
    messages = [
        HumanMessage(content="research topic"),
        ToolMessage(
            content="Verified source summary",
            name="tavily_search",
            tool_call_id="search-1",
        ),
    ]

    result = asyncio.run(
        deep_researcher.researcher(
            {"researcher_messages": messages},
            {"configurable": {}},
        )
    )

    assert result.goto == "compress_research"
    assert len(messages) == 2


def test_researcher_failure_without_search_results_remains_visible(monkeypatch):
    model = FakeModel(RuntimeError("provider overloaded"))
    monkeypatch.setattr(deep_researcher, "configurable_model", model)

    async def fake_get_all_tools(config):
        return [deep_researcher.think_tool]

    monkeypatch.setattr(deep_researcher, "get_all_tools", fake_get_all_tools)

    try:
        asyncio.run(
            deep_researcher.researcher(
                {"researcher_messages": [HumanMessage(content="research topic")]},
                {"configurable": {}},
            )
        )
    except RuntimeError as error:
        assert str(error) == "provider overloaded"
    else:
        raise AssertionError("Researcher failure should not be silently swallowed")


def test_compression_normalizes_output_and_does_not_mutate_state(monkeypatch):
    response = AIMessage(
        content=[
            {"type": "reasoning", "encrypted_content": "must-not-leak"},
            {"type": "text", "text": "# Clean research summary"},
        ]
    )
    model = FakeModel(response)
    monkeypatch.setattr(deep_researcher, "configurable_model", model)
    messages = [
        HumanMessage(content="topic"),
        AIMessage(content="analysis"),
        ToolMessage(content="source summary", tool_call_id="tool-1"),
    ]

    result = asyncio.run(
        deep_researcher.compress_research(
            {"researcher_messages": messages},
            {"configurable": {}},
        )
    )

    assert len(messages) == 3
    assert result["compressed_research"] == "# Clean research summary"
    assert "must-not-leak" not in result["compressed_research"]
    assert "must-not-leak" not in "\n".join(result["raw_notes"])
    assert "source summary" in "\n".join(result["raw_notes"])
    assert _budget_calls(model) == ["compress_research"]


def test_final_report_attaches_context_budget_metadata(monkeypatch):
    model = FakeModel(AIMessage(content="# Final report"))
    monkeypatch.setattr(deep_researcher, "configurable_model", model)

    result = asyncio.run(
        deep_researcher.final_report_generation(
            {
                "research_brief": "Explain the topic",
                "notes": ["source one", "source two"],
                "messages": [HumanMessage(content="original request")],
            },
            {"configurable": {}},
        )
    )

    assert result["final_report"] == "# Final report"
    assert _budget_calls(model) == ["final_report_generation"]


def test_final_report_refuses_to_run_without_usable_research(monkeypatch):
    model = FakeModel(AIMessage(content="# Unsupported report"))
    monkeypatch.setattr(deep_researcher, "configurable_model", model)

    result = asyncio.run(
        deep_researcher.final_report_generation(
            {
                "research_brief": "Explain the topic",
                "notes": ["Error synthesizing research report: retries exhausted"],
                "messages": [HumanMessage(content="original request")],
            },
            {"configurable": {}},
        )
    )

    assert result["final_report"].startswith(
        "Error generating final report: No usable research findings"
    )
    assert model.invoke_count == 0


def test_completed_research_notes_exclude_think_tool_and_errors():
    messages = [
        ToolMessage(
            content="Reflection recorded: plan next step",
            name="think_tool",
            tool_call_id="think-1",
        ),
        ToolMessage(
            content="Verified compressed research",
            name="ConductResearch",
            tool_call_id="research-1",
        ),
        ToolMessage(
            content="Error synthesizing research report: retries exhausted",
            name="ConductResearch",
            tool_call_id="research-2",
        ),
    ]

    assert deep_researcher._get_completed_research_notes(messages) == [
        "Verified compressed research"
    ]
