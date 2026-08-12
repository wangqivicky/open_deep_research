"""Tests for safe tool-based entry routing."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

import open_deep_research.deep_researcher as deep_researcher


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.bound_tools = []
        self.invocations = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def with_retry(self, **kwargs):
        return self

    def with_config(self, config):
        return self

    async def ainvoke(self, messages):
        self.invocations.append(list(messages))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def _tool_response(name, args=None, call_id="route-1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args or {}, "id": call_id}],
    )


def _run_router(monkeypatch, response, configurable=None):
    model = FakeModel(response)
    monkeypatch.setattr(deep_researcher, "configurable_model", model)
    result = asyncio.run(
        deep_researcher.route_user_request(
            {"messages": [HumanMessage(content="user request")]},
            {"configurable": configurable or {}},
        )
    )
    return result, model


def test_router_can_ask_one_clarifying_question(monkeypatch):
    result, _ = _run_router(
        monkeypatch,
        _tool_response("AskClarification", {"question": "请说明目标公司。"}),
    )

    assert result.goto == "clarify_with_user"
    assert result.update["clarification_question"] == "请说明目标公司。"


def test_clarification_node_displays_router_question_without_model(monkeypatch):
    model = FakeModel(RuntimeError("must not be called"))
    monkeypatch.setattr(deep_researcher, "configurable_model", model)

    result = asyncio.run(
        deep_researcher.clarify_with_user(
            {"clarification_question": "请说明目标公司。"},
            {"configurable": {}},
        )
    )

    assert result.goto == "__end__"
    assert result.update["messages"][0].content == "请说明目标公司。"
    assert model.invocations == []


def test_router_sends_stable_question_to_simple_answer(monkeypatch):
    result, _ = _run_router(monkeypatch, _tool_response("AnswerSimply"))

    assert result.goto == "simple_answer"


def test_router_sends_research_request_to_existing_deep_path(monkeypatch):
    result, _ = _run_router(monkeypatch, _tool_response("StartDeepResearch"))

    assert result.goto == "write_research_brief"


def test_router_defaults_to_deep_research_without_one_clear_tool(monkeypatch):
    no_call_result, _ = _run_router(monkeypatch, AIMessage(content="plain text"))
    multiple_calls = AIMessage(
        content="",
        tool_calls=[
            {"name": "AnswerSimply", "args": {}, "id": "route-1"},
            {"name": "StartDeepResearch", "args": {}, "id": "route-2"},
        ],
    )
    multiple_result, _ = _run_router(monkeypatch, multiple_calls)

    assert no_call_result.goto == "write_research_brief"
    assert multiple_result.goto == "write_research_brief"


def test_router_failure_falls_back_to_existing_deep_path(monkeypatch):
    result, _ = _run_router(monkeypatch, RuntimeError("gateway failure"))

    assert result.goto == "write_research_brief"


def test_disabling_clarification_removes_that_routing_tool(monkeypatch):
    result, model = _run_router(
        monkeypatch,
        _tool_response("StartDeepResearch"),
        {"allow_clarification": False},
    )

    assert result.goto == "write_research_brief"
    assert {tool.__name__ for tool in model.bound_tools} == {
        "AnswerSimply",
        "StartDeepResearch",
    }


def test_simple_answer_returns_plain_model_message(monkeypatch):
    response = AIMessage(content="RAG combines retrieval with generation.")
    model = FakeModel(response)
    monkeypatch.setattr(deep_researcher, "configurable_model", model)

    result = asyncio.run(
        deep_researcher.simple_answer(
            {"messages": [HumanMessage(content="What is RAG?")]},
            {"configurable": {}},
        )
    )

    assert result.goto == "__end__"
    assert result.update["messages"] == [response]


def test_simple_answer_failure_falls_back_to_deep_research(monkeypatch):
    model = FakeModel(RuntimeError("gateway failure"))
    monkeypatch.setattr(deep_researcher, "configurable_model", model)

    result = asyncio.run(
        deep_researcher.simple_answer(
            {"messages": [HumanMessage(content="What is RAG?")]},
            {"configurable": {}},
        )
    )

    assert result.goto == "write_research_brief"
