from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import ChatGeneration

from open_deep_research.deep_researcher import _prepare_openai_compatible_messages
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    summarize_webpage_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import ClarifyWithUser


def test_reasoning_and_text_blocks_parse_as_pydantic_json():
    message = AIMessage(
        content=[
            {"type": "reasoning", "summary": []},
            {
                "type": "text",
                "text": (
                    '{"need_clarification":false,"question":"",'
                    '"verification":"start research"}'
                ),
            },
        ]
    )

    parsed = PydanticOutputParser(
        pydantic_object=ClarifyWithUser
    ).parse_result([ChatGeneration(message=message)])

    assert parsed.need_clarification is False
    assert parsed.question == ""
    assert parsed.verification == "start research"


def test_json_mode_prompts_contain_lowercase_json_keyword():
    prompts = [
        clarify_with_user_instructions,
        transform_messages_into_research_topic_prompt,
        summarize_webpage_prompt,
    ]

    assert all("json" in prompt for prompt in prompts)


def test_custom_responses_gateway_maps_system_to_developer(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://compatible.example/v1")
    monkeypatch.setenv("OPENAI_USE_RESPONSES_API", "true")
    original = SystemMessage(content="supervisor instructions")
    human = HumanMessage(content="research brief")

    prepared = _prepare_openai_compatible_messages(
        [original, human],
        "openai:test-model",
    )

    assert prepared[0].additional_kwargs["__openai_role__"] == "developer"
    assert "__openai_role__" not in original.additional_kwargs
    assert prepared[1] is human
