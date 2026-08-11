"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
import logging
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.context_manager import (
    estimate_context_budget,
    log_context_budget,
)
from open_deep_research.email_report import normalize_report_text, send_report_email_async
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_base_url_for_model,
    get_use_responses_api_for_model,
    get_model_token_limit,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=(
        "model",
        "max_tokens",
        "api_key",
        "base_url",
        "use_responses_api",
    ),
)


def _predict_context_budget(
    call_name: str,
    messages,
    configurable: Configuration,
    *,
    context_window: int,
    reserved_output_tokens: int,
    tools=None,
):
    """Estimate and log one model call without changing its input."""
    budget = estimate_context_budget(
        messages,
        context_window=context_window,
        reserved_output_tokens=reserved_output_tokens,
        warning_ratio=configurable.context_warning_ratio,
        compaction_ratio=configurable.context_compaction_ratio,
        hard_limit_ratio=configurable.context_hard_limit_ratio,
        safety_margin_ratio=configurable.context_safety_margin_ratio,
        chars_per_token=configurable.context_estimation_chars_per_token,
        tools=tools,
    )
    log_context_budget(call_name, budget)
    return budget


def _prepare_openai_compatible_messages(messages, model_name: str):
    """Map system instructions to the developer role for custom Responses APIs."""
    if not (
        get_base_url_for_model(model_name)
        and get_use_responses_api_for_model(model_name)
    ):
        return list(messages)

    prepared_messages = []
    for message in messages:
        if isinstance(message, SystemMessage):
            prepared_messages.append(
                message.model_copy(
                    update={
                        "additional_kwargs": {
                            **message.additional_kwargs,
                            "__openai_role__": "developer",
                        }
                    }
                )
            )
        else:
            prepared_messages.append(message)
    return prepared_messages


def _collect_readable_notes(messages) -> str:
    """Collect readable tool and assistant text without provider metadata."""
    readable_parts = (
        normalize_report_text(message)
        for message in filter_messages(messages, include_types=["tool", "ai"])
    )
    return "\n".join(part for part in readable_parts if part)


def _is_usable_research_text(text: str) -> bool:
    """Reject empty or explicit error payloads as research evidence."""
    normalized = text.strip()
    return bool(normalized) and not normalized.startswith(
        (
            "Error executing tool:",
            "Error synthesizing research report:",
            "Error: Did not run this research",
        )
    )


def _has_research_evidence(messages) -> bool:
    """Return whether a researcher state contains usable external-tool output."""
    for message in filter_messages(messages, include_types="tool"):
        if getattr(message, "name", None) in {"think_tool", "ResearchComplete"}:
            continue
        if _is_usable_research_text(normalize_report_text(message)):
            return True
    return False


def _get_completed_research_notes(messages) -> list[str]:
    """Extract successful ConductResearch summaries, excluding reflections."""
    notes = []
    for message in filter_messages(messages, include_types="tool"):
        if getattr(message, "name", None) != "ConductResearch":
            continue
        text = normalize_report_text(message)
        if _is_usable_research_text(text):
            notes.append(text)
    return notes


def _build_supervisor_active_context(
    messages,
    progress_summary: str,
    compacted_message_count: int,
):
    """Build the Supervisor model view while retaining full history in state."""
    base_count = min(2, len(messages))
    compacted_message_count = max(base_count, compacted_message_count)
    active_messages = list(messages[:base_count])
    if progress_summary:
        active_messages.append(
            SystemMessage(
                content=(
                    "Research progress summary from earlier completed rounds:\n\n"
                    f"{progress_summary}"
                )
            )
        )
    active_messages.extend(messages[compacted_message_count:])
    return active_messages


def _find_supervisor_compaction_cutoff(messages, compacted_message_count: int):
    """Return a safe boundary that keeps the most recent AI/tool round intact."""
    start = max(min(2, len(messages)), compacted_message_count)
    ai_indices = [
        index
        for index in range(start, len(messages))
        if isinstance(messages[index], AIMessage)
    ]
    if len(ai_indices) < 2:
        return None
    cutoff = ai_indices[-1]
    return cutoff if cutoff > start else None


async def _summarize_supervisor_history(
    messages,
    existing_summary: str,
    configurable: Configuration,
    config: RunnableConfig,
) -> str:
    """Merge older completed Supervisor rounds into a durable progress summary."""
    prompt = f"""Create an updated research progress summary for a research supervisor.

Preserve concrete findings, source URLs, completed research topics, important strategic
decisions and reflections, unresolved gaps, conflicts, and recommended next actions.
Do not invent facts. Keep enough detail that the supervisor can continue without the
original messages.

EXISTING PROGRESS SUMMARY:
{existing_summary or "None"}

NEWLY ARCHIVED MESSAGES:
{get_buffer_string(messages)}
"""
    summary_messages = [HumanMessage(content=prompt)]
    budget = _predict_context_budget(
        "supervisor_context_compaction",
        summary_messages,
        configurable,
        context_window=configurable.compression_model_context_window,
        reserved_output_tokens=configurable.compression_model_max_tokens,
    )
    summary_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "base_url": get_base_url_for_model(configurable.compression_model),
        "use_responses_api": get_use_responses_api_for_model(configurable.compression_model),
        "tags": ["langsmith:nostream"],
        "metadata": budget.as_metadata("supervisor_context_compaction"),
    })
    response = await summary_model.ainvoke(summary_messages)
    summary = normalize_report_text(response)
    if not summary:
        raise ValueError("Supervisor context compaction returned no readable text")
    return summary


async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_brief")
    
    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "base_url": get_base_url_for_model(configurable.research_model),
        "use_responses_api": get_use_responses_api_for_model(configurable.research_model),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model with structured output and retry logic
    clarification_model = (
        configurable_model.with_config(model_config)
        | PydanticOutputParser(pydantic_object=ClarifyWithUser)
    ).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Proceed to research with verification message
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to research supervisor with initialized context
    """
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "base_url": get_base_url_for_model(configurable.research_model),
        "use_responses_api": get_use_responses_api_for_model(configurable.research_model),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model for structured research question generation
    research_model = (
        configurable_model.with_config(research_model_config)
        | PydanticOutputParser(pydantic_object=ResearchQuestion)
    ).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    
    # Step 2: Generate structured research brief from user messages
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 3: Initialize supervisor with research brief and instructions
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "base_url": get_base_url_for_model(configurable.research_model),
        "use_responses_api": get_use_responses_api_for_model(configurable.research_model),
        "tags": ["langsmith:nostream"]
    }
    
    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # Configure model with tools, retry logic, and model settings
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Build the Supervisor's active model view. Full history remains in
    # graph state; older completed rounds may be represented by one summary.
    supervisor_messages = list(state.get("supervisor_messages", []))
    base_message_count = min(2, len(supervisor_messages))
    progress_summary = state.get("supervisor_progress_summary", "")
    compacted_message_count = max(
        base_message_count,
        state.get("supervisor_compacted_message_count", base_message_count),
    )
    active_messages = _build_supervisor_active_context(
        supervisor_messages,
        progress_summary,
        compacted_message_count,
    )
    model_messages = _prepare_openai_compatible_messages(
        active_messages,
        configurable.research_model,
    )
    initial_budget = _predict_context_budget(
        "supervisor",
        model_messages,
        configurable,
        context_window=configurable.supervisor_model_context_window,
        reserved_output_tokens=configurable.research_model_max_tokens,
        tools=lead_researcher_tools,
    )

    compaction_triggered = False
    if initial_budget.action in {"compact", "hard_limit"}:
        cutoff = _find_supervisor_compaction_cutoff(
            supervisor_messages,
            compacted_message_count,
        )
        if cutoff is not None:
            progress_summary = await _summarize_supervisor_history(
                supervisor_messages[compacted_message_count:cutoff],
                progress_summary,
                configurable,
                config,
            )
            compacted_message_count = cutoff
            active_messages = _build_supervisor_active_context(
                supervisor_messages,
                progress_summary,
                compacted_message_count,
            )
            model_messages = _prepare_openai_compatible_messages(
                active_messages,
                configurable.research_model,
            )
            compaction_triggered = True

    context_budget = _predict_context_budget(
        "supervisor",
        model_messages,
        configurable,
        context_window=configurable.supervisor_model_context_window,
        reserved_output_tokens=configurable.research_model_max_tokens,
        tools=lead_researcher_tools,
    )
    budget_metadata = context_budget.as_metadata("supervisor")
    budget_metadata.update({
        "context_budget_compaction_triggered": compaction_triggered,
        "context_budget_pre_compaction_action": initial_budget.action,
        "context_budget_pre_compaction_utilization": round(
            initial_budget.utilization, 6
        ),
        "context_budget_full_message_count": len(supervisor_messages),
        "context_budget_active_message_count": len(model_messages),
        "context_budget_compacted_message_count": compacted_message_count,
    })
    response = await research_model.with_config(
        {"metadata": budget_metadata}
    ).ainvoke(model_messages)
    
    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
            "supervisor_progress_summary": progress_summary,
            "supervisor_compacted_message_count": compacted_message_count,
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # Define exit criteria for research phase
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": _get_completed_research_notes(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    
    if conduct_research_calls:
        try:
            # Limit concurrent research units to prevent resource exhaustion
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]
            
            # Execute research tasks in parallel
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config) 
                for tool_call in allowed_conduct_research_calls
            ]
            
            tool_results = await asyncio.gather(*research_tasks)
            
            # Create tool messages with research results
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
            
            # Handle overflow research calls with error messages
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
            
            # Aggregate raw notes from all research results
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
            ])
            
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
                
        except Exception as e:
            # A token-limit failure may still leave earlier completed research
            # units available. Other failures must remain visible rather than
            # silently producing a report without evidence.
            if is_token_limit_exceeded(e, configurable.research_model):
                return Command(
                    goto=END,
                    update={
                        "notes": _get_completed_research_notes(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )
            raise
    
    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools", "compress_research"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = list(state.get("researcher_messages", []))
    
    # Get all available research tools (search, MCP, think_tool)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    
    # Step 2: Configure the researcher model with tools
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "base_url": get_base_url_for_model(configurable.research_model),
        "use_responses_api": get_use_responses_api_for_model(configurable.research_model),
        "tags": ["langsmith:nostream"]
    }
    
    # Prepare system prompt with MCP context if available
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    # Configure model with tools, retry logic, and settings
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 3: Generate researcher response with system context
    messages = _prepare_openai_compatible_messages(
        [SystemMessage(content=researcher_prompt)] + researcher_messages,
        configurable.research_model,
    )
    context_budget = _predict_context_budget(
        "researcher",
        messages,
        configurable,
        context_window=configurable.research_model_context_window,
        reserved_output_tokens=configurable.research_model_max_tokens,
        tools=tools,
    )
    try:
        response = await research_model.with_config(
            {"metadata": context_budget.as_metadata("researcher")}
        ).ainvoke(messages)
    except Exception:
        if _has_research_evidence(researcher_messages):
            logging.exception(
                "Researcher model failed after evidence was collected; "
                "falling back to compression"
            )
            return Command(goto="compress_research")
        raise
    
    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = list(state.get("researcher_messages", []))
    most_recent_message = researcher_messages[-1]
    
    # Early exit if no tool calls were made (including native web search)
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or 
        anthropic_websearch_called(most_recent_message)
    )
    
    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")
    
    # Step 2: Handle other tool calls (search, MCP tools, etc.)
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) 
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    
    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    
    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # End research and proceed to compression
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )
    
    # Continue research loop with tool results
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "base_url": get_base_url_for_model(configurable.compression_model),
        "use_responses_api": get_use_responses_api_for_model(configurable.compression_model),
        "tags": ["langsmith:nostream"]
    })
    
    # Step 2: Prepare messages for compression without mutating graph state.
    researcher_messages = list(state.get("researcher_messages", []))
    
    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = _prepare_openai_compatible_messages(
                [SystemMessage(content=compression_prompt)] + researcher_messages,
                configurable.compression_model,
            )
            context_budget = _predict_context_budget(
                "compress_research",
                messages,
                configurable,
                context_window=configurable.compression_model_context_window,
                reserved_output_tokens=configurable.compression_model_max_tokens,
            )
            
            # Execute compression
            response = await synthesizer_model.with_config(
                {"metadata": context_budget.as_metadata("compress_research")}
            ).ainvoke(messages)
            compressed_research = normalize_report_text(response)
            if not compressed_research:
                raise ValueError("Compression model returned no readable text content")
            
            # Extract raw notes from all tool and AI messages
            raw_notes_content = _collect_readable_notes(researcher_messages)
            
            # Return successful compression result
            return {
                "compressed_research": compressed_research,
                "raw_notes": [raw_notes_content]
            }
            
        except Exception as e:
            synthesis_attempts += 1
            logging.exception(
                "Compression attempt %s/%s failed",
                synthesis_attempts,
                max_attempts,
            )
            
            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.compression_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # For other errors, continue retrying
            continue
    
    # Step 4: Return error result if all attempts failed
    raw_notes_content = _collect_readable_notes(researcher_messages)
    
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }

# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)                 # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)     # Tool execution handler
researcher_builder.add_node("compress_research", compress_research)   # Research compression

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("compress_research", END)      # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract research findings and prepare state cleanup
    notes = [
        text
        for note in state.get("notes", [])
        if _is_usable_research_text(text := normalize_report_text(note))
    ]
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)

    if not findings:
        failure_message = (
            "Error generating final report: No usable research findings were "
            "returned. The report was not generated to avoid presenting an "
            "unsupported answer."
        )
        return {
            "final_report": failure_message,
            "messages": [AIMessage(content=failure_message)],
            **cleared_state,
        }
    
    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "base_url": get_base_url_for_model(configurable.final_report_model),
        "use_responses_api": get_use_responses_api_for_model(configurable.final_report_model),
        "tags": ["langsmith:nostream"]
    }
    
    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            final_report_messages = [HumanMessage(content=final_report_prompt)]
            context_budget = _predict_context_budget(
                "final_report_generation",
                final_report_messages,
                configurable,
                context_window=configurable.final_report_model_context_window,
                reserved_output_tokens=configurable.final_report_model_max_tokens,
            )
            
            # Generate the final report
            final_report = await configurable_model.with_config({
                **writer_model_config,
                "metadata": context_budget.as_metadata("final_report_generation"),
            }).ainvoke(final_report_messages)
            final_report_text = normalize_report_text(final_report)
            if not final_report_text:
                raise ValueError("Final report model returned no readable text content")
            
            # Return successful report generation
            return {
                "final_report": final_report_text,
                "messages": [AIMessage(content=final_report_text)],
                **cleared_state
            }
            
        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }
    
    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }


async def send_report_email(state: AgentState, config: RunnableConfig):
    """Send the generated final report by email when delivery is enabled."""
    configurable = Configuration.from_runnable_config(config)

    if not configurable.email_report_enabled:
        return {"email_delivery_status": "disabled"}

    report = normalize_report_text(state.get("final_report", ""))
    if not report or report.startswith("Error generating final report:"):
        return {"email_delivery_status": "skipped: final report generation failed"}

    if not configurable.email_report_to:
        logging.error(
            "Email delivery is enabled, but EMAIL_REPORT_TO is not configured"
        )
        return {"email_delivery_status": "failed: EMAIL_REPORT_TO is required"}

    try:
        await send_report_email_async(
            report=report,
            recipient=configurable.email_report_to,
            subject=configurable.email_report_subject,
        )
        return {"email_delivery_status": "sent"}
    except Exception as email_error:
        logging.exception("Failed to email the final report")
        return {"email_delivery_status": f"failed: {email_error}"}

# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase
deep_researcher_builder.add_node("send_report_email", send_report_email)           # Optional email delivery

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation") # Research to report
deep_researcher_builder.add_edge("final_report_generation", "send_report_email")   # Report to email
deep_researcher_builder.add_edge("send_report_email", END)                         # Final exit point

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()
