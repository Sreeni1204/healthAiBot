# /usr/bin/env python3
"""
healthAiBot graph definition.
"""

from datetime import datetime
import json
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from healthaibot.utils.agent_utils import GraphHelper, tavily_search_tool
from healthaibot.utils.utils import HealthBotState
try:
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
except ImportError:  # Fallback if package structure differs
    from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage  # type: ignore
    try:
        from langchain.schema import ToolMessage  # type: ignore
    except ImportError:  # Define minimal ToolMessage
        class ToolMessage(HumanMessage):  # type: ignore
            def __init__(self, content: str, name: str, tool_call_id: str = ""):
                super().__init__(content=content)
                self.name = name
                self.tool_call_id = tool_call_id


# Feedback router for conditional graph edges after present_feedback
def feedback_router(state: HealthBotState):
    if state.continue_flag == 'quiz':
        return "create_quiz"
    elif state.continue_flag == 'new':
        return "reset_topic_state"
    else:
        return END

def build_healthbot_graph(model) -> StateGraph:
    """
    Build the HealthBot graph with nodes and transitions, using HealthBotState and ToolNode for Tavily search.
    """
    helper = GraphHelper()
    graph = StateGraph(HealthBotState)

    # Real ToolNode usage with tavily_search_tool defined as a LangChain tool.
    tool_node = ToolNode([tavily_search_tool])

    def ensure_tool_call(state: HealthBotState) -> HealthBotState:
        """Ensure there is an AIMessage with a tool call for tavily_search_tool.

        Converts legacy dict messages to LangChain message objects if needed so ToolNode can parse them.
        """
        if not state.messages:
            return state

        # If messages are dicts, convert them.
        if not isinstance(state.messages[0], BaseMessage):
            converted = []
            for m in state.messages:
                role = m.get("role")
                content = m.get("content", "")
                if role == "system":
                    converted.append(SystemMessage(content=content))
                elif role == "user":
                    converted.append(HumanMessage(content=content))
                elif role == "assistant":
                    tool_calls = m.get("tool_calls") or []
                    lc_tool_calls = []
                    for tc in tool_calls:
                        args = tc.get("arguments") or tc.get("args") or {}
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {"raw": args}
                        lc_tool_calls.append({
                            "id": tc.get("id"),
                            "name": tc.get("name"),
                            "args": args,
                        })
                    converted.append(AIMessage(content=content, tool_calls=lc_tool_calls))
                elif role == "tool":
                    converted.append(ToolMessage(content=content, name=m.get("name", "tool"), tool_call_id=m.get("id") or m.get("tool_call_id", "")))
                else:
                    converted.append(HumanMessage(content=content))
            state.messages = converted

        # Now operate on LangChain messages
        last = state.messages[-1]
        if isinstance(last, AIMessage):
            if not getattr(last, 'tool_calls', None):
                last.tool_calls = [{
                    "id": "auto_tavily_" + str(datetime.now().timestamp()),
                    "name": "tavily_search_tool",
                    "args": {"topic": state.topic}
                }]
        else:
            # Append a new AIMessage with tool call
            state.messages.append(AIMessage(content=f"Initiating search for {state.topic}", tool_calls=[{
                "id": "auto_tavily_" + str(datetime.now().timestamp()),
                "name": "tavily_search_tool",
                "args": {"topic": state.topic}
            }]))
        return state

    def process_tool_output(state: HealthBotState) -> HealthBotState:
        """Extract last tool message content into state.search_results for downstream summarization."""
        # Search from end for a ToolMessage (LangChain) first
        found = False
        for msg in reversed(state.messages):
            if isinstance(msg, ToolMessage):
                state.search_results = msg.content
                found = True
                break
            # Legacy dict form
            if isinstance(msg, dict) and msg.get("role") == "tool":
                state.search_results = msg.get("content", "")
                found = True
                break
        if not found or not state.search_results:
            # Fallback: invoke tool directly
            try:
                import os
                if not os.environ.get("TAVILY_API_KEY"):
                    state.search_results = "Missing Tavily API key. Please export TAVILY_API_KEY to enable search."
                else:
                    # Prefer direct call; tavily_search_tool supports .invoke when decorated
                    if hasattr(tavily_search_tool, 'invoke'):
                        fallback_result = tavily_search_tool.invoke({"topic": state.topic})
                    else:
                        fallback_result = tavily_search_tool(state.topic or "")
                    state.search_results = str(fallback_result)
                    # Record as ToolMessage for consistency
                    state.messages.append(ToolMessage(content=f"(Fallback) Search completed for {state.topic}.", name="tavily_search_tool", tool_call_id="fallback"))
            except Exception as e:
                state.search_results = f"No search results captured and fallback failed: {e}"
        return state

    def reset_topic_state(state: HealthBotState) -> HealthBotState:
        """Clear topic-specific fields before starting a new topic cycle."""
        state.focus = None
        state.search_results = None
        state.summary = None
        state.quiz_question = None
        state.quiz_answer = None
        state.grading = None
        state.previous_questions = []
        state.continue_flag = None
        # Do not clear messages entirely to retain audit trail; append a separator marker
        try:
            from langchain_core.messages import HumanMessage  # type: ignore
            state.messages.append(HumanMessage(content="--- NEW TOPIC ---"))
        except Exception:
            state.messages.append({"role": "user", "content": "--- NEW TOPIC ---"})
        return state

    # Add all nodes to the graph
    # Core information gathering & presentation nodes
    graph.add_node("ask_patient", helper.ask_patient)
    graph.add_node("generate_assistant_message", helper.generate_assistant_message)
    graph.add_node("ensure_tool_call", ensure_tool_call)
    graph.add_node("search_tavily", tool_node)
    graph.add_node("process_tool_output", process_tool_output)
    graph.add_node("ask_for_focus", helper.ask_for_focus)
    graph.add_node("summarize_results", helper.summarize_results)
    graph.add_node("present_summary", helper.present_summary)
    graph.add_node("comprehension_prompt", helper.comprehension_prompt)

    # Quiz / feedback flow nodes (previously unwired)
    graph.add_node("create_quiz", helper.create_quiz)
    graph.add_node("present_quiz", helper.present_quiz)
    graph.add_node("get_quiz_answer", helper.get_quiz_answer)
    graph.add_node("grade_quiz", helper.grade_quiz)
    graph.add_node("present_feedback", helper.present_feedback)
    graph.add_node("reset_topic_state", reset_topic_state)
    graph.add_edge("ask_patient", "generate_assistant_message")
    graph.add_edge("generate_assistant_message", "ensure_tool_call")
    graph.add_edge("ensure_tool_call", "search_tavily")
    graph.add_edge("search_tavily", "process_tool_output")
    graph.add_edge("process_tool_output", "ask_for_focus")
    graph.add_edge("ask_for_focus", "summarize_results")
    graph.add_edge("summarize_results", "present_summary")
    graph.add_edge("present_summary", "comprehension_prompt")

    # After comprehension prompt we always generate one quiz for now.
    # If future logic sets continue_flag we can branch using a conditional edge.
    graph.add_edge("comprehension_prompt", "create_quiz")
    graph.add_edge("create_quiz", "present_quiz")
    graph.add_edge("present_quiz", "get_quiz_answer")
    graph.add_edge("get_quiz_answer", "grade_quiz")
    graph.add_edge("grade_quiz", "present_feedback")
    # Conditional routing after feedback: END (default) | quiz (new question) | new (restart)
    graph.add_conditional_edges(
        "present_feedback",
        feedback_router,
        {
            "create_quiz": "create_quiz",  # repeat quiz with new question
            "reset_topic_state": "reset_topic_state",  # reset then new topic
            END: END,
        },
    )
    # After resetting topic-specific state, return to ask_patient for a fresh cycle
    graph.add_edge("reset_topic_state", "ask_patient")
    graph.set_entry_point("ask_patient")

    return graph
