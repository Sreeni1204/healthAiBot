# /usr/bin/env python3
"""
healthAiBot graph definition.
"""

from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from healthaibot.utils.agent_utils import GraphHelper, tavily_search_tool
from healthaibot.utils.utils import HealthBotState


# Feedback router for conditional graph edges after present_feedback
def feedback_router(state: HealthBotState):
    if state.continue_flag == 'quiz':
        return "create_quiz"
    elif state.continue_flag == 'new':
        return "ask_patient"
    else:
        return END

def build_healthbot_graph(model) -> StateGraph:
    """
    Build the HealthBot graph with nodes and transitions, using HealthBotState and ToolNode for Tavily search.
    """
    helper = GraphHelper()
    graph = StateGraph(HealthBotState)

    # Create a custom search function that works with the state
    def search_tavily_node(state: HealthBotState) -> HealthBotState:
        """Execute Tavily search and store results in state with message traceability."""
        try:
            # Add tool call message for traceability
            tool_call_message = {
                "role": "assistant",
                "content": f"I'm searching for information about {state.topic} using Tavily search tool.",
                "tool_call_name": "tavily_search_tool",
                "tool_call_arguments": state.topic,
                "timestamp": str(datetime.now())
            }
            state.messages.append(tool_call_message)
            
            # Execute the search
            results = tavily_search_tool(state.topic)
            state.search_results = str(results)
            
            # Add tool response message for traceability
            tool_response_message = {
                "role": "tool",
                "name": "tavily_search_tool",
                "content": f"Search completed for {state.topic}. Found {len(str(results))} characters of information.",
                "tool_call_id": f"tavily_search_{state.topic}_{datetime.now().timestamp()}"
            }
            state.messages.append(tool_response_message)
            
        except Exception as e:
            state.search_results = f"Error searching for {state.topic}: {str(e)}"
            # Add error message for traceability
            error_message = {
                "role": "tool",
                "name": "tavily_search_tool",
                "content": f"Error occurred during search: {str(e)}",
                "error": "True"
            }
            state.messages.append(error_message)
        return state

    # Add all nodes to the graph
    graph.add_node("ask_patient", helper.ask_patient)
    graph.add_node("generate_assistant_message", helper.generate_assistant_message)
    graph.add_node("search_tavily", search_tavily_node)
    graph.add_node("ask_for_focus", helper.ask_for_focus)
    graph.add_node("summarize_results", helper.summarize_results)
    graph.add_node("present_summary", helper.present_summary)
    graph.add_node("comprehension_prompt", helper.comprehension_prompt)
    graph.add_edge("ask_patient", "generate_assistant_message")
    graph.add_edge("generate_assistant_message", "search_tavily")
    graph.add_edge("search_tavily", "ask_for_focus")
    graph.add_edge("ask_for_focus", "summarize_results")
    graph.add_edge("summarize_results", "present_summary")
    graph.add_edge("present_summary", "comprehension_prompt")
    # Properly terminate the graph workflow
    graph.add_edge("comprehension_prompt", END)
    graph.set_entry_point("ask_patient")

    return graph
