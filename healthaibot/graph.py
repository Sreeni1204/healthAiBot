# /usr/bin/env python3
"""
healthAiBot graph definition.
"""


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
        """Execute Tavily search and store results in state."""
        try:
            results = tavily_search_tool(state.topic)
            state.search_results = str(results)
        except Exception as e:
            state.search_results = f"Error searching for {state.topic}: {str(e)}"
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
    # End the graph after comprehension_prompt - let CLI handle the quiz loop
    graph.set_entry_point("ask_patient")

    return graph
