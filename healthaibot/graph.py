# /usr/bin/env python3
"""
healthAiBot graph definition.
"""

from langgraph.graph import StateGraph, END

from healthaibot.utils.agent_utils import GraphHelper


def build_healthbot_graph() -> None:
    """
    Build the HealthBot graph with nodes and transitions.
    """
    helper = GraphHelper()
    graph = StateGraph(dict)

    graph.add_node("ask_patient", helper.ask_patient)
    graph.add_node("search_tavily", helper.search_tavily)
    graph.add_node("summarize_results", helper.summarize_results)
    graph.add_node("present_summary", helper.present_summary)
    graph.add_node("comprehension_prompt", helper.comprehension_prompt)
    graph.add_node("create_quiz", helper.create_quiz)
    graph.add_node("present_quiz", helper.present_quiz)
    graph.add_node("get_quiz_answer", helper.get_quiz_answer)
    graph.add_node("grade_quiz", helper.grade_quiz)
    graph.add_node("present_feedback", helper.present_feedback)
    graph.add_edge("ask_patient", "search_tavily")
    graph.add_edge("search_tavily", "summarize_results")
    graph.add_edge("summarize_results", "present_summary")
    graph.add_edge("present_summary", "comprehension_prompt")
    graph.add_edge("comprehension_prompt", "create_quiz")
    graph.add_edge("create_quiz", "present_quiz")
    graph.add_edge("present_quiz", "get_quiz_answer")
    graph.add_edge("get_quiz_answer", "grade_quiz")
    graph.add_edge("grade_quiz", "present_feedback")
    graph.add_edge("present_feedback", END)
    graph.set_entry_point("ask_patient")
