# /usr/bin/env python3
# healthAiBot/healthaibot/utils/agent_utils.py
"""
Utility functions for HealthBot agent operations.
"""


from typing import Optional, Callable
from langchain_tavily import TavilySearch
from healthaibot.utils.utils import HealthBotState


def tavily_search_tool(topic: str) -> str:
    """Search for medical information from trusted sources like NIH, Mayo Clinic, and WebMD."""
    search = TavilySearch()
    query = f"{topic} site:nih.gov OR site:mayoclinic.org OR site:webmd.com"
    return search.invoke(query)


class GraphHelper:
    """
    Helper class for managing graph-related operations.
    """
    def __init__(
        self,
    ) -> None:
        """
        Initialize with the current state.
        """

    def ask_patient(
        self,
        state: HealthBotState,
    ) -> HealthBotState:
        """
        Prompt the user for a health topic and update the state.
        """
        try:
            topic = input("What health topic or medical condition would you like to learn about? ")
        except EOFError:
            print("\nInput ended unexpectedly. Exiting HealthBot.")
            exit(0)
        state.topic = topic
        print(f"You have chosen to learn about: {state.topic}")
        # Add a user message for the tool node
        state.messages = [
            {"role": "user", "content": f"I want to learn about {state.topic}."}
        ]
        return state

    def generate_assistant_message(
        self,
        state: HealthBotState,
    ) -> HealthBotState:
        """
        Generate an assistant message after user input, required for ToolNode.
        """
        # Set context and messages
        state.messages = [
            {"role": "system", "content": "You are a helpful medical information assistant."},
            {"role": "user", "content": f"I want to learn about {state.topic}."},
            {"role": "assistant", "content": f"I'll search for accurate information about {state.topic} from reliable medical sources. Let me use the search tool to find relevant details."}
        ]
        return state

    def ask_for_focus(
        self,
        state: HealthBotState,
    ) -> HealthBotState:
        """
        Ask the user if they want to focus on a specific aspect.
        """
        if not state.focus:
            try:
                focus = input("Do you want to focus on a specific aspect (e.g., symptoms, treatment, prevention)? If yes, enter it, otherwise press Enter: ")
            except EOFError:
                print("\nInput ended unexpectedly. Using no specific focus.")
                focus = ""
            if focus.strip():
                state.focus = focus.strip()
        return state

    def ask_for_focus(
        self,
        state: HealthBotState,
    ) -> HealthBotState:
        """
        Ask the user if they want to focus on a specific aspect.
        """
        if not state.focus:
            try:
                focus = input("Do you want to focus on a specific aspect (e.g., symptoms, treatment, prevention)? If yes, enter it, otherwise press Enter: ")
            except EOFError:
                print("\nInput ended unexpectedly. Using no specific focus.")
                focus = ""
            if focus.strip():
                state.focus = focus.strip()
        return state

    def search_tavily(
        self,
        state: HealthBotState,
    ) -> HealthBotState:
        """
        Search for relevant information using the Tavily ToolNode.
        Record tool call event in state.
        """
        # Example event recording (actual tool call handled by ToolNode)
        event = {
            "event": "tool_call",
            "tool": "tavily_search_tool",
            "topic": state.topic
        }
        state.tool_call_events.append(event)
        return state

    def summarize_results(
        self,
        state: HealthBotState,
    ) -> HealthBotState:
        """
        Summarize the search results using the LLM.
        Enforce: summary must be exactly 3–4 paragraphs, use no outside knowledge, and be strictly based on tool output.
        """
        llm = state.llm
        focus = getattr(state, 'focus', None)
        base_prompt = (
            "Summarize the following medical information for a patient in simple, friendly language. "
            "Your summary must be exactly 3–4 paragraphs. "
            "Use no outside knowledge; only use the information provided below. "
            "Ground every statement in the provided search results. "
            "If the user requests a specific focus (e.g., symptoms, treatment, prevention), focus on that aspect. "
        )
        if focus:
            base_prompt += f"Focus on: {focus}. "
        prompt = base_prompt + f"\nSearch Results:\n{state.search_results}"
        summary = llm.invoke(prompt)
        # Extract content from AIMessage if needed
        if hasattr(summary, 'content'):
            state.summary = summary.content
        else:
            state.summary = str(summary)
        return state

    def present_summary(
        self,
        state: HealthBotState,
    ) -> HealthBotState:
        """
        Present the summarized information to the user.
        """
        print("\nHere is a summary of what you asked about:\n")
        print(state.summary)
        return state

    def comprehension_prompt(
        self,
        state: HealthBotState,
    ) -> HealthBotState:
        """
        Prompt the user for a comprehension check.
        """
        try:
            input("\nPress Enter when you are ready to take a comprehension check.")
        except EOFError:
            print("\nInput ended unexpectedly. Proceeding with comprehension check.")
        return state

    def create_quiz(
        self,
        state: HealthBotState,
    ) -> HealthBotState:
        """
        Create a quiz based on the current state.
        Enforce: quiz must be based only on the summary, single question, and output format.
        """
        llm = state.llm
        previous_questions = getattr(state, 'previous_questions', [])
        prompt = (
            "Create ONE multiple-choice quiz question that is directly relevant to the following summary. "
            "Base the question ONLY on the summary below. Do NOT use any outside knowledge. "
            "The question should have exactly 4 distinct answer options labeled a), b), c), and d). "
            "Do NOT reveal the correct answer. "
            "Do NOT repeat previous questions. "
            "Format your response as:\n"
            "Question:\n<your single question>\n"
            "a) <option 1>\n"
            "b) <option 2>\n"
            "c) <option 3>\n"
            "d) <option 4>\n"
            f"Summary: {state.summary}\n"
            f"Previous questions: {previous_questions}"
        )
        quiz_question = llm.invoke(prompt)
        # Extract content from AIMessage if needed
        if hasattr(quiz_question, 'content'):
            state.quiz_question = quiz_question.content
        else:
            state.quiz_question = str(quiz_question)
        return state

    def present_quiz(
        self,
        state: HealthBotState,
    ) -> HealthBotState:
        """
        Present the quiz question to the user.
        """
        print("\nQuiz Question:\n")
        print(state.quiz_question)
        return state

    def get_quiz_answer(
        self,
        state: HealthBotState,
    ) -> HealthBotState:
        """
        Get the user's answer to the quiz question.
        """
        try:
            answer = input("\nEnter your answer to the quiz question: ")
        except EOFError:
            print("\nInput ended unexpectedly. Exiting HealthBot.")
            exit(0)
        state.quiz_answer = answer
        return state

    def grade_quiz(
        self,
        state: HealthBotState,
    ) -> HealthBotState:
        """
        Grade the user's answer to the quiz question.
        Enforce: use only the summary, output letter grade (A–F) plus brief justification.
        """
        llm = state.llm
        prompt = (
            "Grade the following answer to the quiz question. "
            "Use ONLY the summary below as your data source. "
            "Respond with a letter grade (A, B, C, D, or F) and a brief justification (1–2 sentences) referencing the summary and the correct answer. "
            "Format your response as:\n"
            "Grade: <A–F>\nJustification: <brief explanation>\n"
            f"Summary: {state.summary}\n"
            f"Question: {state.quiz_question}\n"
            f"Answer: {state.quiz_answer}"
        )
        grading = llm.invoke(prompt)
        # Extract content from AIMessage if needed
        if hasattr(grading, 'content'):
            state.grading = grading.content
        else:
            state.grading = str(grading)
        return state

    def present_feedback(
        self,
        state: HealthBotState,
    ) -> HealthBotState:
        """
        Present the feedback to the user.
        """
        print("\nYour grade and feedback:\n")
        print(state.grading)
        # Don't prompt for next action here - let the CLI handle it
        return state