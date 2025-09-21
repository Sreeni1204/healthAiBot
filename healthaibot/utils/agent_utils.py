# /usr/bin/env python3
# healthAiBot/healthaibot/utils/agent_utils.py
"""
Utility functions for HealthBot agent operations.
"""

from langchain_tavily import TavilySearch


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
        state: dict,
    ) -> dict:
        """
        Prompt the user for a health topic and update the state.
        """
        topic = input("What health topic or medical condition would you like to learn about? ")
        state['topic'] = topic
        print(f"You have chosen to learn about: {state['topic']}")
        return state

    def search_tavily(
        self,
        state: dict,
    ) -> dict:
        """
        Search for relevant information using the Tavily API.
        """
        tavily_tool = TavilySearch()
        query = state['topic'] + " site:nih.gov OR site:mayoclinic.org OR site:webmd.com"
        results = tavily_tool.run(query)
        state['search_results'] = results
        return state

    def summarize_results(
        self,
        state: dict,
    ) -> dict:
        """
        Summarize the search results using the LLM.
        """
        llm = state['llm']
        focus = state.get('focus', None)
        base_prompt = (
            "Summarize the following medical information for a patient in simple, friendly language. "
            "Use a friendly tone, clear explanations, and actionable advice. "
            "If the user requests a specific focus (e.g., symptoms, treatment, prevention), focus on that aspect. "
        )
        if focus:
            base_prompt += f"Focus on: {focus}. "
        prompt = base_prompt + f"\n{state['search_results']}"
        summary = llm.invoke(prompt)
        state['summary'] = summary
        return state

    def present_summary(
        self,
        state: dict,
    ) -> dict:
        """
        Present the summarized information to the user.
        """
        print("\nHere is a summary of what you asked about:\n")
        print(state['summary'])
        return state

    def comprehension_prompt(
        self,
        state: dict,
    ) -> dict:
        """
        Prompt the user for a comprehension check.
        """
        input("\nPress Enter when you are ready to take a comprehension check.")
        return state

    def create_quiz(
        self,
        state: dict,
    ) -> dict:
        """
        Create a quiz based on the current state.
        """
        llm = state['llm']
        previous_questions = state.get('previous_questions', [])
        prompt = (
            "Create a multiple-choice quiz question that is directly relevant to the following summary. "
            "The question should have exactly 4 distinct answer options labeled a), b), c), and d). "
            "Do NOT reveal the correct answer. "
            "Do NOT repeat previous questions. "
            "Format your response as:\n"
            "Question:\n<your question>\n"
            "a) <option 1>\n"
            "b) <option 2>\n"
            "c) <option 3>\n"
            "d) <option 4>\n"
            f"Summary: {state['summary']}\n"
            f"Previous questions: {previous_questions}"
        )
        quiz_question = llm.invoke(prompt)
        state['quiz_question'] = quiz_question
        return state

    def present_quiz(
        self,
        state: dict,
    ) -> dict:
        """
        Present the quiz question to the user.
        """
        print("\nQuiz Question:\n")
        print(state['quiz_question'])
        return state

    def get_quiz_answer(
        self,
        state: dict,
    ) -> dict:
        """
        Get the user's answer to the quiz question.
        """
        answer = input("\nEnter your answer to the quiz question: ")
        state['quiz_answer'] = answer
        return state

    def grade_quiz(
        self,
        state: dict,
    ) -> dict:
        """
        Grade the user's answer to the quiz question.
        """
        llm = state['llm']
        prompt = (
            "Grade the following answer to the quiz question. "
            "Respond with 'Correct' or 'Incorrect', and provide a brief explanation referencing the summary and the correct answer. "
            "Include citations from the summary if possible.\n"
            f"Summary: {state['summary']}\n"
            f"Question: {state['quiz_question']}\n"
            f"Answer: {state['quiz_answer']}"
        )
        grading = llm.invoke(prompt)
        state['grading'] = grading
        return state

    def present_feedback(
        self,
        state: dict,
    ) -> dict:
        """
        Present the feedback to the user.
        """
        print("\nYour grade and feedback:\n")
        print(state['grading'])
        # Prompt user for next action
        while True:
            next_action = input("Would you like to take another quiz on this topic (enter 'quiz'), or learn about a new topic (enter 'new'), or exit (enter 'exit')? ")
            if next_action.lower() == 'quiz':
                # Optionally, you could loop back to quiz creation and answering
                print("Let's take another quiz on this topic!")
                return 'quiz'
            elif next_action.lower() == 'new':
                print("Let's learn about a new topic!")
                return 'new'
            elif next_action.lower() == 'exit':
                print("Thank you for using HealthBot. Stay healthy!")
                return 'exit'
            else:
                print("Invalid input. Please enter 'quiz', 'new', or 'exit'.")