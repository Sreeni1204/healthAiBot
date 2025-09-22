# /usr/bin/env python3
# healthAiBot/healthaibot/utils/agent_utils.py
"""
Utility functions for HealthBot agent operations.
"""

from datetime import datetime
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
        
        # Add user input to messages for traceability
        user_input_message = {
            "role": "user",
            "content": f"I want to learn about {state.topic}.",
            "action": "topic_selection",
            "timestamp": str(datetime.now())
        }
        state.messages.append(user_input_message)
        
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
                
            # Add focus selection to messages for traceability
            focus_message = {
                "role": "user",
                "content": f"Focus selection: {state.focus if state.focus else 'No specific focus'}",
                "action": "focus_selection",
                "timestamp": str(datetime.now())
            }
            state.messages.append(focus_message)
            
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
            "You are a medical information assistant. Your task is to summarize the provided search results for a patient.\n\n"
            "STRICT REQUIREMENTS:\n"
            "1. Write EXACTLY 3-4 paragraphs - no more, no less\n"
            "2. Use ONLY the information provided in the search results below\n"
            "3. Do NOT add any outside knowledge, personal opinions, or information not found in the search results\n"
            "4. Write in simple, patient-friendly language\n"
            "5. Each paragraph should be 3-5 sentences long\n"
            "6. If information is missing from the search results, explicitly state 'The search results do not provide information about [topic]'\n\n"
        )
        
        if focus:
            base_prompt += f"FOCUS REQUIREMENT: Emphasize information about '{focus}' while maintaining the 3-4 paragraph structure.\n\n"
        
        base_prompt += (
            "FORMAT: Write exactly 3-4 paragraphs separated by blank lines. Do not include headers, bullet points, or numbered lists.\n\n"
            "SEARCH RESULTS TO SUMMARIZE:\n"
        )
        
        prompt = base_prompt + state.search_results
        
        # Add LLM request to messages for traceability
        llm_request_message = {
            "role": "user",
            "content": f"Requesting summary generation for topic: {state.topic}",
            "action": "summarize_results",
            "focus": focus if focus else "None"
        }
        state.messages.append(llm_request_message)
        
        summary = llm.invoke(prompt)
        # Extract content from AIMessage if needed
        if hasattr(summary, 'content'):
            state.summary = summary.content
        else:
            state.summary = str(summary)
            
        # Add LLM response to messages for traceability
        llm_response_message = {
            "role": "assistant",
            "content": f"Generated summary for {state.topic} ({len(state.summary)} characters)",
            "action": "summarize_results_complete",
            "summary_length": str(len(state.summary))
        }
        state.messages.append(llm_response_message)
        
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
        Enforce: quiz must be based only on the summary, single question, and answerable using summary alone.
        """
        llm = state.llm
        previous_questions = getattr(state, 'previous_questions', [])
        
        prompt = (
            "Create ONE comprehension question based EXCLUSIVELY on the provided summary below.\n\n"
            "STRICT REQUIREMENTS:\n"
            "1. Create ONLY ONE question - not multiple choice, just a single question\n"
            "2. The question must be answerable ONLY using information from the summary\n"
            "3. Do NOT use any outside knowledge or information not in the summary\n"
            "4. The question should test understanding of key information from the summary\n"
            "5. Do NOT reveal the correct answer in your response\n"
            "6. Do NOT repeat any of the previous questions listed below\n\n"
            "QUESTION TYPES (choose the most appropriate):\n"
            "- What is/are... (factual questions)\n"
            "- Why does/is... (explanation questions)\n"
            "- How does/can... (process questions)\n"
            "- Which statement best describes... (comprehension questions)\n\n"
            "FORMAT: Provide only the question text, nothing else.\n\n"
            f"SUMMARY TO BASE QUESTION ON:\n{state.summary}\n\n"
            f"PREVIOUS QUESTIONS TO AVOID:\n{previous_questions if previous_questions else 'None'}\n\n"
            "YOUR SINGLE QUESTION:"
        )
        
        # Add quiz creation request to messages for traceability
        quiz_request_message = {
            "role": "user",
            "content": f"Requesting quiz question generation for topic: {state.topic}",
            "action": "create_quiz",
            "previous_questions_count": str(len(previous_questions))
        }
        state.messages.append(quiz_request_message)
        
        quiz_question = llm.invoke(prompt)
        # Extract content from AIMessage if needed
        if hasattr(quiz_question, 'content'):
            state.quiz_question = quiz_question.content
        else:
            state.quiz_question = str(quiz_question)
            
        # Add quiz creation response to messages for traceability
        quiz_response_message = {
            "role": "assistant",
            "content": f"Generated quiz question for {state.topic}",
            "action": "create_quiz_complete",
            "question_preview": state.quiz_question[:100] + "..." if len(state.quiz_question) > 100 else state.quiz_question
        }
        state.messages.append(quiz_response_message)
        
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
        
        # Add user's quiz answer to messages for traceability
        quiz_answer_message = {
            "role": "user",
            "content": f"Quiz answer: {answer}",
            "action": "quiz_answer_submission",
            "question": state.quiz_question,
            "timestamp": str(datetime.now())
        }
        state.messages.append(quiz_answer_message)
        
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
            "You are grading a comprehension question. You must provide EXACTLY a letter grade (A, B, C, D, or F) and justification.\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. You MUST output a letter grade: A, B, C, D, or F (no other grades allowed)\n"
            "2. Use EXCLUSIVELY the information from the summary below - absolutely NO outside knowledge\n"
            "3. Your justification must ONLY reference information that appears in the summary\n"
            "4. If the answer contradicts the summary, grade it lower\n"
            "5. If the answer matches information in the summary, grade it higher\n"
            "6. Do NOT add any information not found in the summary\n\n"
            "GRADING SCALE:\n"
            "A = Completely accurate based on summary information\n"
            "B = Mostly accurate with minor gaps based on summary\n"
            "C = Partially accurate but missing key summary points\n"
            "D = Limited accuracy, contradicts some summary information\n"
            "F = Incorrect or completely contradicts the summary\n\n"
            "REQUIRED FORMAT (follow exactly):\n"
            "Grade: [single letter A, B, C, D, or F]\n"
            "Justification: [1-2 sentences explaining the grade based ONLY on summary content]\n\n"
            "SUMMARY (your ONLY data source):\n"
            f"{state.summary}\n\n"
            "QUESTION:\n"
            f"{state.quiz_question}\n\n"
            "STUDENT'S ANSWER:\n"
            f"{state.quiz_answer}\n\n"
            "PROVIDE YOUR GRADE AND JUSTIFICATION:"
        )
        
        # Add grading request to messages for traceability
        grading_request_message = {
            "role": "user",
            "content": f"Requesting grade for quiz answer on topic: {state.topic}",
            "action": "grade_quiz",
            "user_answer": state.quiz_answer
        }
        state.messages.append(grading_request_message)
        
        grading = llm.invoke(prompt)
        # Extract content from AIMessage if needed
        if hasattr(grading, 'content'):
            state.grading = grading.content
        else:
            state.grading = str(grading)
            
        # Add grading response to messages for traceability
        grading_response_message = {
            "role": "assistant",
            "content": f"Completed grading for {state.topic} quiz question",
            "action": "grade_quiz_complete",
            "grading_preview": state.grading[:100] + "..." if len(state.grading) > 100 else state.grading
        }
        state.messages.append(grading_response_message)
        
        return state

    def present_feedback(
        self,
        state: HealthBotState,
    ) -> HealthBotState:
        """
        Present the feedback to the user with proper grade formatting.
        """
        print("\nYour grade and feedback:\n")
        
        # Ensure the grading follows the required format
        grading_text = state.grading
        
        # Extract grade and justification if they're properly formatted
        lines = grading_text.strip().split('\n')
        grade_line = ""
        justification_line = ""
        
        for line in lines:
            if line.strip().lower().startswith('grade:'):
                grade_line = line.strip()
            elif line.strip().lower().startswith('justification:'):
                justification_line = line.strip()
            elif grade_line and not justification_line and line.strip():
                # If we have a grade but no explicit justification line, treat this as justification
                justification_line = "Justification: " + line.strip()
        
        # Display the formatted feedback
        if grade_line:
            print(grade_line)
        if justification_line:
            print(justification_line)
        
        # If formatting is not as expected, display the raw grading
        if not grade_line or not justification_line:
            print(grading_text)
        
        return state