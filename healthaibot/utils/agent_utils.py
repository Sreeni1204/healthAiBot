"""Utility functions for HealthBot agent operations with quiz flow enforcement."""

from datetime import datetime
from healthaibot.utils.utils import HealthBotState
import os
from langchain_tavily import TavilySearch
from langchain.tools import tool


@tool("tavily_search_tool", return_direct=True)
def tavily_search_tool(topic: str) -> str:
    """Search authoritative medical sources (NIH, Mayo Clinic, WebMD) for the given topic."""
    if not os.environ.get("TAVILY_API_KEY"):
        raise ValueError("Missing Tavily API key. Please export TAVILY_API_KEY before running the agent.")
    search = TavilySearch()
    query = f"{topic} site:nih.gov OR site:mayoclinic.org OR site:webmd.com"
    return str(search.invoke(query))


class GraphHelper:
    def __init__(self) -> None:  # No special init needed
        pass

    # ---------------- Core Interaction Nodes -----------------
    def ask_patient(self, state: HealthBotState) -> HealthBotState:
        try:
            topic = input("What health topic or medical condition would you like to learn about? ")
        except EOFError:
            print("\nInput ended unexpectedly. Exiting HealthBot.")
            exit(0)
        state.topic = topic
        print(f"You have chosen to learn about: {state.topic}")
        state.messages.append({
            "role": "user",
            "content": f"I want to learn about {state.topic}.",
            "action": "topic_selection",
            "timestamp": str(datetime.now())
        })
        return state

    def generate_assistant_message(self, state: HealthBotState) -> HealthBotState:
        # Seed conversation and include an assistant message that simulates a tool call request.
        state.messages = [
            {"role": "system", "content": "You are a helpful medical information assistant."},
            {"role": "user", "content": f"I want to learn about {state.topic}."},
            {"role": "assistant", "content": f"Initiating search for {state.topic} using tavily_search_tool.", "tool_calls": [
                {"id": "call_tavily_1", "name": "tavily_search_tool", "arguments": {"topic": state.topic}}
            ]}
        ]
        return state

    def ask_for_focus(self, state: HealthBotState) -> HealthBotState:
        if not state.focus:
            try:
                focus = input("Do you want to focus on a specific aspect (e.g., symptoms, treatment, prevention)? If yes, enter it, otherwise press Enter: ")
            except EOFError:
                print("\nInput ended unexpectedly. Using no specific focus.")
                focus = ""
            if focus.strip():
                state.focus = focus.strip()
            state.messages.append({
                "role": "user",
                "content": f"Focus selection: {state.focus if state.focus else 'No specific focus'}",
                "action": "focus_selection",
                "timestamp": str(datetime.now())
            })
        return state

    def search_tavily(self, state: HealthBotState) -> HealthBotState:
        state.tool_call_events.append({
            "event": "tool_call", "tool": "tavily_search_tool", "topic": state.topic
        })
        return state

    def summarize_results(self, state: HealthBotState) -> HealthBotState:
        llm = state.llm
        focus = getattr(state, 'focus', None)
        # If no real search results (API key missing or fallback placeholder), avoid hallucination
        if not state.search_results or 'Missing Tavily API key' in state.search_results:
            state.summary = (
                "Search unavailable because Tavily API key is missing. "
                "Set TAVILY_API_KEY and restart to generate an evidence-based summary."
            )
            return state
        base_prompt = (
            "You are a medical information assistant. Summarize the search results for a patient.\n\n"
            "MANDATORY FORMAT & RULES (FOLLOW EXACTLY):\n"
            "1. Output MUST be EXACTLY 3 TO 4 paragraphs. No other number is acceptable.\n"
            "2. Paragraphs are separated by ONE blank line (a single empty line).\n"
            "3. Each paragraph MUST be between 3 and 5 sentences (inclusive).\n"
            "4. Use ONLY information present in the search results. If something isn't there, do NOT invent it.\n"
            "5. If an expected aspect is missing, explicitly state: 'The search results do not provide information about <missing aspect>'.\n"
            "6. Do NOT include bullet lists, numbering, headings, markdown, or metadata. Plain paragraphs only.\n"
            "7. If you cannot satisfy ALL rules with given content, write EXACTLY this sentence alone: 'The search results are insufficient to produce a compliant summary.'\n"
            "8. Do NOT mention these instructions or justify your formatting.\n\n"
            "QUALITY GUIDELINES:\n"
            "- Use clear, patient-friendly language.\n"
            "- Avoid redundancy; group related facts.\n"
            "- Prefer concrete facts over vague generalities.\n\n"
            "ACCEPTABLE EXAMPLE (3 paragraphs):\n"
            "Paragraph 1: Overview sentences 1-5.\n"
            "\nParagraph 2: Focused detail sentences 1-4.\n"
            "\nParagraph 3: Limitations + missing info sentences 1-3.\n\n"
            "UNACCEPTABLE EXAMPLES (DO NOT DO):\n"
            "- A single long block (fails rule 1).\n"
            "- 5 paragraphs (fails rule 1).\n"
            "- Paragraphs with 1–2 sentences (fails rule 3).\n"
            "- Bullet lists or headings (fails rule 6).\n\n"
        )
        if focus:
            base_prompt += f"FOCUS REQUIREMENT: Emphasize information about '{focus}'.\n\n"
        base_prompt += (
            "FORMAT: Write EXACTLY 3 TO 4 paragraphs separated by blank lines. Do not include headers, bullet points, or numbered lists.\n\n"
            "SEARCH RESULTS TO SUMMARIZE:\n"
        )
        prompt = base_prompt + (state.search_results or "")
        state.messages.append({
            "role": "user", "content": f"Requesting summary generation for topic: {state.topic}",
            "action": "summarize_results", "focus": focus if focus else "None"
        })
        if llm is None:
            state.summary = "LLM not initialized."
            return state
        summary = llm.invoke(prompt)
        state.summary = summary.content if hasattr(summary, 'content') else str(summary)
        state.messages.append({
            "role": "assistant",
            "content": f"Generated summary for {state.topic} ({len(state.summary)} characters)",
            "action": "summarize_results_complete",
            "summary_length": str(len(state.summary))
        })
        return state

    def present_summary(self, state: HealthBotState) -> HealthBotState:
        print("\nHere is a summary of what you asked about:\n")
        print(state.summary)
        return state

    def comprehension_prompt(self, state: HealthBotState) -> HealthBotState:
        try:
            input("\nPress Enter when you are ready to take a comprehension check.")
        except EOFError:
            print("\nInput ended unexpectedly. Proceeding with comprehension check.")
        return state

    # ---------------- Quiz Flow Nodes -----------------
    def create_quiz(self, state: HealthBotState) -> HealthBotState:
        llm = state.llm
        previous = list(getattr(state, 'previous_questions', []))
        prompt = (
            "Create ONE comprehension question based EXCLUSIVELY on the provided summary below.\n\n"
            "STRICT REQUIREMENTS:\n"
            "1. Create ONLY ONE question (open-ended)\n"
            "2. The question must be answerable ONLY using information from the summary\n"
            "3. No outside knowledge\n"
            "4. Test key understanding of the summary\n"
            "5. Do NOT reveal the answer\n"
            "6. Do NOT repeat any previous questions\n\n"
            "FORMAT: Output just the question text.\n\n"
            f"SUMMARY:\n{state.summary}\n\n"
            f"PREVIOUS QUESTIONS:\n{previous if previous else 'None'}\n\n"
            "QUESTION:"
        )
        state.messages.append({
            "role": "user", "content": f"Requesting quiz question for {state.topic}",
            "action": "create_quiz", "previous_questions_count": str(len(previous))
        })
        if llm is None:
            state.quiz_question = "LLM not initialized to create quiz question."
            return state
        raw = llm.invoke(prompt)
        raw_text = raw.content.strip() if hasattr(raw, 'content') else str(raw).strip()
        candidate_lines = [ln.strip() for ln in raw_text.split('\n') if ln.strip()]
        selected = ""
        for ln in candidate_lines:
            if '?' in ln and not selected:
                selected = ln
            elif '?' in ln and selected:
                state.messages.append({
                    "role": "assistant", "content": f"Discarded extra question: {ln[:80]}",
                    "action": "create_quiz_sanitizer"
                })
        if not selected and candidate_lines:
            selected = candidate_lines[0]
        for prefix in ["question:", "q:", "q1:"]:
            if selected.lower().startswith(prefix):
                selected = selected[len(prefix):].strip()
        if not selected.endswith('?'):
            selected = selected.rstrip('.') + '?'
        if selected not in previous:
            previous.append(selected)
        else:
            state.messages.append({
                "role": "assistant", "content": "Duplicate question detected (kept).",
                "action": "create_quiz_duplicate"
            })
        state.previous_questions = previous
        state.quiz_question = selected
        preview = selected[:100] + "..." if len(selected) > 100 else selected
        state.messages.append({
            "role": "assistant", "content": f"Generated sanitized quiz question for {state.topic}",
            "action": "create_quiz_complete", "question_preview": preview
        })
        return state

    def present_quiz(self, state: HealthBotState) -> HealthBotState:
        print("\nQuiz Question:\n")
        print(state.quiz_question)
        return state

    def get_quiz_answer(self, state: HealthBotState) -> HealthBotState:
        try:
            answer = input("\nEnter your answer to the quiz question: ")
        except EOFError:
            print("\nInput ended unexpectedly. Exiting HealthBot.")
            exit(0)
        state.quiz_answer = answer
        state.messages.append({
            "role": "user", "content": f"Quiz answer: {answer}",
            "action": "quiz_answer_submission", "question": state.quiz_question,
            "timestamp": str(datetime.now())
        })
        return state

    def grade_quiz(self, state: HealthBotState) -> HealthBotState:
        llm = state.llm
        prompt = (
            "You are a strict grading assistant. You must grade the user's answer using ONLY the provided SUMMARY.\n"
            "If the answer invents information not present in the SUMMARY, penalize it.\n"
            "If the answer contradicts the SUMMARY, penalize it.\n"
            "If the answer partially matches, give a middle grade.\n"
            "If the answer fully and accurately reflects key points in the SUMMARY, give a high grade.\n\n"
            "RESTRICTIONS:\n"
            "- You SHOULD NOT use any knowledge outside the SUMMARY.\n"
            "- Do NOT add new facts.\n"
            "- Justification MUST cite only facts/phrases that appear in the SUMMARY.\n\n"
            "ALLOWED GRADES:\nA = Completely accurate based only on SUMMARY\nB = Mostly accurate, minor omissions\nC = Partially accurate, missing important points\nD = Limited accuracy, several errors or omissions\nF = Incorrect or largely not based on SUMMARY\n\n"
            "OUTPUT FORMAT (must follow exactly, no extra lines):\n"
            "Grade: <A|B|C|D|F>\nJustification: <one concise sentence using only SUMMARY info>\n\n"
            f"SUMMARY (sole source of truth):\n{state.summary}\n\n"
            f"QUESTION:\n{state.quiz_question}\n\n"
            f"USER ANSWER:\n{state.quiz_answer}\n\n"
            "Now produce ONLY the required two-line format."
        )
        state.messages.append({
            "role": "user", "content": f"Requesting grade for quiz on {state.topic}",
            "action": "grade_quiz", "user_answer": state.quiz_answer or ""
        })
        if llm is None:
            state.grading = "LLM not initialized to grade quiz answer."
            return state
        grading = llm.invoke(prompt)
        raw_text = grading.content if hasattr(grading, 'content') else str(grading)

        # Post-process to enforce exact format.
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        grade_val = None
        justification_val = None
        for line in lines:
            low = line.lower()
            if low.startswith('grade:') and grade_val is None:
                possible = line.split(':', 1)[1].strip().upper()
                if possible and possible[0] in {'A','B','C','D','F'}:
                    grade_val = possible[0]
            elif low.startswith('justification:') and justification_val is None:
                justification_val = line.split(':', 1)[1].strip()

        # Fallback extraction if not properly structured.
        if grade_val is None:
            # Search for standalone letter
            for cand in ['A','B','C','D','F']:
                if f' {cand} ' in f' {raw_text} ':
                    grade_val = cand
                    break
        if grade_val is None:
            grade_val = 'F'  # default fail-safe
        if not justification_val:
            justification_val = 'Answer lacks sufficient alignment with the provided summary.'

        # Truncate overly long justification
        if len(justification_val) > 280:
            justification_val = justification_val[:277] + '...'

        state.grading = f"Grade: {grade_val}\nJustification: {justification_val}"
        preview = (state.grading[:100] + "...") if len(state.grading) > 100 else state.grading
        state.messages.append({
            "role": "assistant", "content": f"Completed grading for {state.topic}",
            "action": "grade_quiz_complete", "grading_preview": preview
        })
        return state

    def present_feedback(self, state: HealthBotState) -> HealthBotState:
        print("\nYour grade and feedback:\n")
        grading_text = state.grading or ""
        lines = grading_text.strip().split('\n')
        grade_line = ""
        justification_line = ""
        for ln in lines:
            lower = ln.strip().lower()
            if lower.startswith('grade:'):
                grade_line = ln.strip()
            elif lower.startswith('justification:'):
                justification_line = ln.strip()
            elif grade_line and not justification_line and ln.strip():
                justification_line = "Justification: " + ln.strip()
        if grade_line:
            print(grade_line)
        if justification_line:
            print(justification_line)
        if not grade_line or not justification_line:
            print(grading_text)
        # Ask user for next action to set continue_flag
        try:
            choice = input("\nWhat next? (quiz=another quiz question, new=new topic, enter=exit): ").strip().lower()
        except EOFError:
            choice = ""
        if choice == 'quiz':
            state.continue_flag = 'quiz'
        elif choice == 'new':
            state.continue_flag = 'new'
        else:
            state.continue_flag = None
        state.messages.append({
            "role": "user",
            "content": f"Next action choice: {choice or 'exit'}",
            "action": "post_feedback_choice"
        })
        return state