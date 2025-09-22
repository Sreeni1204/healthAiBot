# /usr/bin/env python3
# healthAiBot/healthaibot/utils/utils.py
"""
Utility functions for HealthBot operations.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

from langchain_openai import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama


class HealthBotState(BaseModel):
    messages: List[Dict[str, str]] = Field(
        default_factory=list, 
        description="List of conversation messages including tool calls for traceability"
    )
    topic: Optional[str] = None
    focus: Optional[str] = None
    search_results: Optional[str] = None
    summary: Optional[str] = None
    question: Optional[str] = None
    quiz_question: Optional[str] = None
    quiz_answer: Optional[str] = None
    quiz_grade: Optional[str] = None
    grading: Optional[str] = None
    continue_flag: Optional[str] = None
    previous_questions: List[str] = Field(default_factory=list)
    tool_call_events: List[Any] = Field(
        default_factory=list, 
        description="Legacy tool call tracking - use messages for better traceability"
    )
    llm: Optional[Any] = None


class HealthBotUtils:
    """
    Utility functions for HealthBot operations.
    """
    def __init__(
        self,
        llm_type: str,
        model_name: str,
        temperature: float = 0.7,
    ) -> None:
        """
        Initialize the HealthBotUtils class.
        Parameters:
            llm_type: Type of LLM to use ('openai' or 'ollama').
            model_name: Name of the model to use.
            temperature: Sampling temperature for the LLM.
        """
        self.llm_type = llm_type
        self.model_name = model_name
        self.temperature = temperature

    def get_llm(
        self,
    ) -> ChatOpenAI | ChatOllama:
        """
        Get the LLM instance based on the specified type.
        Returns:
            An instance of ChatOpenAI or ChatOllama.
        """
        if self.llm_type == "openai":
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature
            )
        elif self.llm_type == "ollama":
            return ChatOllama(
                model=self.model_name,
                temperature=self.temperature
            )
        else:
            raise ValueError("Unsupported LLM type. Choose 'openai' or 'ollama'.")

    def reset_state(
        self,
        llm: ChatOpenAI | ChatOllama,
    ) -> HealthBotState:
        """
        Reset the state of the HealthBot.
        """
        # Clear previous health information to maintain privacy
        return HealthBotState(llm=llm)

    def parse_quiz(
        self,
        quiz_text: str,
    ) -> tuple[str, list[str]]:
        """
        Parse the quiz text into a question and options.
        Now handles both single questions and multiple choice questions.
        """
        # Simple parser to split question and options
        lines = quiz_text.strip().split('\n')
        question = ""
        options = []
        
        for line in lines:
            # Check for multiple choice options
            if line.startswith("a)") or line.startswith("b)") or line.startswith("c)") or line.startswith("d)"):
                options.append(line)
            elif line.lower().startswith("question:"):
                question = line[len("Question:"):].strip()
            elif line and not line.startswith("summary:") and not line.startswith("previous questions:"):
                if question:
                    question += " " + line.strip()
                else:
                    question = line.strip()
        
        # If no options found, this is a single question (not MCQ)
        # Return empty options list to indicate open-ended question
        return question.strip(), options
