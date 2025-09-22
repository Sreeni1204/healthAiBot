# /usr/bin/env python3
"""
HealthBot Command Line Interface (CLI)
This script provides a command-line interface for interacting with the HealthBot application.
Users can specify various parameters such as the LLM backend, model name, and temperature.
"""

import argparse

from healthaibot.utils.utils import HealthBotUtils
from healthaibot.graph import build_healthbot_graph
from healthaibot.utils.agent_utils import GraphHelper


def main():
    """
    Main function to run the HealthBot CLI.
    """
    parser = argparse.ArgumentParser(description="HealthBot CLI")
    parser.add_argument(
        '--llm_type',
        choices=['openai', 'ollama'],
        default='ollama',
        help='Choose LLM backend: openai or ollama'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='gemma3:1b',
        help='Model name for LLM'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Temperature for LLM'
    )
    # Add more arguments as needed
    args = parser.parse_args()

    healthbot = HealthBotUtils(
        llm_type=args.llm_type,
        model_name=args.model_name,
        temperature=args.temperature,
    )
    
    llm = healthbot.get_llm()


    graph = build_healthbot_graph(llm)
    app = graph.compile()

    print("Welcome to HealthBot!")
    while True:
        state = healthbot.reset_state(llm)
        # Run the graph workflow to execute the full flow including focus question
        state = app.invoke(state)

        # The graph already handled focus, search, summarization, and summary presentation
        # No need to duplicate these steps here
        
        # Convert state back to HealthBotState if it's a dict
        if isinstance(state, dict):
            from healthaibot.utils.utils import HealthBotState
            state = HealthBotState(**state)

        # Track previous questions for this topic
        if not hasattr(state, 'previous_questions'):
            state.previous_questions = []
        
        # Create GraphHelper for quiz operations
        graphhelper = GraphHelper()
        quiz_active = True
        while quiz_active:
            # Create and present quiz
            state = graphhelper.create_quiz(state)
            question, options = healthbot.parse_quiz(state.quiz_question)
            print("\nQuiz Question:\n" + question)
            for opt in options:
                print(opt)
            state.previous_questions.append(question)
            state = graphhelper.get_quiz_answer(state)
            state = graphhelper.grade_quiz(state)
            state = graphhelper.present_feedback(state)

            try:
                next_action = input("Would you like to take another quiz on this topic (enter 'quiz'), learn about a new topic (enter 'new'), or exit (enter 'exit')? ")
            except EOFError:
                print("\nInput ended unexpectedly. Thank you for using HealthBot. Stay healthy!")
                return
            if next_action.lower() == 'quiz':
                print("Let's take another quiz on this topic!")
                continue  # Stay in quiz loop
            elif next_action.lower() == 'new':
                print("Let's learn about a new topic!")
                quiz_active = False  # Break quiz loop, go to new topic
            elif next_action.lower() == 'exit':
                print("Thank you for using HealthBot. Stay healthy!")
                return
            else:
                print("Invalid input. Please enter 'quiz', 'new', or 'exit'.")