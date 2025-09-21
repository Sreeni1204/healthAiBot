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
        default='openai',
        help='Choose LLM backend: openai or ollama'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='gpt-3.5-turbo',
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

    build_healthbot_graph()

    healthbot = HealthBotUtils(
        llm_type=args.llm_type,
        model_name=args.model_name,
        temperature=args.temperature,
    )

    llm = healthbot.get_llm()

    graphhelper = GraphHelper()

    print("Welcome to HealthBot!")
    while True:
        state = healthbot.reset_state(llm)
        state['topic'] = input("What health topic or medical condition would you like to learn about? ")
        print(f"You have chosen to learn about: {state['topic']}")

        # Search and summarize
        state = graphhelper.search_tavily(state)
        # Ask for focus after summary
        focus = input("Do you want to focus on a specific aspect (e.g., symptoms, treatment, prevention)? If yes, enter it, otherwise press Enter: ")
        if focus.strip():
            state['focus'] = focus.strip()
        state = graphhelper.summarize_results(state)
        graphhelper.present_summary(state)
        graphhelper.comprehension_prompt(state)

        # Track previous questions for this topic
        state['previous_questions'] = []
        quiz_active = True
        while quiz_active:
            # Create and present quiz
            state = graphhelper.create_quiz(state)
            question, options = healthbot.parse_quiz(state['quiz_question'])
            print("\nQuiz Question:\n" + question)
            for opt in options:
                print(opt)
            state['previous_questions'].append(question)
            state = graphhelper.get_quiz_answer(state)
            state = graphhelper.grade_quiz(state)
            graphhelper.present_feedback(state)

            next_action = input("Would you like to take another quiz on this topic (enter 'quiz'), learn about a new topic (enter 'new'), or exit (enter 'exit')? ")
            if next_action.lower() == 'quiz':
                continue  # Stay in quiz loop, do not prompt again
            elif next_action.lower() == 'new':
                quiz_active = False  # Break quiz loop, go to new topic
            elif next_action.lower() == 'exit':
                print("Thank you for using HealthBot. Stay healthy!")
                return
            else:
                print("Invalid input. Please enter 'quiz', 'new', or 'exit'.")