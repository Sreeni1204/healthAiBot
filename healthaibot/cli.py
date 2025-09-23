# /usr/bin/env python3
"""
HealthBot Command Line Interface (CLI)
This script provides a command-line interface for interacting with the HealthBot application.
Users can specify various parameters such as the LLM backend, model name, and temperature.
"""

import argparse
import os

from healthaibot.utils.utils import HealthBotUtils
from healthaibot.graph import build_healthbot_graph


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

    # Preflight: ensure Tavily API key present before starting agent to avoid hallucinated summaries
    if not os.environ.get("TAVILY_API_KEY"):
        print("ERROR: TAVILY_API_KEY environment variable not set.\n")
        print("To use the search tool, export your key first (replace YOUR_KEY):")
        print("\n  export TAVILY_API_KEY=YOUR_KEY\n")
        print("Then re-run the command: healthaibot --llm_type=ollama --model_name=gemma3:1b")
        return

    healthbot = HealthBotUtils(
        llm_type=args.llm_type,
        model_name=args.model_name,
        temperature=args.temperature,
    )
    
    llm = healthbot.get_llm()


    graph = build_healthbot_graph(llm)
    app = graph.compile()

    print("Welcome to HealthBot!")
    # Single invocation; looping & quiz handled internally by graph via continue_flag routing
    state = healthbot.reset_state(llm)
    state = app.invoke(state, config={"recursion_limit": 100})
    # If user chose to start a new topic or additional quizzes, the graph's conditional edges manage it;
    # CLI exits after first completed flow.
    print("\nThank you for using HealthBot. Stay healthy!")