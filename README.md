# HealthAiBot

HealthAiBot is a modular AI-powered assistant for learning about health topics and medical conditions, featuring interactive quizzes and patient-friendly summaries. It supports both OpenAI and open-source LLMs via Ollama.

## Features
- Modular codebase for easy extension and maintenance
- Interactive CLI for patient engagement
- Summarizes medical topics in simple language
- Generates multiple-choice quizzes with feedback
- Supports OpenAI and Ollama LLMs
- Uses reputable sources via Tavily search

## Project Structure

```
healthaibot/
├── cli.py            # Main entry point for CLI
├── graph.py          # Graph workflow builder
├── source_code.py    # Legacy monolithic workflow (for reference)
├── config/
│   └── config.py     # Configuration management
├── utils/
│   ├── agent_utils.py # Graph and agent helpers
│   └── utils.py      # LLM, state, and quiz utilities
```

## Installation

1. Clone the repository:
	```bash
	git clone <repo-url>
	cd healthAiBot
	```
2. Install dependencies:
	```bash
	poetry install
	```

## Usage

Before running the scripts, export the following env variables:

```bash
export OPENAI_API_KEY=<valid open ai key>
```

```bash
export TAVILY_API_KEY=<valid tavily api key>
```

`OPENAI_API_KEY` is required only if using OpenAI llms.

Run the CLI:

Use Ollama:

```bash
healthaibot --llm_type=ollama --model_name=gemma3:1b
```
Or use OpenAI:
```bash
healthaibot --llm_type=openai --model_name=gpt-3.5-turbo
```

## Configuration

You can configure LLM backend, model name, and temperature via CLI arguments. See `cli.py` for details.

## How it Works

1. Patient chooses a health topic.
2. Bot searches reputable sources and summarizes information.
3. Patient can focus on specific aspects (symptoms, treatment, etc.).
4. Bot generates quizzes and grades answers interactively.
5. Patient can continue with more quizzes or switch topics.

## License

See LICENSE file for details.