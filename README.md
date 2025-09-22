# HealthAiBot

HealthAiBot is a modular AI-powered assistant for learning about health topics and medical conditions, featuring interactive quizzes and patient-friendly summaries. Built with LangGraph for robust workflow management and supports both OpenAI and open-source LLMs via Ollama.

## Features
- **LangGraph-powered workflow**: Robust state management and error handling
- **Interactive CLI**: Patient-friendly interface with graceful error handling
- **Medical information search**: Uses Tavily API for reliable health information
- **Patient-focused summaries**: Complex medical information simplified for patients
- **Interactive quizzes**: Multiple-choice questions with intelligent grading and feedback
- **Flexible focus options**: Users can specify aspects like symptoms, treatment, or prevention
- **Multi-LLM support**: Works with OpenAI GPT models and Ollama open-source models
- **Robust error handling**: Graceful handling of EOF conditions and edge cases
- **Quiz continuity**: Take multiple quizzes on the same topic or switch to new topics

## Project Structure

```
healthaibot/
├── cli.py            # Main entry point with quiz loop management
├── graph.py          # LangGraph workflow definition and compilation
├── source_code.py    # Legacy monolithic workflow (for reference)
├── config/
│   └── config.py     # Configuration management
└── utils/
    ├── agent_utils.py # Core graph nodes and helper functions
    └── utils.py      # LLM utilities, state models, and quiz parsing
```

### Key Components:
- **cli.py**: Handles command-line interface, argument parsing, and quiz interaction loops
- **graph.py**: Defines the LangGraph workflow with nodes for search, focus, summary, and comprehension
- **agent_utils.py**: Contains all graph node implementations and user interaction methods
- **utils.py**: Provides utility classes for LLM management, state handling, and quiz processing

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd healthAiBot
   ```

2. **Install dependencies using Poetry**:
   ```bash
   poetry install
   ```

3. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

4. **Set up API keys**:
   ```bash
   export TAVILY_API_KEY=<your-tavily-api-key>
   export OPENAI_API_KEY=<your-openai-api-key>  # Only if using OpenAI
   ```

## Prerequisites

- **Python 3.8+**
- **Poetry** for dependency management
- **Tavily API Key** (required for medical information search)
- **OpenAI API Key** (only if using OpenAI models)
- **Ollama** (only if using open-source models) - [Install Ollama](https://ollama.ai/)

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

You can configure LLM backend, model name, and temperature via CLI arguments:

```bash
healthaibot --help
```

### Available Options:
- `--llm_type`: Choose between 'openai' or 'ollama' (default: 'openai')
- `--model_name`: Specify model name (default: 'gpt-3.5-turbo')
- `--temperature`: Set response creativity (default: 0.3)

### Example Configurations:
```bash
# Use Ollama with a specific model
healthaibot --llm_type=ollama --model_name=llama2:7b --temperature=0.1

# Use OpenAI with higher creativity
healthaibot --llm_type=openai --model_name=gpt-4 --temperature=0.7
```

## Troubleshooting

### Common Issues:

1. **EOF Error**: If you encounter EOF errors, ensure you're running in an interactive terminal
2. **Tavily API Issues**: Verify your TAVILY_API_KEY is set correctly
3. **Ollama Connection**: Make sure Ollama is running (`ollama serve`) before using Ollama models
4. **Model Not Found**: For Ollama, pull the model first: `ollama pull <model-name>`

### Exit Options:
- Type `exit` when prompted to quit the application
- Use `Ctrl+C` to force quit at any time
- The application handles EOF gracefully in non-interactive environments

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for workflow management
- Uses [Tavily](https://tavily.com/) for reliable medical information search
- Supports [Ollama](https://ollama.ai/) for open-source LLM integration

## How it Works

The HealthBot follows a structured LangGraph workflow:

1. **Topic Selection**: Patient chooses a health topic or medical condition
2. **Focus Specification**: Optional - patient can specify focus areas (symptoms, treatment, prevention, etc.)
3. **Information Search**: Bot searches reputable medical sources using Tavily API
4. **Summary Generation**: Complex medical information is simplified into patient-friendly language
5. **Interactive Quiz**: Bot generates multiple-choice questions based on the summary
6. **Grading & Feedback**: Intelligent grading with explanatory feedback
7. **Continuation Options**: Patient can take more quizzes, learn new topics, or exit

### Workflow Features
- **State Management**: LangGraph ensures consistent state across all workflow steps
- **Error Recovery**: Graceful handling of interruptions and edge cases
- **User Control**: Patients control the pace and can exit or change topics at any time

## Recent Improvements

- ✅ **Fixed LangGraph Integration**: Resolved tool integration and message handling issues
- ✅ **Enhanced User Experience**: Improved workflow timing (focus questions before summary)
- ✅ **Robust Error Handling**: Added EOF error handling for all user input scenarios
- ✅ **Better State Management**: Enhanced state validation and data flow
- ✅ **Exit/Navigation Fixes**: Improved quiz loop and topic switching functionality

## License

See LICENSE file for details.