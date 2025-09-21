import os


class KEYS:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "your-tavily-api-key")