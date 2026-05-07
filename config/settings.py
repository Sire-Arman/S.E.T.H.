"""Configuration settings for Pipecat AI."""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # Server settings
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8765"))

    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    CARTESIA_API_KEY: str = os.getenv("CARTESIA_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")


    # LLM Configuration
    DEFAULT_LLM: str = os.getenv("DEFAULT_LLM", "cohere")  # cohere, gemini, openai, anthropic, ollama
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "500"))

    # Cohere Settings
    COHERE_MODEL: str = os.getenv("COHERE_MODEL", "command-a-03-2025")

    # OpenAI Settings
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Anthropic Settings
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

    # Ollama Settings (local, no API key needed)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3-voice")
    OLLAMA_AGENT_MODEL: str = os.getenv("OLLAMA_AGENT_MODEL", "qwen3:8b")

    # Groq Settings
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # TTS Configuration
    DEFAULT_TTS: str = os.getenv("DEFAULT_TTS", "kokoro")  # kokoro or cartesia

    # Cartesia TTS Settings
    CARTESIA_VOICE_ID: str = os.getenv("CARTESIA_VOICE_ID", "f786b574-daa5-4673-aa0c-cbe3e8534c02")
    CARTESIA_MODEL: str = os.getenv("CARTESIA_MODEL", "sonic-3")

    # Deepgram STT Settings
    DEEPGRAM_MODEL: str = os.getenv("DEEPGRAM_MODEL", "nova-2")
    DEEPGRAM_LANGUAGE: str = os.getenv("DEEPGRAM_LANGUAGE", "en")

    # System Instruction
    @classmethod
    def get_system_instruction(cls) -> str:
        """Load system prompt from file if exists, otherwise fallback to env."""
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "system.prompt")
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        
        return os.getenv(
            "SYSTEM_INSTRUCTION",
            "You are a helpful voice assistant. Keep responses concise and conversational.",
        )

    SYSTEM_INSTRUCTION: str = "" # Initialized in validate()

    # ── Agent Configuration ──────────────────────────────────────
    AGENT_LLM: str = os.getenv("AGENT_LLM", os.getenv("DEFAULT_LLM", "cohere"))
    AGENT_SEARCH: str = os.getenv("AGENT_SEARCH", "tavily")

    # Tavily Search
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # ── Memory Configuration ─────────────────────────────────────
    MEMORY_ENABLED: bool = os.getenv("MEMORY_ENABLED", "true").lower() == "true"
    MEMORY_DB_PATH: str = os.getenv("MEMORY_DB_PATH", "./data/memory.db")
    MEMORY_TOP_K: int = int(os.getenv("MEMORY_TOP_K", "5"))
    # Model used for lightweight fact extraction (defaults to Cohere)
    MEMORY_LLM: str = os.getenv("MEMORY_LLM", "cohere")
    MEMORY_EMBEDDING_MODEL: str = os.getenv("MEMORY_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ── Checkpoint Configuration ──────────────────────────────────
    CHECKPOINT_ENABLED: bool = os.getenv("CHECKPOINT_ENABLED", "true").lower() == "true"
    CHECKPOINT_DB_PATH: str = os.getenv("CHECKPOINT_DB_PATH", "./data/checkpoints.sqlite")

    # ── Default User ──────────────────────────────────────────────
    # Override with --user flag in CLI, or a UUID once login is added.
    DEFAULT_USER_ID: str = os.getenv("DEFAULT_USER_ID", "user_arman_admin")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls) -> None:
        """Validate that required settings are configured."""
        cls.SYSTEM_INSTRUCTION = cls.get_system_instruction()
        
        if not cls.DEEPGRAM_API_KEY:
            raise ValueError("DEEPGRAM_API_KEY environment variable is not set")
        if not cls.DEFAULT_LLM:
            raise ValueError("DEFAULT_LLM environment variable is not set")
