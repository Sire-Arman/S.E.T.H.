"""LLM provider store and interfaces."""
from abc import ABC, abstractmethod
from typing import List
from loguru import logger
from config import Settings
from langchain_core.messages import BaseMessage, AIMessage


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def invoke(self, messages: List[BaseMessage]) -> str:
        """Invoke the LLM with a list of messages."""
        pass

    @abstractmethod
    def invoke_sync(self, messages: List[BaseMessage]) -> str:
        """Synchronous version of invoke."""
        pass


class CohereProvider(LLMProvider):
    """Cohere LLM provider."""

    def __init__(self, api_key: str, model: str, temperature: float):
        """Initialize Cohere provider."""
        from langchain_cohere import ChatCohere

        self.client = ChatCohere(
            cohere_api_key=api_key, model=model, temperature=temperature
        )
        self.model = model

    async def invoke(self, messages: List[BaseMessage]) -> str:
        """Invoke Cohere API asynchronously."""
        import asyncio

        try:
            response = await asyncio.to_thread(self.invoke_sync, messages)
            return response
        except Exception as e:
            logger.error(f"Cohere API error: {e}")
            raise

    def invoke_sync(self, messages: List[BaseMessage]) -> str:
        """Invoke Cohere API synchronously."""
        try:
            response = self.client.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Cohere API error: {e}")
            raise


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, api_key: str, model: str, temperature: float):
        """Initialize OpenAI provider."""
        from langchain_openai import ChatOpenAI

        self.client = ChatOpenAI(
            api_key=api_key, model=model, temperature=temperature
        )
        self.model = model

    async def invoke(self, messages: List[BaseMessage]) -> str:
        """Invoke OpenAI API asynchronously."""
        import asyncio

        try:
            response = await asyncio.to_thread(self.invoke_sync, messages)
            return response
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def invoke_sync(self, messages: List[BaseMessage]) -> str:
        """Invoke OpenAI API synchronously."""
        try:
            response = self.client.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", temperature: float = 0.7):
        """Initialize Gemini provider."""
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature

    async def invoke(self, messages: List[BaseMessage]) -> str:
        """Invoke Gemini API asynchronously."""
        import asyncio

        try:
            response = await asyncio.to_thread(self.invoke_sync, messages)
            return response
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def invoke_sync(self, messages: List[BaseMessage]) -> str:
        """Invoke Gemini API synchronously."""
        try:
            from google import genai
            
            # Convert LangChain messages to Gemini format
            gemini_messages = []
            for msg in messages:
                role = "user" if msg.type == "human" else "model"
                gemini_messages.append({"role": role, "parts": [{"text": msg.content}]})

            # Use google-genai API: client.models.generate_content
            response = self.client.models.generate_content(
                model=self.model,
                contents=gemini_messages,
                config=genai.types.GenerateContentConfig(temperature=self.temperature),
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(self, api_key: str, model: str, temperature: float):
        """Initialize Anthropic provider."""
        from langchain_anthropic import ChatAnthropic

        self.client = ChatAnthropic(
            api_key=api_key, model=model, temperature=temperature
        )
        self.model = model

    async def invoke(self, messages: List[BaseMessage]) -> str:
        """Invoke Anthropic API asynchronously."""
        import asyncio

        try:
            response = await asyncio.to_thread(self.invoke_sync, messages)
            return response
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def invoke_sync(self, messages: List[BaseMessage]) -> str:
        """Invoke Anthropic API synchronously."""
        try:
            response = self.client.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class LLMStore:
    """Store for managing multiple LLM providers."""

    def __init__(self, settings: Settings):
        """Initialize LLM store with configured providers."""
        self.settings = settings
        self.providers = {}
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize all available LLM providers."""
        # Cohere
        if self.settings.COHERE_API_KEY:
            self.providers["cohere"] = CohereProvider(
                api_key=self.settings.COHERE_API_KEY,
                model=self.settings.COHERE_MODEL,
                temperature=self.settings.LLM_TEMPERATURE,
            )
            logger.info("Cohere provider initialized")

        # OpenAI
        if self.settings.OPENAI_API_KEY:
            self.providers["openai"] = OpenAIProvider(
                api_key=self.settings.OPENAI_API_KEY,
                model=self.settings.OPENAI_MODEL,
                temperature=self.settings.LLM_TEMPERATURE,
            )
            logger.info("OpenAI provider initialized")

        # Gemini
        if self.settings.GEMINI_API_KEY:
            self.providers["gemini"] = GeminiProvider(
                api_key=self.settings.GEMINI_API_KEY,
                temperature=self.settings.LLM_TEMPERATURE,
            )
            logger.info("Gemini provider initialized")

        # Anthropic
        if self.settings.ANTHROPIC_API_KEY:
            self.providers["anthropic"] = AnthropicProvider(
                api_key=self.settings.ANTHROPIC_API_KEY,
                model=self.settings.ANTHROPIC_MODEL,
                temperature=self.settings.LLM_TEMPERATURE,
            )
            logger.info("Anthropic provider initialized")

        # Ensure default provider is available
        if self.settings.DEFAULT_LLM not in self.providers:
            logger.error(
                f"Default LLM '{self.settings.DEFAULT_LLM}' not initialized. "
                f"Available providers: {list(self.providers.keys())}"
            )
            raise ValueError(
                f"Default LLM provider '{self.settings.DEFAULT_LLM}' not available"
            )

    def get_provider(self, provider_name: str = None) -> LLMProvider:
        """Get a specific LLM provider."""
        name = provider_name or self.settings.DEFAULT_LLM
        if name not in self.providers:
            raise ValueError(f"Provider '{name}' not available")
        return self.providers[name]

    def list_providers(self) -> List[str]:
        """List all available providers."""
        return list(self.providers.keys())

    async def invoke(
        self, messages: List[BaseMessage], provider: str = None
    ) -> str:
        """Invoke the LLM with a list of messages."""
        llm_provider = self.get_provider(provider)
        return await llm_provider.invoke(messages)
