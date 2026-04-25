"""LLM provider store and interfaces."""
from abc import ABC, abstractmethod
from typing import AsyncIterator, List
from loguru import logger
from config import Settings
from langchain_core.messages import BaseMessage, AIMessage
import re


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

    @abstractmethod
    async def invoke_stream(self, messages: List[BaseMessage]) -> AsyncIterator[str]:
        """
        Stream the LLM response and yield one complete sentence at a time.

        Sentences are delimited by  .  !  ?  ;  so the TTS engine always
        receives a meaningful, intonation-complete chunk rather than raw tokens.
        Yields the remaining buffer after the final sentence-end token too.
        """
        pass

    # ------------------------------------------------------------------
    # Shared helper: accumulate token stream -> yield sentences
    # ------------------------------------------------------------------
    @staticmethod
    async def _stream_sentences(
        token_iter,            # async iterable of token strings
        get_content,           # callable(token) -> str
        min_words: int = 5     # minimum words before yielding
    ) -> AsyncIterator[str]:
        """
        Internal helper that turns a raw token stream into sentence chunks.
        Accumulates chunks until they have at least ~5 words to prevent
        fragmented TTS playback while keeping TTFT low.
        """
        buffer = ""
        # Split after . ! ? ; : \n but only if followed by space or end of string
        _BOUNDARY = re.compile(r"(?<=[.!?;:\n])(?=\s|$)")

        async for token in token_iter:
            content = get_content(token)
            if not content:
                continue
            buffer += content
            
            # Find all complete "sentences" in the buffer
            parts = _BOUNDARY.split(buffer)
            if len(parts) > 1:
                complete_text = "".join(parts[:-1])
                
                # Yield if it has enough words, OR if it contains a newline (paragraph)
                if len(complete_text.split()) >= min_words or "\n" in complete_text:
                    clean_text = complete_text.strip()
                    if clean_text:
                        yield clean_text
                    buffer = parts[-1]

        # Yield whatever is left
        remainder = buffer.strip()
        if remainder:
            yield remainder


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

    async def invoke_stream(self, messages: List[BaseMessage]) -> AsyncIterator[str]:
        """Stream Cohere response as complete sentences."""
        async for sentence in self._stream_sentences(self.client.astream(messages), lambda t: t.content):
            yield sentence


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
    async def invoke_stream(self, messages: List[BaseMessage]) -> AsyncIterator[str]:
        """Stream OpenAI response as complete sentences."""
        async for sentence in self._stream_sentences(self.client.astream(messages), lambda t: t.content):
            yield sentence


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

    async def invoke_stream(self, messages: List[BaseMessage]) -> AsyncIterator[str]:
        """Stream Gemini response as complete sentences via google-genai."""
        from google import genai

        gemini_messages = [
            {"role": "user" if m.type == "human" else "model",
             "parts": [{"text": m.content}]}
            for m in messages
        ]

        async def _token_gen():
            response_stream = await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=gemini_messages,
                config=genai.types.GenerateContentConfig(
                    temperature=self.temperature
                ),
            )
            async for chunk in response_stream:
                yield chunk

        async for sentence in self._stream_sentences(
            _token_gen(), lambda c: c.text or ""
        ):
            yield sentence

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
    async def invoke_stream(self, messages: List[BaseMessage]) -> AsyncIterator[str]:
        """Stream Anthropic response as complete sentences."""
        async for sentence in self._stream_sentences(self.client.astream(messages), lambda t: t.content):
            yield sentence


class LLMStore:
    """Store for managing multiple LLM providers."""

    def __init__(self, settings: Settings):
        """Initialize LLM store with configured providers."""
        self.settings = settings
        self.providers = {}
        self._initialize_providers()

    # Map of provider name -> (factory function, required api_key attr, extra kwargs builder)
    _PROVIDER_FACTORIES = {
        "cohere":    lambda s: CohereProvider(
                         api_key=s.COHERE_API_KEY,
                         model=s.COHERE_MODEL,
                         temperature=s.LLM_TEMPERATURE,
                     ),
        "openai":    lambda s: OpenAIProvider(
                         api_key=s.OPENAI_API_KEY,
                         model=s.OPENAI_MODEL,
                         temperature=s.LLM_TEMPERATURE,
                     ),
        "gemini":    lambda s: GeminiProvider(
                         api_key=s.GEMINI_API_KEY,
                         temperature=s.LLM_TEMPERATURE,
                     ),
        "anthropic": lambda s: AnthropicProvider(
                         api_key=s.ANTHROPIC_API_KEY,
                         model=s.ANTHROPIC_MODEL,
                         temperature=s.LLM_TEMPERATURE,
                     ),
    }

    # Map of provider name -> settings attribute that holds its API key
    _API_KEY_ATTRS = {
        "cohere":    "COHERE_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "gemini":    "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    def _initialize_providers(self) -> None:
        """
        Initialize ONLY the provider set as DEFAULT_LLM in settings.

        Only the single default provider is imported and instantiated at startup.
        To switch provider: change DEFAULT_LLM in .env and restart the server.
        """
        name = self.settings.DEFAULT_LLM

        if name not in self._PROVIDER_FACTORIES:
            raise ValueError(
                f"Unknown LLM provider '{name}'. "
                f"Valid choices: {list(self._PROVIDER_FACTORIES.keys())}"
            )

        key_attr = self._API_KEY_ATTRS[name]
        api_key  = getattr(self.settings, key_attr, "")

        if not api_key:
            raise ValueError(
                f"DEFAULT_LLM is set to '{name}' but {key_attr} is not configured in .env"
            )

        self.providers[name] = self._PROVIDER_FACTORIES[name](self.settings)
        logger.info(f"LLM provider initialized: {name}")

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
        """Invoke the LLM with a list of messages (full response)."""
        llm_provider = self.get_provider(provider)
        return await llm_provider.invoke(messages)

    async def invoke_stream(
        self, messages: List[BaseMessage], provider: str = None
    ) -> AsyncIterator[str]:
        """
        Stream sentences from the LLM response.

        Yields one complete sentence (ending in . ! ? ;) at a time so the TTS
        engine can start speaking before the LLM has finished generating.
        """
        llm_provider = self.get_provider(provider)
        async for sentence in llm_provider.invoke_stream(messages):
            yield sentence
