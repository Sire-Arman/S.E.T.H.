"""Pipecat bot client using WebSocket transport."""
import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.websocket.client import WebsocketClientTransport, WebsocketClientParams

load_dotenv()

load_dotenv()


class VoiceBot:
    """Simple voice bot using Pipecat."""

    def __init__(self):
        self.pipeline = None
        self.task = None
        self.runner = None

    async def setup_pipeline(self):
        """Configure the Pipecat pipeline."""
        # Get WebSocket server URL from environment or use default
        ws_url = os.getenv("PIPECAT_SERVER_URL", "ws://localhost:8765")
        
        logger.info(f"Connecting to server at {ws_url}")
        
        # Transport (WebSocket Client) - connects to the server
        transport = WebsocketClientTransport(
            uri=ws_url,
            params=WebsocketClientParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
            )
        )

        # Speech-to-Text (Deepgram)
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY") or os.getenv("OPENAI_API_KEY")
        )

        # LLM (OpenAI)
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        # Text-to-Speech (Cartesia - fast, natural sounding)
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125b8-6b16-4c1e-b36c-24225e291a53",  # Friendly voice
        )

        # Fallback to OpenAI TTS if Cartesia key not available
        if not os.getenv("CARTESIA_API_KEY"):
            tts = OpenAITTSService(
                api_key=os.getenv("OPENAI_API_KEY"),
                voice="alloy"
            )

        # Conversation context with system message
        messages = [
            {
                "role": "system",
                "content": "You are a helpful voice assistant. Keep responses concise and conversational."
            }
        ]
        context = LLMContext(messages)

        # Build pipeline - simpler architecture for pipecat 0.3.0
        self.pipeline = Pipeline([
            transport.input(),      # Audio from user
            stt,                      # Speech to text
            llm,                      # LLM processing
            tts,                      # Text to speech
            transport.output(),       # Audio to user
        ])

        self.task = PipelineTask(
            self.pipeline,
            params=PipelineParams(allow_interruptions=True)
        )

        self.runner = PipelineRunner()

    async def run(self):
        """Start the bot."""
        await self.setup_pipeline()
        await self.runner.run(self.task)


async def main():
    """Main entry point."""
    bot = VoiceBot()
    await bot.run()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
