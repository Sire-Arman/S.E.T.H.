"""Pipecat bot using Daily.co for WebRTC (production-ready)."""
import asyncio
import os
import sys

from dotenv import load_dotenv

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.daily.transport import DailyTransport, DailyParams

load_dotenv()


async def main():
    """Run the Daily.co bot."""
    # Get room URL from command line or environment
    room_url = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DAILY_ROOM_URL")
    if not room_url:
        print("Usage: python daily_bot.py <room_url>")
        print("Or set DAILY_ROOM_URL environment variable")
        sys.exit(1)

    # Daily transport
    transport = DailyTransport(
        room_url,
        None,  # Token (optional for public rooms)
        "Voice Assistant",
        DailyParams(
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
            transcription_enabled=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )

    # Services
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    tts = OpenAITTSService(api_key=os.getenv("OPENAI_API_KEY"), voice="alloy")

    # Context
    messages = [
        {
            "role": "system",
            "content": "You are a helpful voice assistant. Be concise and friendly."
        }
    ]
    context = LLMContext(messages)

    # Pipeline - simplified for pipecat 0.3.0
    pipeline = Pipeline([
        transport.input(),
        stt,
        llm,
        tts,
        transport.output(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True)
    )

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        """Handle first participant joining."""
        await transport.capture_participant_transcription(participant["id"])
        # Welcome message
        messages.append({"role": "system", "content": "Say hello and ask how you can help."})
        await task.queue_frames([LLMMessagesAppendFrame(messages)])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        """Handle participant leaving."""
        print(f"Participant left: {participant}")
        await task.queue_frames([EndFrame()])

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
