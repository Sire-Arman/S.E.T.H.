"""Pipecat WebSocket server for voice bot."""
import asyncio
import json
import base64
from loguru import logger
import websockets

from config import Settings
from services.llm import LLMStore
from services.stt import DeepgramSTT
from services.tts import KokoroTTS, CartesiaTTS
from models import MessageType
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Initialize configuration and services
settings = Settings()
settings.validate()

llm_store = LLMStore(settings)
stt_service = DeepgramSTT(
    api_key=settings.DEEPGRAM_API_KEY,
    model=settings.DEEPGRAM_MODEL,
    language=settings.DEEPGRAM_LANGUAGE,
)

# TTS provider selection
if settings.DEFAULT_TTS == "cartesia":
    tts_service = CartesiaTTS(
        api_key=settings.CARTESIA_API_KEY,
        voice_id=settings.CARTESIA_VOICE_ID,
        model_id=settings.CARTESIA_MODEL,
    )
else:
    tts_service = KokoroTTS(
        voice="af_heart",   # change to any voice in KOKORO_VOICES
        speed=1.0,
    )

logger.add(
    "logs/pipecat.log",
    level=settings.LOG_LEVEL,
    rotation="500 MB",
    retention="7 days",
)
logger.info("Pipecat Voice Bot Server initialized")
logger.info(f"Default LLM: {settings.DEFAULT_LLM}")
logger.info(f"Available LLMs: {llm_store.list_providers()}")


# ------------------------------------------------------------------
# Helper: process user text, stream LLM → TTS → WebSocket
# ------------------------------------------------------------------

async def process_and_stream(user_text: str, messages: list, websocket) -> str:
    """
    Stream LLM response through TTS and send audio chunks to the browser.

    For each sentence:
      1. Send a 'sentence' message with the text
      2. Send an 'audio_response' message with base64-encoded WAV
      3. At the end, send a 'response' message with the full text

    Returns the full concatenated response text.
    """
    logger.info(f"User text: {user_text}")
    messages.append(HumanMessage(content=user_text))

    try:
        logger.debug(f"Streaming from {settings.DEFAULT_LLM} ...")
        sentence_stream = llm_store.invoke_stream(messages)
        full_response = []

        async for sentence, wav_bytes in tts_service.stream_to_client(sentence_stream):
            full_response.append(sentence)

            # Send sentence text
            await websocket.send(json.dumps({
                "type": MessageType.SENTENCE,
                "data": sentence,
            }))

            # Send audio chunk
            if wav_bytes:
                await websocket.send(json.dumps({
                    "type": MessageType.AUDIO_RESPONSE,
                    "data": base64.b64encode(wav_bytes).decode("ascii"),
                }))

        response_text = " ".join(full_response)
        logger.info(f"Bot response: {response_text}")

        # Add bot response to history
        messages.append(AIMessage(content=response_text))

        # Send full text response (signals completion)
        await websocket.send(json.dumps({
            "type": MessageType.RESPONSE,
            "data": response_text,
        }))

        return response_text

    except Exception as api_error:
        logger.error(f"LLM/TTS pipeline error: {api_error}")
        await websocket.send(json.dumps({
            "type": MessageType.ERROR,
            "data": f"API error: {str(api_error)}",
        }))
        return ""


async def handle_client(websocket):
    """Handle incoming WebSocket client connection."""
    client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    logger.info(f"Client connected: {client_id}")

    # Initialize conversation history
    messages = [
        SystemMessage(content=settings.SYSTEM_INSTRUCTION),
    ]

    try:
        async for message in websocket:
            try:
                # Try to parse as JSON
                data = json.loads(message)
                logger.debug(f"Received from {client_id}: {data.get('type')}")

                if data.get("type") == MessageType.TEXT or data.get("type") == 'message':
                    user_message = data.get("data", "").strip()
                    if not user_message:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": MessageType.ERROR,
                                    "data": "Empty message",
                                }
                            )
                        )
                        continue

                    await process_and_stream(user_message, messages, websocket)

                elif data.get("type") == MessageType.AUDIO or data.get("type") == "audio":
                    logger.info("Audio message received")

                    audio_data = data.get("data", "")
                    if not audio_data:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": MessageType.ERROR,
                                    "data": "Empty audio data",
                                }
                            )
                        )
                        continue

                    try:
                        # Decode base64 audio
                        logger.debug("Decoding base64 audio...")
                        audio_bytes = base64.b64decode(audio_data)
                        logger.info(f"Audio received: {len(audio_bytes)} bytes")

                        # Transcribe using Deepgram
                        logger.debug("Transcribing audio with Deepgram...")
                        transcript = await stt_service.transcribe(audio_bytes)

                        if not transcript:
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": MessageType.ERROR,
                                        "data": "Failed to transcribe audio",
                                    }
                                )
                            )
                            continue

                        # Send transcript back to client for display
                        await websocket.send(json.dumps({
                            "type": "transcript",
                            "data": transcript,
                        }))

                        await process_and_stream(transcript, messages, websocket)

                    except Exception as audio_error:
                        logger.error(f"Audio processing error: {audio_error}")
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": MessageType.ERROR,
                                    "data": f"Audio processing error: {str(audio_error)}",
                                }
                            )
                        )

                else:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": MessageType.ERROR,
                                "data": f"Unknown message type: {data.get('type')}",
                            }
                        )
                    )

            except json.JSONDecodeError:
                # Not JSON, treat as plain text message
                user_message = message.strip()
                logger.info(f"Text message from {client_id}: {user_message}")
                await process_and_stream(user_message, messages, websocket)

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                try:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": MessageType.ERROR,
                                "data": f"Server error: {str(e)}",
                            }
                        )
                    )
                except:
                    pass

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Client error: {e}")


async def main():
    """Run the WebSocket server."""
    logger.info(f"Starting server on ws://{settings.SERVER_HOST}:{settings.SERVER_PORT}")

    try:
        async with websockets.serve(
            handle_client, settings.SERVER_HOST, settings.SERVER_PORT
        ):
            logger.info("WebSocket server ready - waiting for connections")
            await asyncio.Future()  # run forever
    except Exception as e:
        logger.error(f"Server error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
