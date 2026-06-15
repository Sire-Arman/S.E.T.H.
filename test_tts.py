"""Quick test to verify Kokoro TTS synthesis works."""
from services.tts import KokoroTTS

tts = KokoroTTS(voice="af_heart", speed=1.0)
print("Synthesizing 'Hello, how are you?'...")
wav_bytes = tts.synthesize_wav_bytes("Hello, how are you?")
print(f"Result: {len(wav_bytes)} bytes")
if wav_bytes and len(wav_bytes) > 0:
    print("SUCCESS - Kokoro TTS produced audio")
    if wav_bytes[:4] == b'RIFF':
        print("Valid WAV header")
else:
    print("FAILURE")
