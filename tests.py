"""Quick test client for WebSocket server."""
import asyncio
import json
import base64
import websockets
from pathlib import Path


async def test_text_message():
    """Test sending a text message."""
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        # Send text message
        message = json.dumps({
            "type": "text",
            "data": "Hello, what's the weather like?"
        })
        await websocket.send(message)
        
        # Receive response
        response = await websocket.recv()
        print(f"Response: {response}")


async def test_audio_message():
    """Test sending an audio message."""
    # For testing, we'll send a dummy audio file
    # In real usage, this would be actual audio data
    uri = "ws://localhost:8765"
    
    # Try to find a test audio file
    audio_file = Path("test_audio.wav")
    if not audio_file.exists():
        print("No test audio file found. Skipping audio test.")
        return
    
    async with websockets.connect(uri) as websocket:
        # Read and encode audio
        with open(audio_file, "rb") as f:
            audio_data = f.read()
        
        encoded_audio = base64.b64encode(audio_data).decode('utf-8')
        
        # Send audio message
        message = json.dumps({
            "type": "audio",
            "data": encoded_audio
        })
        await websocket.send(message)
        
        # Receive response
        response = await websocket.recv()
        print(f"Response: {response}")


async def main():
    """Run tests."""
    print("Testing Pipecat AI Voice Bot Server")
    print("=" * 50)
    
    try:
        print("\n1. Testing text message...")
        await test_text_message()
        print("✓ Text message test passed")
    except Exception as e:
        print(f"✗ Text message test failed: {e}")
    
    try:
        print("\n2. Testing audio message...")
        await test_audio_message()
        print("✓ Audio message test passed")
    except Exception as e:
        print(f"✗ Audio message test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
