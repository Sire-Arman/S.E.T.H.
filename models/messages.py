"""Message models for WebSocket communication."""
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class MessageType(str, Enum):
    """WebSocket message types."""

    TEXT = "text"
    AUDIO = "audio"
    RESPONSE = "response"
    ERROR = "error"
    STATUS = "status"
    AUDIO_RESPONSE = "audio_response"   # base64 WAV chunk sent to browser
    SENTENCE = "sentence"               # individual sentence text during streaming


class WebSocketMessage(BaseModel):
    """Base WebSocket message model."""

    type: MessageType
    data: str


class TextMessage(WebSocketMessage):
    """Text message model."""

    type: MessageType = MessageType.TEXT


class AudioMessage(WebSocketMessage):
    """Audio message model (base64 encoded)."""

    type: MessageType = MessageType.AUDIO


class ResponseMessage(WebSocketMessage):
    """Response message model."""

    type: MessageType = MessageType.RESPONSE


class ErrorMessage(BaseModel):
    """Error message model."""

    type: MessageType = MessageType.ERROR
    data: str


class StatusMessage(BaseModel):
    """Status message model."""

    type: MessageType = MessageType.STATUS
    data: str
