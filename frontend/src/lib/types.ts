/* ================================================================
   SETH — Shared TypeScript types
   Mirrors the server-side MessageType enum and defines client state.
   ================================================================ */

/** WebSocket connection + UI state machine */
export type ConnectionState =
  | 'disconnected'
  | 'connecting'
  | 'idle'
  | 'listening'
  | 'processing'
  | 'speaking';

/** Chat message roles */
export type MessageRole = 'user' | 'bot' | 'system';

/** A single chat message for display */
export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: number;
}

/**
 * Server → Client message types.
 * Must match Python `models.MessageType` enum exactly.
 */
export type ServerMessageType =
  | 'text'
  | 'audio'
  | 'response'
  | 'error'
  | 'status'
  | 'audio_response'
  | 'sentence'
  | 'transcript';

/** Incoming message from the Python WebSocket server */
export interface ServerMessage {
  type: ServerMessageType;
  data: string;
}

/** Outgoing message to the Python WebSocket server */
export interface ClientMessage {
  type: 'message' | 'audio' | 'text';
  data: string;
}
