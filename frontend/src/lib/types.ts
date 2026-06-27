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
  | 'transcript'
  | 'bot_start'
  | 'bot_chunk'
  | 'bot_end';

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

/** A code artifact displayed in the side panel */
export interface Artifact {
  id: string;
  language: string;
  code: string;
  title?: string;
}

// ── Control Panel types ──────────────────────────────────────────────────────

export interface TokenStats {
  input: number;
  output: number;
  total: number;
  requests: number;
}

export interface ServiceGauge {
  label: string;
  used: number;
  limit: number;
  pct: number;
}

export interface ServiceLimit {
  service: string;
  label: string;
  gauges: ServiceGauge[];
  status: 'ok' | 'warning' | 'critical';
}

export interface EnvVar {
  key: string;
  value: string;
  is_secret: boolean;
  restart_required: boolean;
  hot_reload: boolean;
}

export interface LogEntry {
  timestamp: number;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  message: string;
  source: string;
}

export interface ServerSettings {
  agent: {
    llm: string;
    temperature: number;
    max_tokens: number;
    groq_model: string;
    openai_model: string;
    memory_enabled: boolean;
    memory_top_k: number;
    checkpoint_enabled: boolean;
  };
  tts: {
    provider: string;
    smallest_voice: string;
    smallest_model: string;
    cartesia_voice: string;
    cartesia_model: string;
  };
  server: {
    host: string;
    port: number;
    log_level: string;
  };
  system_prompt_preview: string;
}

export interface StatsSnapshot {
  uptime_seconds: number;
  session_count: number;
  tokens: TokenStats;
  services: Record<string, {
    rpm: number; rpm_pct: number; rpm_limit: number;
    requests_total: number;
    chars_total: number; chars_pct: number;
    audio_seconds: number; audio_pct: number;
    tokens_today: number; tokens_pct: number;
    observations_month: number; obs_pct: number;
    label: string;
  }>;
}
