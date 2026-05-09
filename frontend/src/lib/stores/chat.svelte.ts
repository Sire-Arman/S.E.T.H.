/* ================================================================
   ChatStore — Reactive WebSocket + chat state using Svelte 5 runes.

   Singleton module exporting reactive state and actions.
   Uses $state/$derived for fine-grained reactivity.
   ================================================================ */

import { AudioPlayer } from '../services/audio-player';
import { AudioRecorder } from '../services/audio-recorder';
import { WakeWordDetector } from '../services/wake-word';
import type { ConnectionState, ChatMessage, ServerMessage, MessageRole, Artifact } from '../types';

// ── Helpers ──────────────────────────────────────────────────────

function uid(): string {
  return Math.random().toString(36).substring(2, 10);
}

function getWsUrl(): string {
  if (import.meta.env.DEV) {
    // Dev mode: use Vite's WebSocket proxy
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${window.location.host}/ws`;
  }
  // Production / Tauri: direct connection or env override
  return import.meta.env.VITE_WS_URL || 'ws://127.0.0.1:8765';
}

// ── Services (non-reactive singletons) ──────────────────────────

const audioPlayer = new AudioPlayer();
const audioRecorder = new AudioRecorder();
const wakeWord = new WakeWordDetector('hey');

// ── Reactive State ──────────────────────────────────────────────

let connectionState = $state<ConnectionState>('disconnected');
let messages = $state<ChatMessage[]>([]);
let isRecording = $state(false);
let wakeWordEnabled = $state(false);
let activeArtifact = $state<Artifact | null>(null);
let artifacts = $state<Artifact[]>([]);

// ── Derived ─────────────────────────────────────────────────────

const isConnected = $derived(
  connectionState !== 'disconnected' && connectionState !== 'connecting',
);
const canSend = $derived(
  connectionState === 'idle' || connectionState === 'speaking',
);
const wakeWordSupported = wakeWord.supported;

// ── WebSocket ───────────────────────────────────────────────────

let ws: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

function addMessage(content: string, role: MessageRole): void {
  messages.push({
    id: uid(),
    role,
    content,
    timestamp: Date.now(),
  });
}

function handleServerMessage(msg: ServerMessage): void {
  switch (msg.type) {
    case 'sentence':
      addMessage(msg.data, 'bot');
      if (connectionState === 'processing') connectionState = 'speaking';
      break;

    case 'audio_response':
      audioPlayer.enqueue(msg.data);
      if (connectionState !== 'speaking') connectionState = 'speaking';
      break;

    case 'response':
      // Final full response — sentences already displayed individually.
      break;

    case 'transcript':
      addMessage(msg.data, 'user');
      connectionState = 'processing';
      break;

    case 'status':
      if (msg.data === 'processing') connectionState = 'processing';
      break;

    case 'error':
      addMessage(`Error: ${msg.data}`, 'system');
      connectionState = 'idle';
      break;

    default:
      if (msg.data) addMessage(msg.data, 'bot');
  }
}

// ── Audio player → state sync ───────────────────────────────────

audioPlayer.setStateCallback((playing) => {
  if (!playing && connectionState === 'speaking') {
    connectionState = 'idle';
  }
});

// ── Wake word → auto-record ─────────────────────────────────────

wakeWord.setCallback(async () => {
  if (!ws || ws.readyState !== WebSocket.OPEN || isRecording) return;
  addMessage('Wake word detected!', 'system');
  await startRecording();
  // Auto-stop after 5 seconds
  setTimeout(() => {
    if (isRecording) stopRecording();
  }, 5000);
});

// ── Public actions ──────────────────────────────────────────────

function connect(): void {
  if (ws && ws.readyState <= WebSocket.OPEN) return;

  connectionState = 'connecting';
  const url = getWsUrl();
  ws = new WebSocket(url);

  ws.onopen = () => {
    connectionState = 'idle';
  };

  ws.onmessage = (event) => {
    try {
      const msg: ServerMessage = JSON.parse(event.data);
      handleServerMessage(msg);
    } catch {
      addMessage(event.data, 'bot');
    }
  };

  ws.onerror = () => {
    connectionState = 'disconnected';
  };

  ws.onclose = () => {
    connectionState = 'disconnected';
    ws = null;
    // Auto-reconnect after 3s
    if (reconnectTimer) clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(() => connect(), 3000);
  };
}

function disconnect(): void {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  ws?.close();
  ws = null;
  connectionState = 'disconnected';
}

function sendText(text: string): void {
  const trimmed = text.trim();
  if (!trimmed || !ws || ws.readyState !== WebSocket.OPEN) return;

  addMessage(trimmed, 'user');
  ws.send(JSON.stringify({ type: 'message', data: trimmed }));
  connectionState = 'processing';
}

async function startRecording(): Promise<void> {
  if (isRecording) return;
  try {
    await audioRecorder.start();
    isRecording = true;
    connectionState = 'listening';
  } catch {
    addMessage('Microphone access denied', 'system');
  }
}

async function stopRecording(): Promise<void> {
  if (!isRecording) return;
  try {
    const base64 = await audioRecorder.stop();
    isRecording = false;
    connectionState = 'processing';
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'audio', data: base64 }));
    }
  } catch (err) {
    isRecording = false;
    connectionState = 'idle';
    console.error('[SETH] Recording error:', err);
  }
}

async function toggleRecording(): Promise<void> {
  if (isRecording) {
    await stopRecording();
  } else {
    await startRecording();
  }
}

function toggleWakeWord(): boolean {
  const nowEnabled = wakeWord.toggle();
  wakeWordEnabled = nowEnabled;
  if (nowEnabled) {
    addMessage('Wake word enabled — say "HEY"', 'system');
  } else {
    addMessage('Wake word disabled', 'system');
  }
  return nowEnabled;
}

function openArtifact(artifact: Artifact): void {
  activeArtifact = artifact;
  // Track all opened artifacts
  if (!artifacts.find((a) => a.id === artifact.id)) {
    artifacts.push(artifact);
  }
}

function closeArtifact(): void {
  activeArtifact = null;
}

// ── Exported reactive store ─────────────────────────────────────

export const chat = {
  // Reactive getters (Svelte 5 runes propagate through getters)
  get connectionState() { return connectionState; },
  get messages() { return messages; },
  get isConnected() { return isConnected; },
  get canSend() { return canSend; },
  get isRecording() { return isRecording; },
  get wakeWordEnabled() { return wakeWordEnabled; },
  get wakeWordSupported() { return wakeWordSupported; },
  get activeArtifact() { return activeArtifact; },
  get artifacts() { return artifacts; },

  // Actions
  connect,
  disconnect,
  sendText,
  startRecording,
  stopRecording,
  toggleRecording,
  toggleWakeWord,
  addMessage,
  openArtifact,
  closeArtifact,
};
