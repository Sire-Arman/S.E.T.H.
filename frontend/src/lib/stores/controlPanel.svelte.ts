/* ================================================================
   controlPanel.svelte.ts
   Reactive store for the SETH Control Panel.
   Polls the HTTP API every 3s for stats/logs.
   Handles save operations with restart-required confirmation.
   ================================================================ */

import type {
  StatsSnapshot, ServerSettings, EnvVar,
  ServiceLimit, LogEntry
} from '../types';

const API = '/api';   // Vite proxies this to http://127.0.0.1:8766

// ── State ─────────────────────────────────────────────────────────────────

let isOpen         = $state(false);
let activeTab      = $state<'overview' | 'agent' | 'tts' | 'limits' | 'env' | 'logs'>('overview');
let loading        = $state(false);
let error          = $state<string | null>(null);

let stats          = $state<StatsSnapshot | null>(null);
let settings       = $state<ServerSettings | null>(null);
let envVars        = $state<EnvVar[]>([]);
let limits         = $state<ServiceLimit[]>([]);
let backendLogs    = $state<LogEntry[]>([]);
let frontendLogs   = $state<LogEntry[]>([]);

// Prompt editor
let promptContent  = $state('');
let promptDirty    = $state(false);
let promptSaving   = $state(false);

// Settings editor draft
let settingsDraft  = $state<Partial<Record<string, string | number | boolean>>>({});

// Env editor drafts (key → new value)
let envDraft       = $state<Record<string, string>>({});

// Revealed secret values
let revealed       = $state<Set<string>>(new Set());

// Toast notifications
let toast          = $state<{ message: string; type: 'success' | 'warning' | 'error' } | null>(null);

let pollTimer: ReturnType<typeof setInterval> | null = null;

// ── Frontend log capture ────────────────────────────────────────────────────

function captureConsoleLogs() {
  const _warn = console.warn.bind(console);
  const _error = console.error.bind(console);
  const _info = console.info.bind(console);

  console.warn = (...args) => {
    addFrontendLog('WARNING', args.map(String).join(' '));
    _warn(...args);
  };
  console.error = (...args) => {
    addFrontendLog('ERROR', args.map(String).join(' '));
    _error(...args);
  };
  console.info = (...args) => {
    addFrontendLog('INFO', args.map(String).join(' '));
    _info(...args);
  };
}

function addFrontendLog(level: LogEntry['level'], message: string) {
  frontendLogs.push({
    timestamp: Date.now() / 1000,
    level,
    message,
    source: 'frontend',
  });
  if (frontendLogs.length > 200) frontendLogs.shift();
}

captureConsoleLogs();

// ── API helpers ─────────────────────────────────────────────────────────────

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T | null> {
  try {
    const res = await fetch(`${API}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`${res.status}: ${text}`);
    }
    return await res.json() as T;
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    addFrontendLog('ERROR', `API ${path} failed: ${msg}`);
    return null;
  }
}

// ── Data fetching ────────────────────────────────────────────────────────────

async function fetchStats() {
  const data = await apiFetch<StatsSnapshot>('/stats');
  if (data) stats = data;
}

async function fetchSettings() {
  const data = await apiFetch<ServerSettings>('/settings');
  if (data) {
    settings = data;
    settingsDraft = {};
  }
}

async function fetchEnv() {
  const data = await apiFetch<EnvVar[]>('/env');
  if (data) {
    envVars = data;
    envDraft = {};
  }
}

async function fetchLimits() {
  const data = await apiFetch<ServiceLimit[]>('/limits');
  if (data) limits = data;
}

async function fetchLogs() {
  const data = await apiFetch<LogEntry[]>('/logs?n=150&level=INFO');
  if (data) backendLogs = data;
}

async function fetchPrompt() {
  const data = await apiFetch<{ content: string }>('/prompt');
  if (data) {
    promptContent = data.content;
    promptDirty = false;
  }
}

async function fetchAll() {
  loading = true;
  error = null;
  await Promise.all([fetchStats(), fetchSettings(), fetchLimits(), fetchLogs()]);
  loading = false;
}

// ── Actions ──────────────────────────────────────────────────────────────────

function showToast(message: string, type: 'success' | 'warning' | 'error' = 'success') {
  toast = { message, type };
  setTimeout(() => { toast = null; }, 4000);
}

/**
 * Save settings — if restart is required, prompt the user first.
 */
async function saveSettings(draft: Record<string, string | number | boolean>) {
  const result = await apiFetch<{
    applied: string[];
    restart_required: string[];
    message: string;
  }>('/settings', {
    method: 'PATCH',
    body: JSON.stringify(draft),
  });

  if (!result) return;

  if (result.restart_required.length > 0) {
    showToast(`${result.message} Restart server to apply: ${result.restart_required.join(', ')}`, 'warning');
  } else {
    showToast(result.message, 'success');
  }

  await fetchSettings();
  settingsDraft = {};
}

/**
 * Save env var(s) — warns if restart is needed.
 */
async function saveEnvVars(updates: Record<string, string>) {
  const confirmed = await confirmIfRestartNeeded(Object.keys(updates));
  if (!confirmed) return;

  const result = await apiFetch<{
    applied: string[];
    restart_required: string[];
    message: string;
  }>('/env', {
    method: 'PATCH',
    body: JSON.stringify(updates),
  });

  if (!result) return;

  if (result.restart_required.length > 0) {
    showToast(`Saved. Restart server for: ${result.restart_required.join(', ')}`, 'warning');
  } else {
    showToast(result.message, 'success');
  }

  await fetchEnv();
  envDraft = {};
}

/**
 * Show a native confirm dialog if any of the keys require a restart.
 * Returns true to proceed, false to cancel.
 */
async function confirmIfRestartNeeded(keys: string[]): Promise<boolean> {
  const RESTART_KEYS = new Set([
    'AGENT_LLM', 'DEFAULT_LLM', 'GROQ_MODEL', 'OPENAI_MODEL',
    'COHERE_MODEL', 'GROQ_API_KEY', 'OPENAI_API_KEY', 'DEEPGRAM_API_KEY',
    'SMALLEST_API_KEY', 'CARTESIA_API_KEY', 'SERVER_PORT', 'SERVER_HOST',
    'MEMORY_ENABLED', 'CHECKPOINT_ENABLED',
  ]);
  const needsRestart = keys.filter(k => RESTART_KEYS.has(k));
  if (needsRestart.length === 0) return true;

  return window.confirm(
    `⚠️ The following settings require a server restart to take effect:\n\n` +
    `  ${needsRestart.join(', ')}\n\n` +
    `The changes will be saved to .env. After saving, restart the server.\n\n` +
    `Continue?`
  );
}

async function savePrompt() {
  promptSaving = true;
  const result = await apiFetch<{ saved: boolean; rebuild_scheduled: boolean; message: string }>('/prompt', {
    method: 'PUT',
    body: JSON.stringify({ content: promptContent }),
  });
  promptSaving = false;

  if (result) {
    promptDirty = false;
    showToast(result.message, result.rebuild_scheduled ? 'success' : 'warning');
  }
}

function toggleReveal(key: string) {
  const next = new Set(revealed);
  if (next.has(key)) next.delete(key);
  else next.add(key);
  revealed = next;
}

function setEnvDraft(key: string, value: string) {
  envDraft = { ...envDraft, [key]: value };
}

function discardEnvDraft() {
  envDraft = {};
  // Re-fetch to restore original values in inputs
  fetchEnv();
}

function setSettingsDraft(key: string, value: string | number | boolean) {
  if (key === '__reset__') {
    settingsDraft = {};
    return;
  }
  settingsDraft = { ...settingsDraft, [key]: value };
}

// ── Polling ──────────────────────────────────────────────────────────────────

function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(async () => {
    await fetchStats();
    await fetchLimits();
    await fetchLogs();
  }, 3000);
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function open(tab?: typeof activeTab) {
  isOpen = true;
  if (tab) activeTab = tab;
  await fetchAll();
  if (activeTab === 'agent') await fetchPrompt();
  if (activeTab === 'env') await fetchEnv();
  startPolling();
}

function close() {
  isOpen = false;
  stopPolling();
}

function switchTab(tab: typeof activeTab) {
  activeTab = tab;
  if (tab === 'agent' && !promptContent) fetchPrompt();
  if (tab === 'env' && envVars.length === 0) fetchEnv();
}

// ── Computed ─────────────────────────────────────────────────────────────────

const allLogs = $derived(
  [...backendLogs, ...frontendLogs]
    .sort((a, b) => a.timestamp - b.timestamp)
    .slice(-200)
);

const criticalServices = $derived(
  limits.filter(l => l.status === 'critical').map(l => l.label)
);

const warningServices = $derived(
  limits.filter(l => l.status === 'warning').map(l => l.label)
);

// ── Export ────────────────────────────────────────────────────────────────────

export const controlPanel = {
  get isOpen()           { return isOpen; },
  get activeTab()        { return activeTab; },
  get loading()          { return loading; },
  get error()            { return error; },
  get stats()            { return stats; },
  get settings()         { return settings; },
  get envVars()          { return envVars; },
  get limits()           { return limits; },
  get allLogs()          { return allLogs; },
  get promptContent()    { return promptContent; },
  get promptDirty()      { return promptDirty; },
  get promptSaving()     { return promptSaving; },
  get settingsDraft()    { return settingsDraft; },
  get envDraft()         { return envDraft; },
  get revealed()         { return revealed; },
  get toast()            { return toast; },
  get criticalServices() { return criticalServices; },
  get warningServices()  { return warningServices; },

  // Actions
  open,
  close,
  switchTab,
  saveSettings,
  saveEnvVars,
  savePrompt,
  toggleReveal,
  setEnvDraft,
  discardEnvDraft,
  setSettingsDraft,

  /** Update the prompt editor content and mark it dirty. */
  setPromptContent(v: string) { promptContent = v; promptDirty = true; },
};
