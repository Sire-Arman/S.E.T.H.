<script lang="ts">
  import type { ConnectionState } from '../types';

  /**
   * Header bar with avatar, status indicator, and action buttons.
   */
  let {
    connectionState,
    wakeWordEnabled = false,
    wakeWordSupported = false,
    onToggleWake,
  }: {
    connectionState: ConnectionState;
    wakeWordEnabled: boolean;
    wakeWordSupported: boolean;
    onToggleWake: () => void;
  } = $props();

  const statusConfig: Record<ConnectionState, { label: string; dotClass: string }> = {
    disconnected: { label: 'Disconnected', dotClass: '' },
    connecting:   { label: 'Connecting...', dotClass: 'processing' },
    idle:         { label: 'Ready', dotClass: 'connected' },
    listening:    { label: 'Listening...', dotClass: 'listening' },
    processing:   { label: 'Thinking...', dotClass: 'processing' },
    speaking:     { label: 'Speaking...', dotClass: 'speaking' },
  };

  const status = $derived(statusConfig[connectionState]);
</script>

<header class="status-bar">
  <div class="avatar">S</div>

  <div class="info">
    <div class="name">SETH</div>
    <div class="status-text">
      <span class="dot {status.dotClass}"></span>
      <span>{status.label}</span>
    </div>
  </div>

  <div class="actions">
    {#if wakeWordSupported}
      <button
        class="header-btn"
        class:wake-active={wakeWordEnabled}
        title={wakeWordEnabled ? 'Wake word ON — say "HEY"' : 'Wake word OFF'}
        onclick={onToggleWake}
      >
        <!-- Wi-Fi / signal icon -->
        <svg viewBox="0 0 24 24"><path d="M1 9l2 2c4.97-4.97 13.03-4.97 18 0l2-2C16.93 2.93 7.08 2.93 1 9zm8 8l3 3 3-3a4.24 4.24 0 0 0-6 0zm-4-4l2 2a7.07 7.07 0 0 1 10 0l2-2C15.14 9.14 8.87 9.14 5 13z"/></svg>
      </button>
    {/if}
  </div>
</header>

<style>
  .status-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 20px;
    background: rgba(255, 255, 255, 0.02);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }

  .avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 17px;
    font-weight: 700;
    color: #fff;
    flex-shrink: 0;
  }

  .info { flex: 1; }
  .info .name { font-weight: 600; font-size: 15px; }

  .status-text {
    font-size: 12px;
    color: var(--text-dim);
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 2px;
  }

  .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--text-dim);
    transition: background 0.3s;
    flex-shrink: 0;
  }
  .dot.connected  { background: var(--success); }
  .dot.listening  { background: var(--danger); animation: blink 0.8s infinite; }
  .dot.processing { background: #f59e0b; animation: blink 1s infinite; }
  .dot.speaking   { background: var(--accent); animation: blink 0.6s infinite; }

  @keyframes blink {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.3; }
  }

  .actions { display: flex; gap: 4px; }

  .header-btn {
    width: 34px;
    height: 34px;
    border-radius: 8px;
    border: none;
    background: transparent;
    color: var(--text-dim);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.2s, color 0.2s;
  }
  .header-btn:hover { background: rgba(255, 255, 255, 0.06); color: var(--text); }
  .header-btn svg { width: 17px; height: 17px; fill: currentColor; }
  .header-btn.wake-active { color: var(--success); }
</style>
