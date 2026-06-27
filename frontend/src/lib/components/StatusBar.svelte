<script lang="ts">
  import type { ConnectionState } from '../types';
  import { controlPanel as cp } from '../stores/controlPanel.svelte';

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

    <!-- Dev Mode toggle -->
    <button
      class="dev-toggle"
      class:dev-active={cp.isOpen}
      title={cp.isOpen ? 'Close Dev Panel' : 'Open Dev Panel'}
      onclick={() => cp.isOpen ? cp.close() : cp.open()}
    >
      <span class="dev-led" class:led-critical={cp.criticalServices.length > 0} class:led-warn={cp.warningServices.length > 0 && cp.criticalServices.length === 0}></span>
      DEV
    </button>
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

  /* Dev Mode toggle */
  .dev-toggle {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px 4px 8px;
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    background: rgba(255, 255, 255, 0.04);
    color: var(--text-dim);
    font-family: var(--font);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.8px;
    cursor: pointer;
    transition: all 0.2s;
  }
  .dev-toggle:hover {
    border-color: rgba(99, 102, 241, 0.4);
    color: var(--text);
    background: rgba(99, 102, 241, 0.08);
  }
  .dev-toggle.dev-active {
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(99, 102, 241, 0.12);
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.2);
  }

  .dev-led {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-dim);
    transition: background 0.2s;
    flex-shrink: 0;
  }
  .dev-toggle.dev-active .dev-led { background: var(--accent); animation: blink 1.5s infinite; }
  .dev-led.led-critical { background: var(--danger) !important; animation: blink 0.6s infinite !important; }
  .dev-led.led-warn     { background: #f59e0b !important; }
</style>
