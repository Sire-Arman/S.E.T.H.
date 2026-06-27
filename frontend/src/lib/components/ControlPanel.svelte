<script lang="ts">
  import { controlPanel as cp } from '../stores/controlPanel.svelte';

  // ── Helpers ────────────────────────────────────────────────────────────
  function fmt(n: number): string {
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
    if (n >= 1_000)     return (n / 1_000).toFixed(1) + 'K';
    return String(n);
  }

  function fmtUptime(secs: number): string {
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    const s = Math.floor(secs % 60);
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  }

  function fmtTime(ts: number): string {
    return new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  function gaugeColor(pct: number): string {
    if (pct >= 85) return 'var(--danger)';
    if (pct >= 60) return '#f59e0b';
    return 'var(--success)';
  }

  function levelColor(level: string): string {
    switch (level) {
      case 'ERROR':   return '#ef4444';
      case 'WARNING': return '#f59e0b';
      case 'DEBUG':   return '#6b7280';
      default:        return '#60a5fa';
    }
  }

  const TABS = [
    { id: 'overview', icon: '◎',  label: 'Overview' },
    { id: 'agent',    icon: '⚙',  label: 'Agent' },
    { id: 'tts',      icon: '◈',  label: 'TTS' },
    { id: 'limits',   icon: '▦',  label: 'API Limits' },
    { id: 'env',      icon: '⊞',  label: 'Environment' },
    { id: 'logs',     icon: '≡',   label: 'Logs' },
  ] as const;
</script>

{#if cp.isOpen}
  <!-- Right-side Dev Panel (no backdrop — chat shifts left via App layout) -->
  <aside class="control-panel">

    <!-- Header -->
    <header class="cp-header">
      <div class="cp-title">
        <span class="cp-icon">◈</span>
        <span>Dev Panel</span>
      </div>
      {#if cp.criticalServices.length > 0}
        <span class="cp-alert critical">⚠ {cp.criticalServices.join(', ')}</span>
      {:else if cp.warningServices.length > 0}
        <span class="cp-alert warning">⚠ {cp.warningServices.join(', ')}</span>
      {/if}
    </header>

    <!-- Tab bar -->
    <nav class="cp-tabs">
      {#each TABS as tab}
        <button
          class="cp-tab"
          class:active={cp.activeTab === tab.id}
          onclick={() => cp.switchTab(tab.id)}
          title={tab.label}
        >
          <span class="tab-icon">{tab.icon}</span>
          <span class="tab-label">{tab.label}</span>
        </button>
      {/each}
    </nav>

    <!-- Body -->
    <div class="cp-body">

      <!-- ── OVERVIEW ─────────────────────────────────────────────── -->
      {#if cp.activeTab === 'overview'}
        <section class="cp-section">
          {#if cp.loading && !cp.stats}
            <div class="cp-loading">Loading…</div>
          {:else if cp.stats}
            <!-- Uptime + sessions -->
            <div class="stat-grid">
              <div class="stat-card">
                <div class="stat-value">{fmtUptime(cp.stats.uptime_seconds)}</div>
                <div class="stat-label">Uptime</div>
              </div>
              <div class="stat-card">
                <div class="stat-value">{cp.stats.session_count}</div>
                <div class="stat-label">Sessions</div>
              </div>
              <div class="stat-card">
                <div class="stat-value">{cp.stats.tokens.requests}</div>
                <div class="stat-label">LLM Calls</div>
              </div>
            </div>

            <!-- Token bar -->
            <div class="cp-card mt">
              <div class="card-header">Token Usage</div>
              <div class="token-row">
                <span class="token-label">Input</span>
                <span class="token-val">{fmt(cp.stats.tokens.input)}</span>
              </div>
              <div class="token-row">
                <span class="token-label">Output</span>
                <span class="token-val">{fmt(cp.stats.tokens.output)}</span>
              </div>
              <div class="token-divider"></div>
              <div class="token-row total">
                <span class="token-label">Total</span>
                <span class="token-val">{fmt(cp.stats.tokens.total)}</span>
              </div>
            </div>

            <!-- Per-service mini cards -->
            <div class="card-header mt-sm">Request Rate (RPM)</div>
            <div class="service-grid">
              {#each Object.entries(cp.stats.services) as [name, svc]}
                {#if svc.requests_total > 0 || name === 'groq' || name === 'deepgram'}
                  <div class="svc-card">
                    <div class="svc-top">
                      <span class="svc-name">{svc.label}</span>
                      <span class="svc-rpm" style="color:{gaugeColor(svc.rpm_pct)}">{svc.rpm} rpm</span>
                    </div>
                    <div class="svc-bar-bg">
                      <div class="svc-bar-fill" style="width:{svc.rpm_pct}%;background:{gaugeColor(svc.rpm_pct)}"></div>
                    </div>
                    <div class="svc-total">{svc.requests_total} total reqs</div>
                  </div>
                {/if}
              {/each}
            </div>
          {/if}
        </section>

      <!-- ── AGENT ─────────────────────────────────────────────────── -->
      {:else if cp.activeTab === 'agent'}
        <section class="cp-section">
          {#if cp.settings}
            <div class="cp-card">
              <div class="card-header">LLM Configuration</div>
              <div class="field-row">
                <label class="field-label">Provider</label>
                <select class="field-input" value={cp.settings.agent.llm}
                  onchange={(e) => cp.setSettingsDraft('llm', (e.target as HTMLSelectElement).value)}>
                  {#each ['groq', 'openai', 'cohere', 'anthropic', 'ollama', 'gemini'] as p}
                    <option value={p} selected={cp.settings.agent.llm === p}>{p}</option>
                  {/each}
                </select>
                <span class="restart-badge">restart</span>
              </div>
              <div class="field-row">
                <label class="field-label">Temperature</label>
                <input type="number" class="field-input sm" min="0" max="2" step="0.1"
                  value={cp.settings.agent.temperature}
                  oninput={(e) => cp.setSettingsDraft('temperature', parseFloat((e.target as HTMLInputElement).value))} />
              </div>
              <div class="field-row">
                <label class="field-label">Max Tokens</label>
                <input type="number" class="field-input sm" min="50" max="4000" step="50"
                  value={cp.settings.agent.max_tokens}
                  oninput={(e) => cp.setSettingsDraft('max_tokens', parseInt((e.target as HTMLInputElement).value))} />
              </div>
              <div class="field-row">
                <label class="field-label">Groq Model</label>
                <select class="field-input"
                  onchange={(e) => cp.setSettingsDraft('groq_model', (e.target as HTMLSelectElement).value)}>
                  {#each ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'llama-3.2-90b-vision-preview', 'mixtral-8x7b-32768'] as m}
                    <option value={m} selected={cp.settings.agent.groq_model === m}>{m}</option>
                  {/each}
                </select>
                <span class="restart-badge">restart</span>
              </div>
              <div class="field-row">
                <label class="field-label">Memory Top-K</label>
                <input type="number" class="field-input sm" min="1" max="20"
                  value={cp.settings.agent.memory_top_k}
                  oninput={(e) => cp.setSettingsDraft('memory_top_k', parseInt((e.target as HTMLInputElement).value))} />
              </div>

              {#if Object.keys(cp.settingsDraft).length > 0}
                <div class="action-row">
                  <button class="btn-primary" onclick={() => cp.saveSettings(cp.settingsDraft as any)}>Save Changes</button>
                  <button class="btn-ghost" onclick={() => cp.setSettingsDraft('__reset__', '__reset__')}>Reset</button>
                </div>
              {/if}
            </div>

            <!-- System Prompt Editor -->
            <div class="cp-card mt">
              <div class="card-header-row">
                <span class="card-header">System Prompt</span>
                {#if cp.promptDirty}
                  <span class="dirty-badge">unsaved</span>
                {/if}
              </div>
              <textarea
                class="prompt-editor"
                rows="10"
                value={cp.promptContent}
                oninput={(e) => cp.setPromptContent((e.target as HTMLTextAreaElement).value)}
                placeholder="Loading system prompt…"
                spellcheck="false"
              ></textarea>
              {#if cp.promptDirty}
                <div class="action-row">
                  <button class="btn-primary" onclick={() => cp.savePrompt()} disabled={cp.promptSaving}>
                    {cp.promptSaving ? 'Saving…' : 'Save Prompt'}
                  </button>
                </div>
              {/if}
            </div>
          {:else}
            <div class="cp-loading">Loading…</div>
          {/if}
        </section>

      <!-- ── TTS ───────────────────────────────────────────────────── -->
      {:else if cp.activeTab === 'tts'}
        <section class="cp-section">
          {#if cp.settings}
            <div class="cp-card">
              <div class="card-header">TTS Provider</div>
              <div class="field-row">
                <label class="field-label">Provider</label>
                <select class="field-input"
                  onchange={(e) => cp.setSettingsDraft('tts_provider', (e.target as HTMLSelectElement).value)}>
                  {#each ['smallest', 'cartesia', 'kokoro'] as p}
                    <option value={p} selected={cp.settings.tts.provider === p}>{p}</option>
                  {/each}
                </select>
              </div>
            </div>

            <!-- Smallest.ai -->
            <div class="cp-card mt">
              <div class="card-header">Smallest.ai Settings</div>
              <div class="field-row">
                <label class="field-label">Voice</label>
                <select class="field-input"
                  onchange={(e) => cp.setSettingsDraft('smallest_voice', (e.target as HTMLSelectElement).value)}>
                  {#each ['olivia', 'rachel', 'lauren', 'hannah', 'chloe', 'kaitlyn', 'meher', 'sophie', 'savannah'] as v}
                    <option value={v} selected={cp.settings.tts.smallest_voice === v}>{v}</option>
                  {/each}
                </select>
                <span class="hot-badge">hot</span>
              </div>
              <div class="field-row">
                <label class="field-label">Model</label>
                <select class="field-input"
                  onchange={(e) => cp.setSettingsDraft('smallest_model', (e.target as HTMLSelectElement).value)}>
                  {#each ['lightning_v3.1', 'lightning_v3.1_pro'] as m}
                    <option value={m} selected={cp.settings.tts.smallest_model === m}>{m}</option>
                  {/each}
                </select>
                <span class="hot-badge">hot</span>
              </div>
            </div>

            <!-- Cartesia -->
            <div class="cp-card mt">
              <div class="card-header">Cartesia Settings</div>
              <div class="field-row">
                <label class="field-label">Voice ID</label>
                <input class="field-input" type="text" value={cp.settings.tts.cartesia_voice}
                  oninput={(e) => cp.setSettingsDraft('cartesia_voice', (e.target as HTMLInputElement).value)} />
                <span class="hot-badge">hot</span>
              </div>
              <div class="field-row">
                <label class="field-label">Model</label>
                <select class="field-input"
                  onchange={(e) => cp.setSettingsDraft('cartesia_model', (e.target as HTMLSelectElement).value)}>
                  {#each ['sonic-3', 'sonic-2', 'sonic'] as m}
                    <option value={m} selected={cp.settings.tts.cartesia_model === m}>{m}</option>
                  {/each}
                </select>
                <span class="hot-badge">hot</span>
              </div>
            </div>

            {#if Object.keys(cp.settingsDraft).length > 0}
              <div class="action-row mt">
                <button class="btn-primary" onclick={() => cp.saveSettings(cp.settingsDraft as any)}>Apply TTS Settings</button>
              </div>
            {/if}
          {:else}
            <div class="cp-loading">Loading…</div>
          {/if}
        </section>

      <!-- ── API LIMITS ────────────────────────────────────────────── -->
      {:else if cp.activeTab === 'limits'}
        <section class="cp-section">
          {#if cp.limits.length === 0}
            <div class="cp-loading">Loading…</div>
          {:else}
            {#each cp.limits as svc}
              <div class="cp-card mt-sm">
                <div class="limit-header">
                  <span class="limit-name">{svc.label}</span>
                  <span class="limit-status" class:ok={svc.status==='ok'} class:warn={svc.status==='warning'} class:crit={svc.status==='critical'}>
                    {svc.status === 'ok' ? '✓ OK' : svc.status === 'warning' ? '⚠ Warning' : '✕ Critical'}
                  </span>
                </div>
                {#each svc.gauges as g}
                  <div class="gauge-row">
                    <div class="gauge-label-row">
                      <span class="gauge-label">{g.label}</span>
                      <span class="gauge-val">{fmt(g.used)} / {fmt(g.limit)}</span>
                    </div>
                    <div class="gauge-bg">
                      <div class="gauge-fill" style="width:{g.pct}%;background:{gaugeColor(g.pct)}">
                        <span class="gauge-pct">{g.pct.toFixed(0)}%</span>
                      </div>
                    </div>
                  </div>
                {/each}
              </div>
            {/each}

            <div class="cp-note mt">
              ⓘ Limits are estimated from free-tier quotas. Actual usage may differ.
              Data resets on server restart.
            </div>
          {/if}
        </section>

      <!-- ── ENVIRONMENT ───────────────────────────────────────────── -->
      {:else if cp.activeTab === 'env'}
        <section class="cp-section">
          {#if cp.envVars.length === 0}
            <div class="cp-loading">Loading env vars…</div>
          {:else}
            <div class="env-list">
              {#each cp.envVars as ev}
                <div class="env-row">
                  <div class="env-key-row">
                    <span class="env-key">{ev.key}</span>
                    <div class="env-badges">
                      {#if ev.restart_required}<span class="restart-badge">restart</span>{/if}
                      {#if ev.hot_reload}<span class="hot-badge">hot</span>{/if}
                    </div>
                  </div>
                  <div class="env-val-row">
                    <input
                      class="env-input"
                      type={ev.is_secret && !cp.revealed.has(ev.key) ? 'password' : 'text'}
                      value={cp.envDraft[ev.key] ?? ev.value}
                      oninput={(e) => cp.setEnvDraft(ev.key, (e.target as HTMLInputElement).value)}
                    />
                    {#if ev.is_secret}
                      <button class="reveal-btn" onclick={() => cp.toggleReveal(ev.key)}
                        title={cp.revealed.has(ev.key) ? 'Hide' : 'Reveal'}>
                        {cp.revealed.has(ev.key) ? '◑' : '◐'}
                      </button>
                    {/if}
                  </div>
                </div>
              {/each}
            </div>

            {#if Object.keys(cp.envDraft).length > 0}
              <div class="action-row mt sticky-action">
                <button class="btn-primary" onclick={() => cp.saveEnvVars(cp.envDraft)}>
                  Save {Object.keys(cp.envDraft).length} Change{Object.keys(cp.envDraft).length > 1 ? 's' : ''}
                </button>
                <button class="btn-ghost" onclick={() => cp.discardEnvDraft()}>Discard</button>
              </div>
            {/if}
          {/if}
        </section>

      <!-- ── LOGS ──────────────────────────────────────────────────── -->
      {:else if cp.activeTab === 'logs'}
        <section class="cp-section logs-section">
          <div class="logs-toolbar">
            <span class="logs-count">{cp.allLogs.length} entries</span>
            <div class="log-legend">
              <span style="color:#60a5fa">INFO</span>
              <span style="color:#f59e0b">WARN</span>
              <span style="color:#ef4444">ERR</span>
              <span style="color:#6b7280">DBG</span>
            </div>
          </div>
          <div class="log-list">
            {#each cp.allLogs as entry}
              <div class="log-entry" class:log-error={entry.level === 'ERROR'} class:log-warn={entry.level === 'WARNING'}>
                <span class="log-time">{fmtTime(entry.timestamp)}</span>
                <span class="log-level" style="color:{levelColor(entry.level)}">{entry.level.slice(0,3)}</span>
                <span class="log-src">[{entry.source}]</span>
                <span class="log-msg">{entry.message}</span>
              </div>
            {/each}
          </div>
        </section>
      {/if}

    </div><!-- /cp-body -->

    <!-- Toast -->
    {#if cp.toast}
      <div class="cp-toast" class:toast-warn={cp.toast.type === 'warning'} class:toast-err={cp.toast.type === 'error'}>
        {cp.toast.message}
      </div>
    {/if}

  </aside>
{/if}

<style>
  /* ── Backdrop ─────────────────────────────────────────────────────── */
  /* Removed — panel is a persistent sidebar, no overlay */

  /* ── Panel ────────────────────────────────────────────────────────── */
  .control-panel {
    position: fixed;
    top: 0;
    right: 0;
    width: 380px;
    height: 100dvh;
    background: #13151f;
    border-left: 1px solid rgba(255,255,255,0.07);
    z-index: 50;  /* below artifact panel (100) so it doesn't cover it */
    display: flex;
    flex-direction: column;
    animation: slide-in-right 0.28s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: -8px 0 40px rgba(0, 0, 0, 0.5);
  }

  @media (max-width: 600px) {
    .control-panel { width: 100vw; }
  }

  /* ── Header ───────────────────────────────────────────────────────── */
  .cp-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.02);
    flex-shrink: 0;
  }

  .cp-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
    font-size: 14px;
    flex: 1;
  }

  .cp-icon { font-size: 16px; color: var(--accent); }

  .cp-alert {
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 600;
  }
  .cp-alert.critical { background: rgba(239,68,68,0.18); color: #ef4444; }
  .cp-alert.warning  { background: rgba(245,158,11,0.18); color: #f59e0b; }


  /* ── Tab Bar ──────────────────────────────────────────────────────── */
  .cp-tabs {
    display: flex;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    flex-shrink: 0;
    overflow-x: auto;
    scrollbar-width: none;
    background: rgba(0,0,0,0.2);
  }
  .cp-tabs::-webkit-scrollbar { display: none; }

  .cp-tab {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    flex: 1;
    min-width: 54px;
    padding: 8px 4px;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--text-dim);
    cursor: pointer;
    font-family: var(--font);
    font-size: 10px;
    transition: color 0.15s, border-color 0.15s;
  }
  .cp-tab:hover { color: var(--text); }
  .cp-tab.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }

  .tab-icon { font-size: 15px; line-height: 1; }
  .tab-label { font-size: 9px; font-weight: 500; letter-spacing: 0.3px; }

  /* ── Body ─────────────────────────────────────────────────────────── */
  .cp-body {
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
    scrollbar-width: thin;
  }

  .cp-section {
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 0;
  }

  .cp-loading {
    text-align: center;
    color: var(--text-dim);
    font-size: 13px;
    padding: 40px 0;
  }

  /* ── Cards ────────────────────────────────────────────────────────── */
  .cp-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 12px;
  }
  .mt { margin-top: 10px; }
  .mt-sm { margin-top: 6px; }

  .card-header {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--text-dim);
    margin-bottom: 10px;
  }
  .card-header-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }

  /* ── Stats grid ───────────────────────────────────────────────────── */
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }

  .stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 12px 10px;
    text-align: center;
  }

  .stat-value {
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
    line-height: 1.2;
  }

  .stat-label {
    font-size: 10px;
    color: var(--text-dim);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  /* ── Token rows ───────────────────────────────────────────────────── */
  .token-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    font-size: 13px;
  }
  .token-label { color: var(--text-dim); }
  .token-val   { font-weight: 600; font-variant-numeric: tabular-nums; }
  .token-divider { height: 1px; background: rgba(255,255,255,0.05); margin: 4px 0; }
  .token-row.total .token-label { color: var(--text); }
  .token-row.total .token-val   { color: var(--accent); font-size: 15px; }

  /* ── Service grid ─────────────────────────────────────────────────── */
  .service-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 6px;
    margin-top: 6px;
  }

  .svc-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 8px 10px;
  }

  .svc-top {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    margin-bottom: 6px;
  }
  .svc-name { color: var(--text-dim); font-weight: 500; }
  .svc-rpm  { font-weight: 700; font-variant-numeric: tabular-nums; }

  .svc-bar-bg { height: 3px; background: rgba(255,255,255,0.08); border-radius: 2px; overflow: hidden; }
  .svc-bar-fill { height: 100%; border-radius: 2px; transition: width 0.4s ease; }

  .svc-total { font-size: 10px; color: var(--text-dim); margin-top: 5px; }

  /* ── Fields ───────────────────────────────────────────────────────── */
  .field-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }

  .field-label {
    font-size: 12px;
    color: var(--text-dim);
    min-width: 90px;
    flex-shrink: 0;
  }

  .field-input {
    flex: 1;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    color: var(--text);
    font-family: var(--font);
    font-size: 12px;
    padding: 5px 8px;
    min-width: 0;
    transition: border-color 0.15s;
  }
  .field-input:focus { outline: none; border-color: var(--accent); }
  .field-input.sm { max-width: 80px; }

  select.field-input {
    cursor: pointer;
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%2371717a'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 8px center;
    padding-right: 22px;
  }

  /* ── Badges ───────────────────────────────────────────────────────── */
  .restart-badge {
    font-size: 9px;
    padding: 2px 5px;
    border-radius: 4px;
    background: rgba(239,68,68,0.15);
    color: #ef4444;
    font-weight: 600;
    flex-shrink: 0;
  }

  .hot-badge {
    font-size: 9px;
    padding: 2px 5px;
    border-radius: 4px;
    background: rgba(34,197,94,0.15);
    color: #22c55e;
    font-weight: 600;
    flex-shrink: 0;
  }

  .dirty-badge {
    font-size: 9px;
    padding: 2px 6px;
    border-radius: 4px;
    background: rgba(245,158,11,0.15);
    color: #f59e0b;
    font-weight: 600;
  }

  /* ── Prompt editor ────────────────────────────────────────────────── */
  .prompt-editor {
    width: 100%;
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    color: var(--text);
    font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
    font-size: 11.5px;
    line-height: 1.6;
    padding: 10px;
    resize: vertical;
    transition: border-color 0.15s;
  }
  .prompt-editor:focus { outline: none; border-color: var(--accent); }

  /* ── Action row ───────────────────────────────────────────────────── */
  .action-row {
    display: flex;
    gap: 8px;
    margin-top: 12px;
  }

  .btn-primary {
    padding: 7px 16px;
    border-radius: 8px;
    border: none;
    background: var(--accent);
    color: #fff;
    font-family: var(--font);
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.15s;
  }
  .btn-primary:hover { opacity: 0.85; }
  .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }

  .btn-ghost {
    padding: 7px 14px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.08);
    background: transparent;
    color: var(--text-dim);
    font-family: var(--font);
    font-size: 12px;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
  }
  .btn-ghost:hover { background: rgba(255,255,255,0.05); color: var(--text); }

  /* ── API Limits ───────────────────────────────────────────────────── */
  .limit-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }
  .limit-name { font-size: 13px; font-weight: 600; }
  .limit-status { font-size: 11px; font-weight: 600; }
  .limit-status.ok   { color: var(--success); }
  .limit-status.warn { color: #f59e0b; }
  .limit-status.crit { color: var(--danger); }

  .gauge-row { margin-bottom: 8px; }
  .gauge-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    margin-bottom: 4px;
  }
  .gauge-label { color: var(--text-dim); }
  .gauge-val   { color: var(--text); font-variant-numeric: tabular-nums; }

  .gauge-bg {
    height: 6px;
    background: rgba(255,255,255,0.07);
    border-radius: 3px;
    overflow: hidden;
  }
  .gauge-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
    position: relative;
    min-width: 2px;
  }
  .gauge-pct {
    display: none; /* shown only when > 10% via JS if needed */
  }

  .cp-note {
    font-size: 11px;
    color: var(--text-dim);
    line-height: 1.5;
    padding: 10px 12px;
    background: rgba(255,255,255,0.02);
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.04);
    margin-top: 10px;
  }

  /* ── Env list ─────────────────────────────────────────────────────── */
  .env-list {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .env-row {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 8px;
    padding: 8px 10px;
  }

  .env-key-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 5px;
  }
  .env-key {
    font-size: 11px;
    font-family: 'Menlo', monospace;
    color: var(--accent);
    font-weight: 500;
  }
  .env-badges { display: flex; gap: 4px; margin-left: auto; }

  .env-val-row {
    display: flex;
    gap: 6px;
    align-items: center;
  }

  .env-input {
    flex: 1;
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 6px;
    color: var(--text);
    font-family: 'Menlo', monospace;
    font-size: 11px;
    padding: 4px 8px;
    min-width: 0;
    transition: border-color 0.15s;
  }
  .env-input:focus { outline: none; border-color: var(--accent); }

  .reveal-btn {
    background: transparent;
    border: none;
    color: var(--text-dim);
    cursor: pointer;
    font-size: 16px;
    width: 24px;
    flex-shrink: 0;
    transition: color 0.15s;
  }
  .reveal-btn:hover { color: var(--accent); }

  .sticky-action {
    position: sticky;
    bottom: 0;
    background: #13151f;
    padding: 10px 0;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 8px;
  }

  /* ── Logs ─────────────────────────────────────────────────────────── */
  .logs-section {
    padding: 0 !important;
    height: 100%;
    overflow: hidden;
  }

  .logs-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    background: rgba(0,0,0,0.2);
    flex-shrink: 0;
  }

  .logs-count { font-size: 11px; color: var(--text-dim); }

  .log-legend {
    display: flex;
    gap: 10px;
    font-size: 10px;
    font-weight: 600;
  }

  .log-list {
    flex: 1;
    overflow-y: auto;
    font-family: 'Menlo', 'Monaco', monospace;
    font-size: 10.5px;
    line-height: 1.5;
    padding: 4px 0;
    scrollbar-width: thin;
    height: calc(100dvh - 130px);
  }

  .log-entry {
    display: flex;
    gap: 6px;
    align-items: flex-start;
    padding: 2px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.02);
    transition: background 0.1s;
  }
  .log-entry:hover { background: rgba(255,255,255,0.025); }
  .log-entry.log-error { background: rgba(239,68,68,0.04); }
  .log-entry.log-warn  { background: rgba(245,158,11,0.04); }

  .log-time  { color: #4b5563; flex-shrink: 0; }
  .log-level { font-weight: 700; flex-shrink: 0; width: 28px; }
  .log-src   { color: #6b7280; flex-shrink: 0; max-width: 70px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .log-msg   { color: var(--text); word-break: break-word; flex: 1; }

  /* ── Toast ────────────────────────────────────────────────────────── */
  .cp-toast {
    position: absolute;
    bottom: 20px;
    left: 14px;
    right: 14px;
    padding: 10px 14px;
    border-radius: 10px;
    font-size: 12px;
    font-weight: 500;
    background: rgba(34, 197, 94, 0.18);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.25);
    animation: toast-in 0.2s ease;
    z-index: 10;
    line-height: 1.5;
  }
  .cp-toast.toast-warn {
    background: rgba(245, 158, 11, 0.18);
    color: #f59e0b;
    border-color: rgba(245, 158, 11, 0.25);
  }
  .cp-toast.toast-err {
    background: rgba(239, 68, 68, 0.18);
    color: #ef4444;
    border-color: rgba(239, 68, 68, 0.25);
  }

  /* ── Animations ───────────────────────────────────────────────────── */
  @keyframes slide-in-right {
    from { transform: translateX(100%); }
    to   { transform: translateX(0); }
  }

  @keyframes fade-in {
    from { opacity: 0; }
    to   { opacity: 1; }
  }

  @keyframes toast-in {
    from { transform: translateY(10px); opacity: 0; }
    to   { transform: translateY(0); opacity: 1; }
  }
</style>
