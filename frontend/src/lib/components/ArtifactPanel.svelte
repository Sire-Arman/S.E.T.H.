<script lang="ts">
  import type { Artifact } from '../types';
  import { highlightCode } from '../utils/shiki';

  /**
   * Sliding side panel for viewing code artifacts.
   * Shows Shiki-highlighted code with copy button and language badge.
   */
  let {
    artifact,
    onClose,
  }: {
    artifact: Artifact | null;
    onClose: () => void;
  } = $props();

  let copyLabel = $state('Copy');

  function handleCopy() {
    if (!artifact) return;
    navigator.clipboard.writeText(artifact.code).then(() => {
      copyLabel = 'Copied!';
      setTimeout(() => (copyLabel = 'Copy'), 2000);
    });
  }

  const highlightedHtml = $derived(
    artifact ? highlightCode(artifact.code, artifact.language) : '',
  );

  const lineCount = $derived(
    artifact ? artifact.code.split('\n').length : 0,
  );
</script>

{#if artifact}
  <!-- Backdrop -->
  <button class="artifact-backdrop" onclick={onClose} aria-label="Close panel"></button>

  <aside class="artifact-panel">
    <header class="panel-header">
      <div class="panel-info">
        <span class="lang-badge">{artifact.language}</span>
        <span class="line-count">{lineCount} lines</span>
      </div>
      <div class="panel-actions">
        <button class="panel-btn" onclick={handleCopy}>{copyLabel}</button>
        <button class="panel-btn close-btn" onclick={onClose}>✕</button>
      </div>
    </header>

    <div class="panel-code">
      {@html highlightedHtml}
    </div>
  </aside>
{/if}

<style>
  .artifact-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.5);
    z-index: 90;
    border: none;
    cursor: default;
    backdrop-filter: blur(2px);
    animation: fade-in 0.2s ease;
  }

  .artifact-panel {
    position: fixed;
    top: 0;
    right: 0;
    width: min(50vw, 720px);
    height: 100dvh;
    background: var(--bg-panel);
    border-left: 1px solid var(--border);
    z-index: 100;
    display: flex;
    flex-direction: column;
    animation: slide-in 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: -8px 0 40px rgba(0, 0, 0, 0.4);
  }

  @media (max-width: 768px) {
    .artifact-panel {
      width: 100vw;
    }
  }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 18px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    background: rgba(0, 0, 0, 0.2);
  }

  .panel-info {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .lang-badge {
    background: var(--accent);
    color: #fff;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .line-count {
    font-size: 12px;
    color: var(--text-dim);
  }

  .panel-actions {
    display: flex;
    gap: 6px;
  }

  .panel-btn {
    padding: 6px 14px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-dim);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
    font-family: var(--font);
  }
  .panel-btn:hover {
    background: rgba(255, 255, 255, 0.06);
    color: var(--text);
  }
  .close-btn {
    border: none;
    font-size: 15px;
    width: 32px;
    padding: 0;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .panel-code {
    flex: 1;
    overflow: auto;
    padding: 16px;
  }

  .panel-code :global(pre) {
    margin: 0;
    border-radius: 8px;
    font-size: 13px;
    line-height: 1.6;
  }

  @keyframes slide-in {
    from { transform: translateX(100%); }
    to   { transform: translateX(0); }
  }

  @keyframes fade-in {
    from { opacity: 0; }
    to   { opacity: 1; }
  }
</style>
