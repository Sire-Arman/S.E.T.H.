<script lang="ts">
  import type { ChatMessage } from '../types';
  import { chat } from '../stores/chat.svelte';
  import { purify, getLastArtifacts } from '../utils/purifier';
  import mermaid from 'mermaid';

  /**
   * Single chat message bubble with:
   * - Markdown rendering (via marked)
   * - Shiki syntax highlighting
   * - KaTeX math rendering
   * - Mermaid diagram rendering (via use:action)
   * - Artifact panel integration
   * - Privacy redaction
   */
  let { message }: { message: ChatMessage } = $props();

  const isSystem = $derived(message.role === 'system');

  // Render and detect artifacts
  const rendered = $derived.by(() => {
    const html = purify(message.content);
    const detectedArtifacts = getLastArtifacts();
    return { html, detectedArtifacts };
  });

  // Initialize Mermaid with dark theme
  mermaid.initialize({
    startOnLoad: false,
    theme: 'dark',
    themeVariables: {
      darkMode: true,
      background: '#1a1d27',
      primaryColor: '#6366f1',
      primaryTextColor: '#e4e4e7',
      lineColor: '#71717a',
    },
  });

  /**
   * Fix common LLM-generated Mermaid issues:
   * - Unquoted node labels with special characters (&, |, etc.)
   */
  function sanitizeMermaid(code: string): string {
    // Quote node labels that contain & or other special chars
    // Pattern: match `[Label with & special]` or `(Label with & chars)`
    return code.replace(
      /(\[|(?<!\|)\()([^\]\)]*[&<>][^\]\)]*)(\]|\))/g,
      (_, open, label, close) => {
        const q = open === '[' ? '["' : '("';
        const qc = close === ']' ? '"]' : '")';
        return q + label.trim() + qc;
      }
    );
  }

  /** Svelte action: renders mermaid placeholders after DOM injection. */
  function renderMermaid(node: HTMLElement) {
    const placeholders = node.querySelectorAll('.mermaid-placeholder');
    placeholders.forEach(async (el) => {
      const encoded = el.getAttribute('data-mermaid-code');
      const id = el.getAttribute('data-mermaid-id');
      if (!encoded || !id) return;

      try {
        let code = decodeURIComponent(atob(encoded));
        code = sanitizeMermaid(code);
        const { svg } = await mermaid.render(id, code);
        el.innerHTML = svg;
        el.classList.add('mermaid-rendered');
      } catch {
        // Fallback: show raw code in a styled code block
        const code = decodeURIComponent(atob(encoded));
        const escaped = code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        el.innerHTML = `<div class="mermaid-fallback"><span class="mermaid-label">mermaid (render failed)</span><pre><code>${escaped}</code></pre></div>`;
      }
    });

    // Wire up copy buttons
    node.querySelectorAll('.code-copy-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const encoded = btn.getAttribute('data-code');
        if (!encoded) return;
        const code = decodeURIComponent(atob(encoded));
        navigator.clipboard.writeText(code).then(() => {
          btn.textContent = 'Copied!';
          setTimeout(() => (btn.textContent = 'Copy'), 2000);
        });
      });
    });

    // Wire up "Open in Panel" buttons
    node.querySelectorAll('.artifact-open-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const artId = btn.getAttribute('data-artifact-id');
        if (!artId) return;
        const found = rendered.detectedArtifacts.find((a) => a.id === artId);
        if (found) chat.openArtifact(found);
      });
    });
  }
</script>

<div class="msg {message.role}">
  {#if isSystem}
    {message.content}
  {:else}
    <div class="markdown-body" use:renderMermaid>
      {@html rendered.html}
    </div>
  {/if}
</div>

<style>
  .msg {
    max-width: 82%;
    padding: 10px 14px;
    border-radius: 14px;
    font-size: 13.5px;
    line-height: 1.55;
    animation: msg-in 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    word-wrap: break-word;
  }

  /* ── Markdown body ──────────────────────────────────── */
  .markdown-body :global(p) { margin: 0; }
  .markdown-body :global(p + p) { margin-top: 8px; }
  .markdown-body :global(ul), .markdown-body :global(ol) {
    margin: 8px 0;
    padding-left: 20px;
  }
  .markdown-body :global(li) { margin: 4px 0; }

  /* Inline code */
  .markdown-body :global(code) {
    background: rgba(255, 255, 255, 0.1);
    padding: 2px 5px;
    border-radius: 4px;
    font-family: ui-monospace, 'Cascadia Code', monospace;
    font-size: 0.88em;
  }

  /* Code block wrapper */
  .markdown-body :global(.code-block-wrapper) {
    position: relative;
    margin: 10px 0;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
    background: var(--bg-dark);
  }
  .markdown-body :global(.code-block-wrapper pre) {
    margin: 0;
    border-radius: 0;
    font-size: 12.5px;
    line-height: 1.6;
    padding: 14px 16px;
    overflow-x: auto;
  }
  .markdown-body :global(.code-block-wrapper pre code) {
    background: transparent;
    padding: 0;
    font-size: inherit;
  }

  /* Language badge */
  .markdown-body :global(.code-lang) {
    position: absolute;
    top: 6px;
    left: 10px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-dim);
    opacity: 0.7;
    z-index: 2;
  }

  /* Code toolbar (copy + artifact buttons) */
  .markdown-body :global(.code-toolbar) {
    position: absolute;
    top: 4px;
    right: 6px;
    display: flex;
    gap: 4px;
    z-index: 2;
    opacity: 0;
    transition: opacity 0.2s;
  }
  .markdown-body :global(.code-block-wrapper:hover .code-toolbar) {
    opacity: 1;
  }
  .markdown-body :global(.code-copy-btn),
  .markdown-body :global(.artifact-open-btn) {
    padding: 3px 10px;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    background: rgba(0, 0, 0, 0.6);
    color: var(--text-dim);
    font-size: 10.5px;
    cursor: pointer;
    backdrop-filter: blur(4px);
    font-family: var(--font);
    transition: all 0.15s;
  }
  .markdown-body :global(.code-copy-btn:hover),
  .markdown-body :global(.artifact-open-btn:hover) {
    color: #fff;
    border-color: var(--accent);
  }
  .markdown-body :global(.artifact-open-btn) {
    color: var(--accent);
  }

  /* Mermaid diagrams */
  .markdown-body :global(.mermaid-rendered) {
    display: flex;
    justify-content: center;
    padding: 16px;
    background: var(--bg-dark);
    border-radius: 10px;
    border: 1px solid var(--border);
    margin: 10px 0;
    overflow-x: auto;
  }
  .markdown-body :global(.mermaid-fallback) {
    background: var(--bg-dark);
    border-radius: 10px;
    border: 1px solid var(--border);
    margin: 10px 0;
    overflow: hidden;
  }
  .markdown-body :global(.mermaid-fallback .mermaid-label) {
    display: block;
    padding: 6px 12px;
    font-size: 10px;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .markdown-body :global(.mermaid-fallback pre) {
    margin: 0;
    padding: 12px 16px;
    font-size: 12px;
    line-height: 1.6;
  }
  .markdown-body :global(.mermaid-fallback code) {
    background: transparent;
    padding: 0;
  }

  /* Shiki fallback */
  .markdown-body :global(.shiki-fallback) {
    background: var(--bg-dark);
    padding: 14px 16px;
    font-family: ui-monospace, monospace;
    font-size: 12.5px;
    line-height: 1.6;
  }

  /* Text formatting */
  .markdown-body :global(strong) { font-weight: 700; color: #fff; }
  .markdown-body :global(em) { font-style: italic; opacity: 0.9; }
  .markdown-body :global(a) { color: var(--accent); text-decoration: underline; }
  .markdown-body :global(blockquote) {
    border-left: 3px solid var(--accent);
    padding-left: 12px;
    margin: 8px 0;
    color: var(--text-dim);
  }

  /* ── Message role styling ───────────────────────── */

  .msg.user {
    align-self: flex-end;
    background: var(--bg-msg-user);
    color: #fff;
    border-bottom-right-radius: 4px;
  }
  .msg.user .markdown-body :global(strong) { color: #fff; }

  .msg.bot {
    align-self: flex-start;
    background: var(--bg-msg-bot);
    color: var(--text);
    border-bottom-left-radius: 4px;
    border: 1px solid var(--border);
  }

  .msg.system {
    align-self: center;
    background: transparent;
    color: var(--text-dim);
    font-size: 11px;
    padding: 4px 8px;
  }

  @keyframes msg-in {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
</style>
