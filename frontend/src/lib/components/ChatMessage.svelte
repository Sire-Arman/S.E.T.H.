<script lang="ts">
  import type { ChatMessage } from '../types';

  /**
   * Single chat message bubble.
   * Phase 4 hook: add `rich` prop for Shiki/KaTeX/Mermaid rendering.
   */
  let { message }: { message: ChatMessage } = $props();
</script>

<div class="msg {message.role}">
  {message.content}
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
    white-space: pre-wrap;
  }

  .msg.user {
    align-self: flex-end;
    background: var(--bg-msg-user);
    color: #fff;
    border-bottom-right-radius: 4px;
  }

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
