<script lang="ts">
  /** Animated typing indicator (three bouncing dots). */
  let { visible = false }: { visible: boolean } = $props();
</script>

{#if visible}
  <div class="typing-indicator">
    <span></span>
    <span></span>
    <span></span>
  </div>
{/if}

<style>
  .typing-indicator {
    display: flex;
    gap: 4px;
    padding: 10px 14px;
    align-self: flex-start;
    background: var(--bg-msg-bot);
    border-radius: 14px;
    border-bottom-left-radius: 4px;
    border: 1px solid var(--border);
  }

  .typing-indicator span {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-dim);
    animation: typing-bounce 1.2s infinite;
  }

  .typing-indicator span:nth-child(2) { animation-delay: 0.15s; }
  .typing-indicator span:nth-child(3) { animation-delay: 0.3s; }

  @keyframes typing-bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-6px); }
  }
</style>
