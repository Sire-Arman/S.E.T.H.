<script lang="ts">
  import type { ChatMessage as ChatMessageType, ConnectionState } from '../types';
  import ChatMessage from './ChatMessage.svelte';
  import TypingIndicator from './TypingIndicator.svelte';
  import Waveform from './Waveform.svelte';

  /**
   * Scrollable chat message panel with empty state, typing indicator,
   * and waveform animation.
   */
  let {
    messages,
    connectionState,
  }: {
    messages: ChatMessageType[];
    connectionState: ConnectionState;
  } = $props();

  let chatArea: HTMLElement;

  // Auto-scroll to bottom when new messages arrive
  $effect(() => {
    // Access messages.length to create the reactive dependency
    if (messages.length && chatArea) {
      requestAnimationFrame(() => {
        chatArea.scrollTop = chatArea.scrollHeight;
      });
    }
  });
</script>

<div class="chat-panel" bind:this={chatArea}>
  {#if messages.length === 0}
    <div class="chat-empty">
      <div class="logo">S</div>
      <div>Hi, I'm <strong>SETH</strong></div>
      <div>Type a message or tap the mic</div>
      <div class="hint">Say <em>"HEY"</em> to wake me up</div>
    </div>
  {:else}
    {#each messages as msg (msg.id)}
      <ChatMessage message={msg} />
    {/each}
  {/if}

  <TypingIndicator visible={connectionState === 'processing'} />
  <Waveform visible={connectionState === 'speaking'} />
</div>

<style>
  .chat-panel {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    scroll-behavior: smooth;
  }

  .chat-panel::-webkit-scrollbar { width: 4px; }
  .chat-panel::-webkit-scrollbar-track { background: transparent; }
  .chat-panel::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
  }

  .chat-empty {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    color: var(--text-dim);
    font-size: 14px;
    text-align: center;
  }

  .chat-empty .logo {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    font-weight: 700;
    color: #fff;
    margin-bottom: 6px;
    box-shadow: 0 4px 24px rgba(99, 102, 241, 0.35);
  }

  .chat-empty .hint {
    font-size: 11px;
    margin-top: 4px;
    opacity: 0.6;
  }
</style>
