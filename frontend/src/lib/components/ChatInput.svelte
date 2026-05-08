<script lang="ts">
  /**
   * Bottom input bar with text field, mic button, and send button.
   */
  let {
    canSend = false,
    isConnected = false,
    isRecording = false,
    onSendText,
    onToggleRecording,
  }: {
    canSend: boolean;
    isConnected: boolean;
    isRecording: boolean;
    onSendText: (text: string) => void;
    onToggleRecording: () => void;
  } = $props();

  let inputValue = $state('');
  let inputEl: HTMLInputElement;

  function handleSend() {
    const text = inputValue.trim();
    if (!text) return;
    onSendText(text);
    inputValue = '';
    inputEl?.focus();
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }
</script>

<div class="chat-input">
  <div class="input-row">
    <button
      class="mic-btn"
      class:recording={isRecording}
      disabled={!isConnected}
      title={isRecording ? 'Stop recording' : 'Start recording'}
      onclick={onToggleRecording}
    >
      <svg viewBox="0 0 24 24">
        <path d="M12 2a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3zm6 9a1 1 0 0 0-2 0 4 4 0 0 1-8 0 1 1 0 0 0-2 0 6 6 0 0 0 5 5.91V20H8a1 1 0 0 0 0 2h8a1 1 0 0 0 0-2h-3v-3.09A6 6 0 0 0 18 11z" />
      </svg>
    </button>

    <input
      type="text"
      bind:this={inputEl}
      bind:value={inputValue}
      placeholder="Type a message..."
      autocomplete="off"
      disabled={!isConnected}
      onkeydown={handleKeydown}
    />

    <button
      class="send-btn"
      disabled={!canSend || !inputValue.trim()}
      title="Send message"
      onclick={handleSend}
    >
      <svg viewBox="0 0 24 24">
        <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
      </svg>
    </button>
  </div>
</div>

<style>
  .chat-input {
    padding: 14px 20px;
    border-top: 1px solid var(--border);
    background: rgba(0, 0, 0, 0.2);
    flex-shrink: 0;
  }

  .input-row {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .input-row input {
    flex: 1;
    padding: 11px 16px;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: var(--bg-input);
    color: var(--text);
    font-size: 13.5px;
    font-family: var(--font);
    outline: none;
    transition: border-color 0.2s;
  }
  .input-row input::placeholder { color: var(--text-dim); }
  .input-row input:focus { border-color: var(--accent); }
  .input-row input:disabled { opacity: 0.5; }

  .send-btn {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: transform 0.15s, opacity 0.2s;
    flex-shrink: 0;
  }
  .send-btn:hover { transform: scale(1.05); }
  .send-btn:active { transform: scale(0.95); }
  .send-btn:disabled { opacity: 0.35; cursor: default; transform: none; }
  .send-btn svg { width: 18px; height: 18px; fill: currentColor; }

  .mic-btn {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    border: 2px solid var(--border);
    background: transparent;
    color: var(--text-dim);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    flex-shrink: 0;
  }
  .mic-btn:hover { border-color: var(--accent); color: var(--accent); }
  .mic-btn svg { width: 18px; height: 18px; fill: currentColor; }
  .mic-btn:disabled { opacity: 0.35; cursor: default; }

  .mic-btn.recording {
    border-color: var(--danger);
    color: var(--danger);
    background: rgba(239, 68, 68, 0.1);
    animation: mic-pulse 1.2s infinite;
  }

  @keyframes mic-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.3); }
    50%      { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
  }
</style>
