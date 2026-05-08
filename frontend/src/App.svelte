<script lang="ts">
  import { chat } from './lib/stores/chat.svelte';
  import StatusBar from './lib/components/StatusBar.svelte';
  import ChatPanel from './lib/components/ChatPanel.svelte';
  import ChatInput from './lib/components/ChatInput.svelte';

  // Connect on mount
  $effect(() => {
    chat.connect();
    return () => chat.disconnect();
  });
</script>

<div class="app">
  <StatusBar
    connectionState={chat.connectionState}
    wakeWordEnabled={chat.wakeWordEnabled}
    wakeWordSupported={chat.wakeWordSupported}
    onToggleWake={() => chat.toggleWakeWord()}
  />

  <ChatPanel
    messages={chat.messages}
    connectionState={chat.connectionState}
  />

  <ChatInput
    canSend={chat.canSend}
    isConnected={chat.isConnected}
    isRecording={chat.isRecording}
    onSendText={(text) => chat.sendText(text)}
    onToggleRecording={() => chat.toggleRecording()}
  />
</div>

<style>
  .app {
    width: 100%;
    max-width: 720px;
    height: 100dvh;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    background: var(--bg-panel);
    border-left: 1px solid var(--border);
    border-right: 1px solid var(--border);
    position: relative;
  }

  /* Subtle side glow for desktop */
  @media (min-width: 740px) {
    .app {
      box-shadow:
        -40px 0 80px -20px rgba(99, 102, 241, 0.06),
         40px 0 80px -20px rgba(139, 92, 246, 0.06);
    }
  }
</style>
