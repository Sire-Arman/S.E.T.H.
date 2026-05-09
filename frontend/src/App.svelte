<script lang="ts">
  import { chat } from './lib/stores/chat.svelte';
  import StatusBar from './lib/components/StatusBar.svelte';
  import ChatPanel from './lib/components/ChatPanel.svelte';
  import ChatInput from './lib/components/ChatInput.svelte';
  import ArtifactPanel from './lib/components/ArtifactPanel.svelte';
  import { preloadHighlighter } from './lib/utils/shiki';
  import { preloadKatex } from './lib/utils/purifier';

  // Connect and preload rendering engines on mount
  $effect(() => {
    chat.connect();
    preloadHighlighter();
    preloadKatex();
    return () => chat.disconnect();
  });
</script>

<div class="app" class:panel-open={chat.activeArtifact !== null}>
  <div class="chat-container">
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

  <ArtifactPanel
    artifact={chat.activeArtifact}
    onClose={() => chat.closeArtifact()}
  />
</div>

<style>
  .app {
    width: 100%;
    height: 100dvh;
    display: flex;
    justify-content: center;
    position: relative;
  }

  .chat-container {
    width: 100%;
    max-width: 720px;
    height: 100dvh;
    display: flex;
    flex-direction: column;
    background: var(--bg-panel);
    border-left: 1px solid var(--border);
    border-right: 1px solid var(--border);
    transition: max-width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
  }

  /* When artifact panel is open, shrink the chat */
  .app.panel-open .chat-container {
    max-width: 50vw;
    margin-right: 0;
    margin-left: auto;
  }

  @media (max-width: 768px) {
    .app.panel-open .chat-container {
      max-width: 100vw;
    }
  }

  /* Subtle side glow for desktop */
  @media (min-width: 740px) {
    .chat-container {
      box-shadow:
        -40px 0 80px -20px rgba(99, 102, 241, 0.06),
         40px 0 80px -20px rgba(139, 92, 246, 0.06);
    }
  }
</style>
