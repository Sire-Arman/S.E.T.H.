<script lang="ts">
  /** Audio waveform animation shown when SETH is speaking. */
  let { visible = false }: { visible: boolean } = $props();
</script>

{#if visible}
  <div class="waveform">
    <span></span>
    <span></span>
    <span></span>
    <span></span>
    <span></span>
  </div>
{/if}

<style>
  .waveform {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 3px;
    height: 28px;
    padding: 4px 0;
  }

  .waveform span {
    width: 3px;
    border-radius: 2px;
    background: var(--accent);
    animation: wave 0.8s ease-in-out infinite;
  }

  .waveform span:nth-child(1) { height:  8px; animation-delay: 0.0s; }
  .waveform span:nth-child(2) { height: 16px; animation-delay: 0.1s; }
  .waveform span:nth-child(3) { height: 10px; animation-delay: 0.2s; }
  .waveform span:nth-child(4) { height: 20px; animation-delay: 0.3s; }
  .waveform span:nth-child(5) { height: 12px; animation-delay: 0.4s; }

  @keyframes wave {
    0%, 100% { transform: scaleY(0.5); }
    50%      { transform: scaleY(1.2); }
  }
</style>
