/* ================================================================
   AudioPlayer — Queue-based WAV playback via Web Audio API.
   Decodes base64-encoded WAV chunks and plays them sequentially.
   ================================================================ */

export class AudioPlayer {
  private queue: string[] = [];
  private playing = false;
  private ctx: AudioContext | null = null;
  private onStateChange: ((playing: boolean) => void) | null = null;
  private currentSource: AudioBufferSourceNode | null = null;
  private safetyTimer: ReturnType<typeof setTimeout> | null = null;
  private warmedUp = false;

  /** Register a callback fired when playback starts/stops. */
  setStateCallback(cb: (playing: boolean) => void): void {
    this.onStateChange = cb;
  }

  private async getContext(): Promise<AudioContext> {
    if (!this.ctx) {
      this.ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      console.log('[SETH] AudioContext created, state:', this.ctx.state);
    }
    // Resume if suspended (browser autoplay policy) — MUST await
    if (this.ctx.state === 'suspended') {
      try {
        await this.ctx.resume();
        console.log('[SETH] AudioContext resumed, state:', this.ctx.state);
      } catch (err) {
        console.warn('[SETH] AudioContext resume failed:', err);
      }
    }
    return this.ctx;
  }

  /**
   * Pre-create and resume AudioContext during a user gesture.
   * Call this from click/keypress handlers (sendText, toggleRecording)
   * so the context is unlocked BEFORE server audio arrives.
   */
  async warmup(): Promise<void> {
    if (this.warmedUp) return;
    try {
      await this.getContext();
      this.warmedUp = true;
      console.log('[SETH] AudioPlayer warmed up — context state:', this.ctx?.state);
    } catch (err) {
      console.warn('[SETH] AudioPlayer warmup failed:', err);
    }
  }

  /** Add a base64-encoded WAV to the playback queue. */
  enqueue(base64Wav: string): void {
    console.log('[SETH] Audio enqueued, queue size:', this.queue.length + 1, 'playing:', this.playing, 'ctx state:', this.ctx?.state);
    this.queue.push(base64Wav);
    if (!this.playing) this.playNext();
  }

  private clearSafetyTimer(): void {
    if (this.safetyTimer) {
      clearTimeout(this.safetyTimer);
      this.safetyTimer = null;
    }
  }

  private async playNext(): Promise<void> {
    this.clearSafetyTimer();
    this.currentSource = null;

    if (this.queue.length === 0) {
      this.playing = false;
      this.onStateChange?.(false);
      return;
    }

    this.playing = true;
    this.onStateChange?.(true);

    const base64 = this.queue.shift()!;
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }

    try {
      const ctx = await this.getContext();
      console.log('[SETH] Decoding audio, context state:', ctx.state, 'data size:', bytes.length);
      const buffer = await ctx.decodeAudioData(bytes.buffer.slice(0));

      // Skip zero-length buffers (would stall playback)
      if (buffer.length === 0 || buffer.duration === 0) {
        console.warn('[SETH] Skipping zero-length audio buffer');
        this.playNext();
        return;
      }

      console.log('[SETH] Playing audio buffer: duration=', buffer.duration.toFixed(2), 's, sampleRate=', buffer.sampleRate);

      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      this.currentSource = source;

      let ended = false;
      source.onended = () => {
        if (ended) return; // Prevent double-fire
        ended = true;
        console.log('[SETH] Audio chunk finished playing');
        this.clearSafetyTimer();
        this.playNext();
      };

      source.start();

      // Safety timeout: if onended doesn't fire within buffer duration + 5s,
      // force advance to prevent permanent stuck state
      const timeoutMs = (buffer.duration + 5) * 1000;
      this.safetyTimer = setTimeout(() => {
        if (!ended) {
          console.warn('[SETH] Audio onended did not fire, forcing advance');
          ended = true;
          try { source.stop(); } catch { /* already stopped */ }
          this.playNext();
        }
      }, timeoutMs);
    } catch (err) {
      console.error('[SETH] Audio decode/playback error:', err);
      this.playNext(); // Skip corrupted chunk
    }
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  /** Check if there is queued audio waiting to play. */
  get hasQueued(): boolean {
    return this.queue.length > 0;
  }

  /** Clear the playback queue (does not stop current playback). */
  clear(): void {
    this.queue = [];
  }

  /** Force stop all playback and reset state. */
  forceStop(): void {
    this.clearSafetyTimer();
    this.queue = [];
    try { this.currentSource?.stop(); } catch { /* already stopped */ }
    this.currentSource = null;
    if (this.playing) {
      this.playing = false;
      this.onStateChange?.(false);
    }
  }
}

