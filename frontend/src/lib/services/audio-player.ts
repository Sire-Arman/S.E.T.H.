/* ================================================================
   AudioPlayer — Queue-based WAV playback via Web Audio API.
   Decodes base64-encoded WAV chunks and plays them sequentially.
   ================================================================ */

export class AudioPlayer {
  private queue: string[] = [];
  private playing = false;
  private ctx: AudioContext | null = null;
  private onStateChange: ((playing: boolean) => void) | null = null;

  /** Register a callback fired when playback starts/stops. */
  setStateCallback(cb: (playing: boolean) => void): void {
    this.onStateChange = cb;
  }

  private getContext(): AudioContext {
    if (!this.ctx) {
      this.ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    // Resume if suspended (browser autoplay policy)
    if (this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
    return this.ctx;
  }

  /** Add a base64-encoded WAV to the playback queue. */
  enqueue(base64Wav: string): void {
    this.queue.push(base64Wav);
    if (!this.playing) this.playNext();
  }

  private async playNext(): Promise<void> {
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
      const ctx = this.getContext();
      const buffer = await ctx.decodeAudioData(bytes.buffer.slice(0));
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      source.onended = () => this.playNext();
      source.start();
    } catch (err) {
      console.error('[SETH] Audio decode error:', err);
      this.playNext(); // Skip corrupted chunk
    }
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  /** Clear the playback queue (does not stop current playback). */
  clear(): void {
    this.queue = [];
  }
}
