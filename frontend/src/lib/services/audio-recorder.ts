/* ================================================================
   AudioRecorder — MediaRecorder wrapper for mic capture.
   Records audio as WebM, returns base64 for WebSocket transport.
   Designed as a swappable service for Tauri native mic plugin.
   ================================================================ */

export class AudioRecorder {
  private mediaRecorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private stream: MediaStream | null = null;
  private _recording = false;

  get recording(): boolean {
    return this._recording;
  }

  /** Start capturing audio from the microphone. */
  async start(): Promise<void> {
    if (this._recording) return;

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.stream = stream;
    this.mediaRecorder = new MediaRecorder(stream);
    this.chunks = [];

    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) this.chunks.push(e.data);
    };

    this.mediaRecorder.start();
    this._recording = true;
  }

  /** Stop recording and return the captured audio as a base64 string. */
  stop(): Promise<string> {
    return new Promise((resolve, reject) => {
      if (!this._recording || !this.mediaRecorder) {
        reject(new Error('Not recording'));
        return;
      }

      this.mediaRecorder.onstop = async () => {
        // Small delay to ensure all chunks are flushed
        await new Promise((r) => setTimeout(r, 100));

        const blob = new Blob(this.chunks, { type: 'audio/webm' });
        const reader = new FileReader();

        reader.onload = () => {
          const base64 = (reader.result as string).split(',')[1];
          resolve(base64);
        };
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(blob);

        // Release the mic
        this.stream?.getTracks().forEach((t) => t.stop());
        this.stream = null;
      };

      this.mediaRecorder.stop();
      this._recording = false;
    });
  }
}
