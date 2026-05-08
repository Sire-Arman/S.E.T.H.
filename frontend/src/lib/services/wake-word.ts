/* ================================================================
   WakeWordDetector — Browser SpeechRecognition wake word listener.

   Uses the Web Speech API for continuous listening. The phrase is
   configurable (default: "hey").

   ⚠ Known Issues / Limitations:
   1. "hey" alone has HIGH false-positive rate — very common word.
      Consider a longer phrase for production use.
   2. Chrome sends audio to Google's servers for recognition (privacy).
   3. NOT available in Tauri system webview — swap for a WASM-based
      VAD (e.g. @ricky0123/vad-web with Silero) or Tauri plugin.
   4. Can conflict with MediaRecorder on some platforms when both
      access the microphone simultaneously.
   5. No actual Voice Activity Detection — this is speech-to-text,
      not energy-based silence detection. For auto-stop recording
      based on silence, a true VAD is needed.

   Architecture note: This class is designed as a swappable service.
   For Tauri, replace with a native wake word engine via the plugin
   system while keeping the same enable/disable/callback interface.
   ================================================================ */

export type WakeWordCallback = () => void;

export class WakeWordDetector {
  private recognition: SpeechRecognition | null = null;
  private _enabled = false;
  private _supported = false;
  private phrase: string;
  private onDetected: WakeWordCallback | null = null;

  constructor(phrase: string = 'hey') {
    this.phrase = phrase.toLowerCase();

    const SpeechRecognitionAPI =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognitionAPI) {
      console.warn('[SETH] SpeechRecognition API not available in this browser');
      return;
    }

    this._supported = true;
    this.recognition = new SpeechRecognitionAPI();
    this.recognition.continuous = true;
    this.recognition.interimResults = true;
    this.recognition.lang = 'en-US';

    this.recognition.onresult = (event: SpeechRecognitionEvent) => {
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript.toLowerCase().trim();
        if (transcript.includes(this.phrase)) {
          console.log('[SETH] Wake word detected:', transcript);
          this.onDetected?.();
          return;
        }
      }
    };

    this.recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      if (event.error === 'no-speech' || event.error === 'aborted') return;
      console.warn('[SETH] Recognition error:', event.error);
    };

    this.recognition.onend = () => {
      // Auto-restart if still enabled
      if (this._enabled) {
        try {
          this.recognition?.start();
        } catch {
          /* already started */
        }
      }
    };
  }

  get supported(): boolean {
    return this._supported;
  }

  get enabled(): boolean {
    return this._enabled;
  }

  /** Register the callback fired when the wake word is detected. */
  setCallback(cb: WakeWordCallback): void {
    this.onDetected = cb;
  }

  enable(): void {
    if (!this.recognition) return;
    this._enabled = true;
    try {
      this.recognition.start();
    } catch {
      /* already started */
    }
  }

  disable(): void {
    if (!this.recognition) return;
    this._enabled = false;
    try {
      this.recognition.stop();
    } catch {
      /* not started */
    }
  }

  toggle(): boolean {
    if (this._enabled) {
      this.disable();
    } else {
      this.enable();
    }
    return this._enabled;
  }
}
