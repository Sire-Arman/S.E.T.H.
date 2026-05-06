/* ================================================================
   SETH Widget — Core Logic
   ================================================================ */

(function () {
  "use strict";

  // ── Config ──────────────────────────────────────────────────
  const WS_URL = "ws://127.0.0.1:8765";
  const WAKE_PHRASE = "hey seth";

  // ── State ───────────────────────────────────────────────────
  let ws = null;
  let isOpen = false;
  let isRecording = false;
  let mediaRecorder = null;
  let audioChunks = [];
  let wakeEnabled = false;
  let recognition = null;
  let audioQueue = [];
  let isPlaying = false;
  let audioCtx = null;
  let currentState = "idle"; // idle | listening | processing | speaking | disconnected

  // ── DOM refs ────────────────────────────────────────────────
  const $ = (s) => document.getElementById(s);
  const fab = $("widget-fab");
  const panel = $("widget-panel");
  const chatArea = $("chat-area");
  const chatEmpty = $("chat-empty");
  const textInput = $("text-input");
  const sendBtn = $("send-btn");
  const micBtn = $("mic-btn");
  const wakeBtn = $("wake-btn");
  const minimizeBtn = $("minimize-btn");
  const typingEl = $("typing-indicator");
  const waveformEl = $("waveform");
  const statusDot = $("status-dot");
  const statusLabel = $("status-label");

  // ── Utilities ───────────────────────────────────────────────
  function setState(state) {
    currentState = state;
    const dot = statusDot;
    dot.className = "dot";

    switch (state) {
      case "idle":
        dot.classList.add("connected");
        statusLabel.textContent = "Ready";
        typingEl.classList.add("hidden");
        waveformEl.classList.add("hidden");
        break;
      case "listening":
        dot.classList.add("listening");
        statusLabel.textContent = "Listening...";
        typingEl.classList.add("hidden");
        waveformEl.classList.add("hidden");
        break;
      case "processing":
        dot.classList.add("processing");
        statusLabel.textContent = "Thinking...";
        typingEl.classList.remove("hidden");
        waveformEl.classList.add("hidden");
        scrollChat();
        break;
      case "speaking":
        dot.classList.add("speaking");
        statusLabel.textContent = "Speaking...";
        typingEl.classList.add("hidden");
        waveformEl.classList.remove("hidden");
        break;
      case "disconnected":
        dot.className = "dot";
        statusLabel.textContent = "Disconnected";
        typingEl.classList.add("hidden");
        waveformEl.classList.add("hidden");
        break;
    }
  }

  function addMessage(text, type) {
    // type: "user" | "bot" | "system"
    if (chatEmpty) chatEmpty.style.display = "none";
    const div = document.createElement("div");
    div.className = "msg " + type;
    div.textContent = text;
    chatArea.appendChild(div);
    scrollChat();
  }

  function scrollChat() {
    requestAnimationFrame(() => {
      chatArea.scrollTop = chatArea.scrollHeight;
    });
  }

  // ── WebSocket ───────────────────────────────────────────────
  function connect() {
    if (ws && ws.readyState <= 1) return; // already open/connecting

    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      setState("idle");
      sendBtn.disabled = false;
      micBtn.disabled = false;
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleServerMessage(msg);
      } catch (e) {
        addMessage(event.data, "bot");
      }
    };

    ws.onerror = () => {
      setState("disconnected");
    };

    ws.onclose = () => {
      setState("disconnected");
      sendBtn.disabled = true;
      micBtn.disabled = true;
      // Auto-reconnect after 3s
      setTimeout(() => {
        if (isOpen) connect();
      }, 3000);
    };
  }

  function handleServerMessage(msg) {
    switch (msg.type) {
      case "sentence":
        // Streaming sentence text — show as bot message
        addMessage(msg.data, "bot");
        if (currentState === "processing") setState("speaking");
        break;

      case "audio_response":
        // Base64 WAV — queue for playback
        queueAudio(msg.data);
        break;

      case "response":
        // Final full response — streaming is done
        // (sentences already displayed individually)
        break;

      case "transcript":
        // User's speech transcription
        addMessage(msg.data, "user");
        setState("processing");
        break;

      case "error":
        addMessage("Error: " + msg.data, "system");
        setState("idle");
        break;

      default:
        if (msg.data) addMessage(msg.data, "bot");
    }
  }

  // ── Audio Playback ──────────────────────────────────────────
  function getAudioContext() {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    return audioCtx;
  }

  function queueAudio(base64Wav) {
    audioQueue.push(base64Wav);
    if (!isPlaying) playNext();
  }

  async function playNext() {
    if (audioQueue.length === 0) {
      isPlaying = false;
      if (currentState === "speaking") setState("idle");
      return;
    }

    isPlaying = true;
    if (currentState !== "speaking") setState("speaking");

    const base64 = audioQueue.shift();
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }

    try {
      const ctx = getAudioContext();
      const audioBuffer = await ctx.decodeAudioData(bytes.buffer.slice(0));
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      source.onended = () => playNext();
      source.start();
    } catch (err) {
      console.error("[SETH] Audio decode error:", err);
      playNext();
    }
  }

  // ── Text Input ──────────────────────────────────────────────
  function sendText() {
    const text = textInput.value.trim();
    if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

    addMessage(text, "user");
    ws.send(JSON.stringify({ type: "message", data: text }));
    textInput.value = "";
    setState("processing");
  }

  // ── Mic Recording ──────────────────────────────────────────
  async function startRecording() {
    if (isRecording) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        await new Promise((r) => setTimeout(r, 100));
        const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
        const reader = new FileReader();
        reader.onload = () => {
          const base64 = reader.result.split(",")[1];
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "audio", data: base64 }));
            setState("processing");
          }
        };
        reader.readAsDataURL(audioBlob);
        stream.getTracks().forEach((t) => t.stop());
      };

      mediaRecorder.start();
      isRecording = true;
      micBtn.classList.add("recording");
      setState("listening");
    } catch (err) {
      addMessage("Microphone access denied", "system");
    }
  }

  function stopRecording() {
    if (!isRecording || !mediaRecorder) return;
    mediaRecorder.stop();
    isRecording = false;
    micBtn.classList.remove("recording");
  }

  function toggleRecording() {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }

  // ── Wake Word Detection ─────────────────────────────────────
  function initWakeWord() {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.warn("[SETH] SpeechRecognition not supported in this browser");
      wakeBtn.style.display = "none";
      return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript.toLowerCase().trim();
        if (transcript.includes(WAKE_PHRASE)) {
          console.log("[SETH] Wake word detected:", transcript);
          onWakeWordDetected();
          return;
        }
      }
    };

    recognition.onerror = (event) => {
      if (event.error === "no-speech" || event.error === "aborted") return;
      console.warn("[SETH] Recognition error:", event.error);
    };

    recognition.onend = () => {
      // Auto-restart if still enabled and not currently recording
      if (wakeEnabled && !isRecording) {
        try {
          recognition.start();
        } catch (e) {
          /* already started */
        }
      }
    };
  }

  function onWakeWordDetected() {
    // Flash the FAB / panel briefly
    panel.style.animation = "none";
    void panel.offsetWidth; // reflow
    panel.style.animation = "";

    // Ensure widget is open
    if (!isOpen) togglePanel();

    // Auto-start recording if connected
    if (ws && ws.readyState === WebSocket.OPEN && !isRecording) {
      addMessage("Wake word detected!", "system");
      startRecording();
      // Auto-stop after 5 seconds
      setTimeout(() => {
        if (isRecording) stopRecording();
      }, 5000);
    }
  }

  function toggleWakeWord() {
    if (!recognition) {
      addMessage("Wake word not supported in this browser", "system");
      return;
    }

    wakeEnabled = !wakeEnabled;
    wakeBtn.classList.toggle("wake-active", wakeEnabled);
    wakeBtn.title = wakeEnabled
      ? 'Wake word ON — say "hey SETH"'
      : "Wake word OFF";

    if (wakeEnabled) {
      try {
        recognition.start();
      } catch (e) {
        /* already started */
      }
      addMessage('Wake word enabled — say "hey SETH"', "system");
    } else {
      try {
        recognition.stop();
      } catch (e) {
        /* not started */
      }
      addMessage("Wake word disabled", "system");
    }
  }

  // ── Panel Toggle ────────────────────────────────────────────
  function togglePanel() {
    isOpen = !isOpen;
    panel.classList.toggle("hidden", !isOpen);
    fab.classList.toggle("hidden", isOpen);

    if (isOpen) {
      connect();
      textInput.focus();
    }
  }

  // ── Event Binding ───────────────────────────────────────────
  fab.addEventListener("click", togglePanel);
  minimizeBtn.addEventListener("click", togglePanel);
  sendBtn.addEventListener("click", sendText);
  micBtn.addEventListener("click", toggleRecording);
  wakeBtn.addEventListener("click", toggleWakeWord);
  textInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendText();
    }
  });

  // Init wake word engine
  initWakeWord();
})();
