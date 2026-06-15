/**
 * SETH Rendering Pipeline
 *
 * Flow: raw text → redact sensitive data → KaTeX math → Marked markdown
 *       (with Shiki code blocks + Mermaid placeholders) → DOMPurify → safe HTML
 */

import { Marked } from 'marked';
import DOMPurify from 'dompurify';
import { highlightCode } from './shiki';

// ── Privacy Shield ───────────────────────────────────────────────

const SENSITIVE = {
  apiKey: /\b(sk|pk|ak|uk|tvly|gsk|hf)[-_][a-zA-Z0-9]{15,}\b/g,
  bearer: /Bearer\s+[a-zA-Z0-9._\-]{20,}/g,
};

function redact(text: string): string {
  let out = text;
  out = out.replace(SENSITIVE.apiKey, (m) => m.substring(0, 6) + '•••' + m.slice(-3));
  out = out.replace(SENSITIVE.bearer, 'Bearer •••••••');
  return out;
}

// ── KaTeX Math ───────────────────────────────────────────────────

let katex: typeof import('katex') | null = null;

async function ensureKatex() {
  if (!katex) {
    katex = await import('katex');
  }
  return katex;
}

/** Synchronous KaTeX render (after dynamic import). */
function renderMath(latex: string, displayMode: boolean): string {
  if (!katex) {
    return displayMode ? `<div class="math-fallback">$$${latex}$$</div>`
                       : `<code class="math-fallback">$${latex}$</code>`;
  }
  try {
    return katex.default.renderToString(latex, {
      displayMode,
      throwOnError: false,
      output: 'htmlAndMathml',
    });
  } catch {
    return `<code class="math-error">${latex}</code>`;
  }
}

/**
 * Replace $...$ and $$...$$ with KaTeX-rendered HTML.
 * Must run BEFORE marked to protect math from markdown parsing.
 */
function processKatex(text: string): string {
  // Block math: $$...$$
  let out = text.replace(/\$\$([^$]+?)\$\$/g, (_, tex) => {
    return renderMath(tex.trim(), true);
  });
  // Inline math: $...$  (not greedy, no newlines)
  out = out.replace(/\$([^$\n]+?)\$/g, (_, tex) => {
    return renderMath(tex.trim(), false);
  });
  return out;
}

// ── Mermaid Placeholder ──────────────────────────────────────────

let mermaidIdCounter = 0;
let renderCycleId = 0;

/**
 * Wraps mermaid code in a placeholder div.
 * The actual rendering happens in the Svelte component via `use:action`.
 */
function mermaidPlaceholder(code: string): string {
  const id = `mermaid-${++mermaidIdCounter}`;
  const encoded = btoa(encodeURIComponent(code));
  return `<div class="mermaid-placeholder" data-mermaid-id="${id}" data-mermaid-code="${encoded}"></div>`;
}

// ── Artifact Detection ───────────────────────────────────────────

export interface DetectedArtifact {
  id: string;
  language: string;
  code: string;
}

const ARTIFACT_LINE_THRESHOLD = 12;
let artifactCounter = 0;
let lastDetectedArtifacts: DetectedArtifact[] = [];

/** Get artifacts detected during the last render call. */
export function getLastArtifacts(): DetectedArtifact[] {
  return lastDetectedArtifacts;
}

// ── Marked Instance ──────────────────────────────────────────────

const renderer = {
  code(token: { text: string; lang?: string }): string {
    const code = token.text;
    const lang = (token.lang || '').trim().toLowerCase();

    // Mermaid diagrams → placeholder for Svelte action
    if (lang === 'mermaid') {
      return mermaidPlaceholder(code);
    }

    const lines = code.split('\n').length;
    const highlighted = highlightCode(code, lang || 'text');

    // Detect artifacts (large code blocks)
    let artifactBtn = '';
    if (lines >= ARTIFACT_LINE_THRESHOLD) {
      const artId = `artifact-${++artifactCounter}`;
      lastDetectedArtifacts.push({ id: artId, language: lang || 'text', code });
      artifactBtn = `<button class="artifact-open-btn" data-artifact-id="${artId}">Open in Panel ↗</button>`;
    }

    // Language badge + copy button wrapper
    const langBadge = lang ? `<span class="code-lang">${lang}</span>` : '';
    const copyBtn = `<button class="code-copy-btn" data-code="${btoa(encodeURIComponent(code))}">Copy</button>`;

    return `<div class="code-block-wrapper">${langBadge}<div class="code-toolbar">${copyBtn}${artifactBtn}</div>${highlighted}</div>`;
  },
};

const markedInstance = new Marked({ renderer, breaks: true });

// ── DOMPurify Config ─────────────────────────────────────────────

// Allow Shiki's inline styles, KaTeX elements, Mermaid SVGs, and our custom buttons
const PURIFY_CONFIG = {
  ADD_ATTR: ['style', 'data-mermaid-id', 'data-mermaid-code', 'data-artifact-id', 'data-code'],
  ADD_TAGS: ['svg', 'path', 'g', 'rect', 'circle', 'line', 'polyline', 'polygon',
             'text', 'tspan', 'defs', 'clipPath', 'use', 'foreignObject',
             'annotation', 'semantics', 'mrow', 'mi', 'mo', 'mn', 'msup',
             'msub', 'mfrac', 'mover', 'munder', 'math', 'button', 'span'],
};

/**
 * Convert <artifact> tags into standard Markdown code blocks.
 * Handles flexible attribute ordering (language/title in any order).
 */
function processArtifacts(text: string): string {
  return text.replace(
    /<artifact\s+([^>]*?)>([\s\S]*?)<\/artifact>/g,
    (_, attrs, code) => {
      const langMatch = attrs.match(/language="([^"]+)"/);
      const titleMatch = attrs.match(/title="([^"]+)"/);
      const lang = langMatch ? langMatch[1] : 'text';
      const title = titleMatch ? titleMatch[1] : '';
      const header = title ? `### ${title}\n` : '';
      return `\n\n${header}\`\`\`${lang}\n${code.trim()}\n\`\`\`\n\n`;
    }
  );
}

/**
 * Convert raw <function=generate_code>{...}</function> tags into Markdown code blocks.
 *
 * Groq/Llama models sometimes emit raw tool-call syntax instead of actually
 * invoking the tool. This catches those and renders them as proper code blocks
 * so they don't leak as ugly raw text into the chat UI.
 */
function processRawFunctionTags(text: string): string {
  return text.replace(
    /<function=generate_code>\s*(\{[\s\S]*?\})\s*<\/function>/g,
    (_, jsonStr) => {
      try {
        const parsed = JSON.parse(jsonStr);
        const lang = parsed.language || 'text';
        const code = parsed.code || '';
        const title = parsed.title || '';
        const header = title ? `### ${title}\n` : '';
        return `\n\n${header}\`\`\`${lang}\n${code}\n\`\`\`\n\n`;
      } catch {
        // If JSON parse fails, just strip the tags and show as-is
        return `\n\n\`\`\`\n${jsonStr}\n\`\`\`\n\n`;
      }
    }
  );
}

/**
 * Auto-close any unclosed fenced code blocks.
 * LLMs sometimes truncate mid-block, leaving orphan ``` openers.
 */
function autoCloseFences(text: string): string {
  // Count triple-backtick fences — if odd, append a closing one
  const fences = text.match(/^```/gm);
  if (fences && fences.length % 2 !== 0) {
    return text + '\n```';
  }
  return text;
}

// ── Public API ───────────────────────────────────────────────────

/**
 * Full rendering pipeline for chat messages.
 * Returns sanitized HTML ready for {@html} injection.
 */
export function purify(text: string): string {
  if (!text) return '';

  // Reset per-render state
  lastDetectedArtifacts = [];
  mermaidIdCounter = 0;       // Prevent stale mermaid IDs across re-renders
  renderCycleId++;

  // 1. Redact sensitive data
  let processed = redact(text);

  // 2. Convert raw <function=generate_code> tags → Markdown (Groq/Llama leakage)
  processed = processRawFunctionTags(processed);

  // 3. Process custom <artifact> tags → Markdown
  processed = processArtifacts(processed);

  // 4. Auto-close any orphan code fences (LLM truncation)
  processed = autoCloseFences(processed);

  // 4. Process KaTeX math (before markdown to protect $ delimiters)
  processed = processKatex(processed);

  // 5. Markdown → HTML (with Shiki code blocks + Mermaid placeholders)
  const html = markedInstance.parse(processed) as string;

  // 6. Sanitize
  return DOMPurify.sanitize(html, PURIFY_CONFIG);
}

/** Pre-warm KaTeX (call once at startup). */
export function preloadKatex(): void {
  ensureKatex().catch((e) => console.warn('[SETH] KaTeX preload failed:', e));
}
