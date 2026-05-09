/**
 * Shiki Highlighter Singleton
 *
 * Lazy-loads the WASM highlighter on first use.
 * Once loaded, `codeToHtml()` is synchronous and fast.
 */

import type { BundledLanguage, BundledTheme, HighlighterGeneric } from 'shiki';

let _highlighter: HighlighterGeneric<BundledLanguage, BundledTheme> | null = null;
let _loading: Promise<HighlighterGeneric<BundledLanguage, BundledTheme>> | null = null;

const THEME: BundledTheme = 'github-dark';

const LANGS: BundledLanguage[] = [
  'javascript', 'typescript', 'python', 'html', 'css',
  'json', 'bash', 'rust', 'go', 'java', 'c', 'cpp',
  'sql', 'yaml', 'toml', 'markdown', 'svelte', 'jsx', 'tsx',
];

/** Get the cached highlighter, or initialize it. */
export async function getHighlighter() {
  if (_highlighter) return _highlighter;

  if (!_loading) {
    _loading = (async () => {
      const { createHighlighter } = await import('shiki');
      const h = await createHighlighter({ themes: [THEME], langs: LANGS });
      _highlighter = h;
      return h;
    })();
  }

  return _loading;
}

/**
 * Synchronous highlight — returns highlighted HTML if the highlighter
 * is already loaded, otherwise returns a styled `<pre>` fallback.
 */
export function highlightCode(code: string, lang: string): string {
  if (!_highlighter) {
    // Fallback: plain pre/code with language class
    const escaped = code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return `<pre class="shiki-fallback"><code class="language-${lang}">${escaped}</code></pre>`;
  }

  try {
    // Check if language is loaded
    const loadedLangs = _highlighter.getLoadedLanguages();
    const safeLang = loadedLangs.includes(lang as BundledLanguage) ? lang : 'text';

    return _highlighter.codeToHtml(code, { lang: safeLang, theme: THEME });
  } catch {
    const escaped = code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return `<pre class="shiki-fallback"><code>${escaped}</code></pre>`;
  }
}

/** Pre-warm the highlighter (call from main.ts or App.svelte). */
export function preloadHighlighter(): void {
  getHighlighter().catch((e) => console.warn('[SETH] Shiki preload failed:', e));
}
