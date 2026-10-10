/** Compile authored Markdown into a static, accessible documentation website. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.dirname(here);
const out = path.join(root, process.env.ECG_DOCS_OUTPUT || 'docs');
const require = createRequire(import.meta.url);
const markedPath = require.resolve('marked', { paths: [here, ...(process.env.NODE_PATH || '').split(path.delimiter).filter(Boolean)] });
const { marked } = await import(pathToFileURL(markedPath).href);
const data = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const slug = value => value.toLowerCase().replace(/<[^>]+>/g, '').replace(/&[^;]+;/g, '').replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
const brand = `<svg viewBox="0 0 36 36" aria-hidden="true"><rect width="36" height="36" rx="10" fill="currentColor"/><path d="M5 19h7l3-9 4 17 3-8h9" fill="none" stroke="white" stroke-width="2.2" stroke-linejoin="round"/></svg>`;
const sidebar = active => {
  let previous = '';
  return data.pages.map(page => {
    const heading = previous !== page.section ? `<p class="nav-section">${escape(page.section)}</p>` : '';
    previous = page.section;
    return heading + `<a ${active === page.id ? 'aria-current="page"' : ''} href="${page.id}.html">${escape(page.title)}</a>`;
  }).join('');
};

function gallery() {
  const categories = [...new Set(data.gallery.examples.map(item => item.category))];
  return `<div class="gallery-tools"><label class="gallery-search">Find an example<input id="gallery-search" type="search" placeholder="Try crumpled, V1, JPEG…"></label><p id="gallery-count" aria-live="polite">${data.gallery.figures} examples</p></div>
    <div class="gallery-filters" aria-label="Filter examples by category"><button class="active" data-category="All" aria-pressed="true">All</button>${categories.map(category => `<button data-category="${escape(category)}" aria-pressed="false">${escape(category)}</button>`).join('')}</div>
    <div class="gallery-grid">${data.gallery.examples.map(item => `<a class="gallery-card" href="${item.image}" data-gallery-id="${escape(item.id)}" data-category="${escape(item.category)}" data-search="${escape((item.title + ' ' + item.description).toLowerCase())}"><img loading="lazy" width="480" height="360" src="${item.thumbnail}" alt="${escape(item.title)}: ${escape(item.description)}"><div><span>${escape(item.category)}</span><h3>${escape(item.title)}</h3><p>${escape(item.description)}</p></div></a>`).join('')}</div>
    <p id="gallery-empty" hidden>No examples match this filter.</p>`;
}

const search = [];
await fs.mkdir(out, { recursive: true });
await fs.copyFile(path.join(here, 'assets/style.css'), path.join(out, 'style.css'));
await fs.copyFile(path.join(here, 'assets/app.js'), path.join(out, 'app.js'));
await fs.writeFile(path.join(out, 'favicon.svg'), `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36"><rect width="36" height="36" rx="10" fill="#df5548"/><path d="M5 19h7l3-9 4 17 3-8h9" fill="none" stroke="white" stroke-width="2.2"/></svg>`);
for (const [index, page] of data.pages.entries()) {
  let source = await fs.readFile(path.join(here, 'pages', page.id + '.md'), 'utf8');
  for (const [key, value] of Object.entries(data.replacements)) source = source.replaceAll('{{' + key + '}}', value);
  source = source.replaceAll('{{gallery}}', gallery());
  if (/\{\{\w+\}\}/.test(source)) throw new Error('Unresolved content placeholder in ' + page.id);
  let body = marked.parse(source, { gfm: true });
  const toc = [];
  const ids = new Map();
  body = body.replace(/<h([23])>([\s\S]*?)<\/h\1>/g, (match, level, title) => {
    const base = slug(title);
    const count = ids.get(base) || 0;
    ids.set(base, count + 1);
    const id = count ? base + '-' + count : base;
    if (level === '2') toc.push({ title: title.replace(/<[^>]+>/g, ''), id });
    return `<h${level} id="${id}">${title}<a class="heading-anchor" href="#${id}" aria-label="Link to ${escape(title.replace(/<[^>]+>/g, ''))}">#</a></h${level}>`;
  });
  body = body.replace(/<p>(<img[^>]+>)<\/p>/g, '<figure class="doc-figure">$1</figure>');
  body = body.replace(/<table>/g, '<div class="table-scroll"><table>').replace(/<\/table>/g, '</table></div>');
  const previous = data.pages[index - 1];
  const next = data.pages[index + 1];
  const pagination = `<nav class="pagination" aria-label="Documentation pages">${previous ? `<a href="${previous.id}.html"><small>Previous</small>${escape(previous.title)}</a>` : '<span></span>'}${next ? `<a href="${next.id}.html"><small>Next</small>${escape(next.title)} <span aria-hidden="true">→</span></a>` : '<span></span>'}</nav>`;
  const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="description" content="${escape(page.description)}"><meta name="theme-color" content="#df5548"><title>${escape(page.title)} · ECG Torch Renderer</title><link rel="icon" href="favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="style.css"><script>try{document.documentElement.dataset.theme=localStorage.getItem('ecg-docs-theme')||'light'}catch{}</script><script src="app.js" defer></script></head>
<body><a class="skip-link" href="#main">Skip to content</a><div class="reading-progress" aria-hidden="true"></div>
<header class="topbar"><button id="menu-button" class="icon-button menu-button" aria-label="Toggle navigation" aria-controls="sidebar" aria-expanded="false">☰</button><a class="brand" href="index.html">${brand}<span>ECG<span class="brand-sub">Torch Renderer</span></span></a><span class="version">v${data.version}</span><div class="top-actions"><button id="search-button" class="search-button"><span>Search documentation</span><kbd>⌘ K</kbd></button><a class="github-link" href="https://github.com/fmenegui/ecg-torch-renderer">GitHub ↗</a><button id="theme-button" class="icon-button" aria-label="Switch color theme">◐</button></div></header>
<div class="docs-shell"><aside id="sidebar" class="sidebar"><nav aria-label="Main documentation">${sidebar(page.id)}</nav><div class="sidebar-note">Built for on-device training.<br>Measured results, explicit contracts.</div></aside>
<main id="main"><article class="document ${page.id === 'index' ? 'home' : ''}" data-page="${page.id}">${page.id !== 'index' ? `<div class="page-eyebrow">${escape(page.section)}</div>` : ''}${body}${pagination}<footer>ECG Torch Renderer · MIT · <a href="https://github.com/fmenegui/ecg-torch-renderer/blob/main/documentation/pages/${page.id}.md">Edit this page ↗</a><span>v${data.version}</span></footer></article></main>
<aside class="toc" aria-label="On this page"><p>On this page</p>${toc.map(item => `<a href="#${item.id}">${item.title}</a>`).join('')}<a class="toc-top" href="#main">Back to top ↑</a></aside></div>
<dialog id="search-dialog" class="search-dialog"><div class="dialog-top"><label for="search-input">Search the guide</label><button class="icon-button" data-close-dialog="search-dialog" aria-label="Close search">×</button></div><input id="search-input" type="search" placeholder="Masks, layouts, profiling…" autocomplete="off"><p id="search-status" class="search-status" aria-live="polite">Type to search all pages.</p><div id="search-results"></div></dialog>
<dialog id="figure-dialog" class="figure-dialog"><div class="dialog-top"><h2 id="figure-title">Example</h2><button class="icon-button" data-close-dialog="figure-dialog" aria-label="Close figure">×</button></div><img id="figure-image" alt=""><p id="figure-description"></p><div class="recipe-top"><strong>Rendering recipe</strong><button id="copy-recipe" class="button secondary">Copy JSON</button></div><pre id="figure-recipe"></pre><a id="figure-download" href="gallery.json">Open full image ↗</a></dialog>
<p id="copy-status" class="sr-only" aria-live="polite"></p></body></html>`;
  await fs.writeFile(path.join(out, page.id + '.html'), html);
  search.push({ title: page.title, description: page.description, url: page.id + '.html',
    text: source.replace(/<[^>]+>/g, ' ').replace(/```[\s\S]*?```/g, ' ').replace(/[#*`\[\]]/g, ' ').replace(/\s+/g, ' ') });
}
await fs.writeFile(path.join(out, 'search-index.json'), JSON.stringify(search));
await fs.writeFile(path.join(out, 'build-info.json'), JSON.stringify({ version: data.version, pages: data.pages.length, figures: data.gallery.figures, search: 'local full-text', theme: 'light and dark' }, null, 2) + '\n');
process.stdout.write(JSON.stringify({ pages: data.pages.length, figures: data.gallery.figures, output: out }) + '\n');
