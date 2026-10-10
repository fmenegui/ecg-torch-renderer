/* Progressive enhancement. Pages and image links remain usable without JS. */
(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const announce = message => { byId('copy-status').textContent = message; };
  async function copy(text, button, label = 'Copied') {
    const original = button.textContent;
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = label;
      announce(label);
    } catch {
      button.textContent = 'Select and copy';
      announce('Clipboard unavailable. Select the text and copy it manually.');
    }
    setTimeout(() => { button.textContent = original; }, 1600);
  }
  document.querySelectorAll('pre > code').forEach(code => {
    const button = document.createElement('button');
    button.className = 'copy-code';
    button.textContent = 'Copy';
    button.setAttribute('aria-label', 'Copy code example');
    button.addEventListener('click', () => copy(code.textContent, button));
    code.parentElement.append(button);
  });
  byId('theme-button').addEventListener('click', () => {
    const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('ecg-docs-theme', theme); } catch {}
    byId('theme-button').setAttribute('aria-label', 'Switch to ' + (theme === 'dark' ? 'light' : 'dark') + ' theme');
  });
  byId('menu-button').addEventListener('click', () => {
    const open = byId('sidebar').classList.toggle('open');
    byId('menu-button').setAttribute('aria-expanded', String(open));
  });
  document.querySelectorAll('[data-close-dialog]').forEach(button => {
    button.addEventListener('click', () => byId(button.dataset.closeDialog).close());
  });
  document.querySelectorAll('dialog').forEach(dialog => {
    dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  });
  let index;
  async function openSearch() {
    const dialog = byId('search-dialog');
    dialog.showModal();
    byId('search-input').focus();
    try {
      if (!index) {
        const response = await fetch('search-index.json');
        if (!response.ok) throw new Error('Search index unavailable');
        index = await response.json();
      }
      search();
    } catch {
      byId('search-status').textContent = 'Search is unavailable. Use the page navigation.';
    }
  }
  function search() {
    if (!index) return;
    const terms = byId('search-input').value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const results = byId('search-results');
    results.replaceChildren();
    if (!terms.length) {
      byId('search-status').textContent = 'Type to search all pages.';
      return;
    }
    const matches = index.map(page => {
      const title = page.title.toLowerCase();
      const body = (page.description + ' ' + page.text).toLowerCase();
      const score = terms.every(term => title.includes(term) || body.includes(term)) ?
        terms.reduce((total, term) => total + (title.includes(term) ? 8 : 1), 0) : 0;
      return { page, score };
    }).filter(item => item.score).sort((a, b) => b.score - a.score).slice(0, 8);
    byId('search-status').textContent = matches.length ? matches.length + ' matching pages' : 'No matching pages. Try masks, layout or profiling.';
    matches.forEach(({ page }) => {
      const link = document.createElement('a');
      link.className = 'search-result';
      link.href = page.url;
      const title = document.createElement('strong');
      title.textContent = page.title;
      const description = document.createElement('p');
      description.textContent = page.description;
      link.append(title, description);
      results.append(link);
    });
  }
  byId('search-button').addEventListener('click', openSearch);
  byId('search-input').addEventListener('input', search);
  document.addEventListener('keydown', event => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      if (!byId('search-dialog').open) openSearch();
    }
    if (event.key === 'Escape') {
      byId('sidebar').classList.remove('open');
      byId('menu-button').setAttribute('aria-expanded', 'false');
    }
  });
  const progress = document.querySelector('.reading-progress');
  const updateProgress = () => {
    const max = document.documentElement.scrollHeight - innerHeight;
    progress.style.width = (max > 0 ? Math.min(100, scrollY / max * 100) : 100) + '%';
  };
  addEventListener('scroll', updateProgress, { passive: true });
  updateProgress();

  const cards = [...document.querySelectorAll('[data-gallery-id]')];
  if (!cards.length) return;
  let category = new URLSearchParams(location.search).get('category') || 'All';
  const categories = [...document.querySelectorAll('button[data-category]')];
  if (!categories.some(button => button.dataset.category === category)) category = 'All';
  function filter() {
    const query = byId('gallery-search').value.toLowerCase().trim();
    let count = 0;
    cards.forEach(card => {
      const match = (category === 'All' || card.dataset.category === category) &&
        (card.dataset.search + ' ' + card.dataset.galleryId.toLowerCase()).includes(query);
      card.hidden = !match;
      if (match) count++;
    });
    categories.forEach(button => {
      const active = button.dataset.category === category;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    byId('gallery-count').textContent = count + ' examples';
    byId('gallery-empty').hidden = count > 0;
  }
  categories.forEach(button => button.addEventListener('click', () => { category = button.dataset.category; filter(); }));
  byId('gallery-search').addEventListener('input', filter);
  filter();
  let gallery;
  let recipe = '';
  cards.forEach(card => card.addEventListener('click', async event => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    try {
      if (!gallery) {
        const response = await fetch('gallery.json');
        if (!response.ok) throw new Error('Recipes unavailable');
        gallery = await response.json();
      }
      const item = gallery.examples.find(example => example.id === card.dataset.galleryId);
      if (!item) throw new Error('Recipe not found');
      byId('figure-title').textContent = item.title;
      byId('figure-image').src = item.image;
      byId('figure-image').alt = item.title + ': ' + item.description;
      byId('figure-description').textContent = item.description;
      recipe = JSON.stringify({ configuration: item.configuration, parameters: item.parameters,
        seed: item.seed, photo: item.photo, ...(item.supervision ? { supervision: item.supervision } : {}),
        device: gallery.device, compile_raster: gallery.compile_raster, torch_version: gallery.torch_version }, null, 2);
      byId('figure-recipe').textContent = recipe;
      byId('figure-download').href = item.image;
      byId('figure-dialog').showModal();
    } catch { location.href = card.href; }
  }));
  byId('copy-recipe').addEventListener('click', () => copy(recipe, byId('copy-recipe'), 'Recipe copied'));
})();
