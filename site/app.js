'use strict';
const GUIDE = 'https://github.com/Temporary-Bot-5/DESMAN-P/blob/master/docs/guide-';
const languages = window.I18N || {};
const select = document.getElementById('language');
let current = 'en';

function known(code) {
  return typeof code === 'string' && Object.hasOwn(languages, code);
}

function word(key) {
  const own = known(current) ? languages[current][key] : undefined;
  return own || (known('en') ? languages.en[key] : undefined);
}

function remembered() {
  try { return localStorage.getItem('desman-language'); } catch { return null; }
}

function remember(value) {
  try { localStorage.setItem('desman-language', value); } catch { /* private mode: not kept */ }
}

function chosen() {
  for (const asked of [new URLSearchParams(location.search).get('lang'), remembered()]) {
    if (known(asked)) return asked;
  }
  for (const tag of navigator.languages || [navigator.language || 'en']) {
    const code = tag.toLowerCase().split('-')[0];
    if (known(code)) return code;
  }
  return 'en';
}

function translate(value) {
  current = known(value) ? value : 'en';
  select.value = current;
  document.documentElement.lang = current;
  document.querySelectorAll('[data-i18n]').forEach(node => {
    const text = word(node.dataset.i18n);
    if (text) node.textContent = text; // without i18n.js the English markup stays
  });
  document.querySelectorAll('.guide-link').forEach(node => { node.href = `${GUIDE}${current}.md`; });
  document.getElementById('copy-status').textContent = '';
}

document.getElementById('linux').addEventListener('click', event => {
  const panel = document.getElementById('linux-command');
  panel.hidden = !panel.hidden;
  event.currentTarget.setAttribute('aria-expanded', String(!panel.hidden));
});

document.getElementById('copy').addEventListener('click', async () => {
  const status = document.getElementById('copy-status');
  try {
    await navigator.clipboard.writeText(document.getElementById('command').textContent);
    status.textContent = word('copied') || 'Copied.';
  } catch {
    status.textContent = word('copy_failed') || 'Select and copy the command manually.';
  }
});

select.addEventListener('change', () => {
  remember(select.value);
  const url = new URL(location.href);
  if (url.searchParams.has('lang')) {
    url.searchParams.set('lang', select.value);
    history.replaceState(null, '', url);
  }
  translate(select.value);
});

translate(chosen());
