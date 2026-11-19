'use strict';
const GUIDE = 'https://github.com/Temporary-Bot-5/DESMAN-P/blob/master/docs/guide-';
const languages = window.I18N || {};
const select = document.getElementById('language');
let current = 'en';

function word(key) {
  return (languages[current] && languages[current][key]) || languages.en[key];
}

function remembered() {
  try { return localStorage.getItem('desman-language'); } catch { return null; }
}

function remember(value) {
  try { localStorage.setItem('desman-language', value); } catch { /* private mode: not kept */ }
}

function chosen() {
  const asked = new URLSearchParams(location.search).get('lang') || remembered();
  if (asked && languages[asked]) return asked;
  for (const tag of navigator.languages || [navigator.language || 'en']) {
    const code = tag.toLowerCase().split('-')[0];
    if (languages[code]) return code;
  }
  return 'en';
}

function translate(value) {
  current = languages[value] ? value : 'en';
  select.value = current;
  document.documentElement.lang = current;
  document.querySelectorAll('[data-i18n]').forEach(node => { node.textContent = word(node.dataset.i18n); });
  document.querySelectorAll('.guide-link').forEach(node => { node.href = `${GUIDE}${current}.md`; });
  document.getElementById('copy-status').textContent = '';
}

translate(chosen());
select.addEventListener('change', () => { remember(select.value); translate(select.value); });

document.getElementById('linux').addEventListener('click', event => {
  const panel = document.getElementById('linux-command');
  panel.hidden = !panel.hidden;
  event.currentTarget.setAttribute('aria-expanded', String(!panel.hidden));
});

document.getElementById('copy').addEventListener('click', async () => {
  const status = document.getElementById('copy-status');
  try {
    await navigator.clipboard.writeText(document.getElementById('command').textContent);
    status.textContent = word('copied');
  } catch {
    status.textContent = word('copy_failed');
  }
});
