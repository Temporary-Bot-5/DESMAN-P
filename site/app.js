'use strict';
const language = document.getElementById('language');
function translate(value) {
  document.documentElement.lang = value;
  document.querySelectorAll('[data-en]').forEach(node => { node.textContent = node.dataset[value]; });
  document.getElementById('copy-status').textContent = '';
}
language.value = navigator.language.toLowerCase().startsWith('ru') ? 'ru' : 'en';
translate(language.value);
language.addEventListener('change', () => translate(language.value));
document.getElementById('linux').addEventListener('click', event => {
  const panel = document.getElementById('linux-command');
  panel.hidden = !panel.hidden;
  event.currentTarget.setAttribute('aria-expanded', String(!panel.hidden));
});
document.getElementById('copy').addEventListener('click', async () => {
  const status = document.getElementById('copy-status');
  try {
    await navigator.clipboard.writeText(document.getElementById('command').textContent);
    status.textContent = language.value === 'ru' ? 'Скопировано.' : 'Copied.';
  } catch {
    status.textContent = language.value === 'ru' ? 'Выделите и скопируйте команду вручную.' : 'Select and copy the command manually.';
  }
});
