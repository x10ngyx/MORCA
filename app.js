'use strict';
const dialog = document.querySelector('#image-dialog');
for (const button of document.querySelectorAll('.image-zoom')) {
  button.addEventListener('click', () => {
    const img = button.querySelector('img');
    document.querySelector('#expanded-image').src = button.dataset.image;
    document.querySelector('#expanded-image').alt = img.alt;
    document.querySelector('#original-image').href = button.dataset.image;
    dialog.showModal();
  });
}
dialog.querySelector('.close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
document.querySelector('#copy-citation').addEventListener('click', async () => {
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(document.querySelector('#bibtex').textContent);
    status.textContent = 'BibTeX copied.';
  } catch {
    status.textContent = 'Copy unavailable in this browser. Select the citation above or download the .bib file.';
  }
});
(async () => {
  try {
    const response = await fetch('assets/results.json');
    if (!response.ok) throw new Error('Results unavailable');
    const data = await response.json();
    let model = 'Wan2.2';
    const select = document.querySelector('#target');
    function render() {
      const rows = data[model][select.value];
      const body = document.querySelector('#result-rows');
      body.replaceChildren();
      const best = [3, 4, 5].map((index) => Math[index === 5 ? 'min' : 'max'](...rows.map(row => Number(row[index]))));
      for (const row of rows) {
        const tr = document.createElement('tr');
        if (row[0] === 'MORCA') tr.className = 'ours';
        row.forEach((value, index) => {
          const cell = document.createElement(index ? 'td' : 'th');
          if (!index) cell.scope = 'row';
          const text = value + (index === 2 ? '×' : '');
          if (index >= 3 && Number(value) === best[index - 3]) {
            const strong = document.createElement('strong'); strong.textContent = text; cell.append(strong);
          } else cell.textContent = text;
          tr.append(cell);
        });
        body.append(tr);
      }
      document.querySelector('#table-caption').textContent = `${model} · target speedup ${select.value}×`;
      document.querySelector('#result-note').textContent = `Original latency: ${model === 'Wan2.2' ? '990.56' : '255.94'} s. Bold indicates the best quality result in this comparison.`;
      document.querySelector('#results-status').textContent = `${model}, target ${select.value} times speedup, five methods shown.`;
    }
    for (const button of document.querySelectorAll('[data-model]')) button.addEventListener('click', () => {
      model = button.dataset.model;
      document.querySelectorAll('[data-model]').forEach(el => el.setAttribute('aria-pressed', String(el === button)));
      select.replaceChildren(...Object.keys(data[model]).map(target => new Option(`${target}×`, target)));
      render();
    });
    select.addEventListener('change', render);
    render();
  } catch {
    document.querySelector('#result-note').textContent = 'Showing Wan2.2 at a 1.8× target. See the paper for all results.';
    document.querySelector('.result-controls').hidden = true;
  }
})();
