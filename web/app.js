async function loadPeople() {
  const res = await fetch('./data/people.json', { cache: 'no-store' });
  if (!res.ok) throw new Error(`Falha ao carregar people.json: ${res.status}`);
  return await res.json();
}

function normalizeCategory(v) {
  if (!v) return 'desconhecido';
  const s = String(v).trim();
  if (!s) return 'desconhecido';
  return s;
}

function countsBy(people, field, weightMode) {
  const counts = new Map();
  for (const p of people) {
    const key = normalizeCategory(p[field]);
    const w = weightMode === 'mentions' ? Number(p.mentions || 0) : 1;
    counts.set(key, (counts.get(key) || 0) + w);
  }

  // Sort: put "desconhecido" last, then by count desc
  const rows = Array.from(counts.entries()).map(([k, v]) => ({ k, v }));
  rows.sort((a, b) => {
    if (a.k === 'desconhecido' && b.k !== 'desconhecido') return 1;
    if (b.k === 'desconhecido' && a.k !== 'desconhecido') return -1;
    if (b.v !== a.v) return b.v - a.v;
    return a.k.localeCompare(b.k);
  });

  return rows;
}

function makeBarChart(ctx, rows, label) {
  const labels = rows.map(r => r.k);
  const data = rows.map(r => r.v);

  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label,
          data,
          backgroundColor: labels.map(l => (l === 'desconhecido' ? 'rgba(148, 163, 184, 0.35)' : 'rgba(245, 158, 11, 0.55)')),
          borderColor: labels.map(l => (l === 'desconhecido' ? 'rgba(148, 163, 184, 0.6)' : 'rgba(245, 158, 11, 0.85)')),
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.parsed.y}`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: 'rgba(233, 238, 248, 0.85)' },
          grid: { color: 'rgba(255,255,255,0.06)' },
        },
        y: {
          ticks: { color: 'rgba(233, 238, 248, 0.85)' },
          grid: { color: 'rgba(255,255,255,0.06)' },
        },
      },
    },
  });
}

function renderTable(people) {
  const tbody = document.querySelector('#peopleTable tbody');
  tbody.innerHTML = '';

  for (const p of people) {
    const tr = document.createElement('tr');
    const tdName = document.createElement('td');
    tdName.textContent = p.name;

    const tdMentions = document.createElement('td');
    tdMentions.textContent = String(p.mentions ?? '');

    const tdRaca = document.createElement('td');
    tdRaca.textContent = normalizeCategory(p.raca);

    const tdClasse = document.createElement('td');
    tdClasse.textContent = normalizeCategory(p.classe);

    const tdGenero = document.createElement('td');
    tdGenero.textContent = normalizeCategory(p.genero);

    const tdOri = document.createElement('td');
    tdOri.textContent = normalizeCategory(p.orientacao_sexual);

    const tdImgs = document.createElement('td');
    tdImgs.textContent = (p.images || []).join(', ');

    tr.appendChild(tdName);
    tr.appendChild(tdMentions);
    tr.appendChild(tdRaca);
    tr.appendChild(tdClasse);
    tr.appendChild(tdGenero);
    tr.appendChild(tdOri);
    tr.appendChild(tdImgs);
    tbody.appendChild(tr);
  }
}

async function main() {
  const people = await loadPeople();
  const totalPeople = people.length;
  const totalMentions = people.reduce((acc, p) => acc + Number(p.mentions || 0), 0);

  const chip = document.getElementById('summaryChip');
  chip.textContent = `${totalPeople} pessoas • ${totalMentions} menções`;

  const weightSelect = document.getElementById('weight');

  const charts = {
    raca: null,
    classe: null,
    genero: null,
    orientacao: null,
  };

  function rerender() {
    const mode = weightSelect.value;

    const racaRows = countsBy(people, 'raca', mode);
    const classeRows = countsBy(people, 'classe', mode);
    const generoRows = countsBy(people, 'genero', mode);
    const oriRows = countsBy(people, 'orientacao_sexual', mode);

    for (const k of Object.keys(charts)) {
      if (charts[k]) {
        charts[k].destroy();
        charts[k] = null;
      }
    }

    charts.raca = makeBarChart(document.getElementById('chartRaca'), racaRows, 'Raça');
    charts.classe = makeBarChart(document.getElementById('chartClasse'), classeRows, 'Classe');
    charts.genero = makeBarChart(document.getElementById('chartGenero'), generoRows, 'Gênero');
    charts.orientacao = makeBarChart(document.getElementById('chartOrientacao'), oriRows, 'Orientação sexual');
  }

  renderTable(people);
  rerender();

  weightSelect.addEventListener('change', rerender);
}

main().catch((err) => {
  console.error(err);
  const chip = document.getElementById('summaryChip');
  if (chip) chip.textContent = 'Erro ao carregar dados';
});
