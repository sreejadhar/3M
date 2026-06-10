import { useEffect, useRef } from 'react';
import { marked } from 'marked';
import Chart from 'chart.js/auto';

marked.setOptions({ breaks: true, gfm: true });

export function escHtml(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// marked + wrap tables for horizontal scroll (mirrors processInsightCallouts).
export function renderMarkdown(text) {
  let html;
  try {
    html = marked.parse(String(text || ''));
  } catch {
    html = escHtml(text).replace(/\n/g, '<br>');
  }
  html = html.replace(/<table>/g, '<div class="md-table-wrap"><table>').replace(/<\/table>/g, '</table></div>');
  return html;
}

export function formatNumber(v) {
  if (v == null || v === '') return '';
  const n = Number(v);
  if (Number.isNaN(n)) return String(v);
  if (Number.isInteger(n)) return n.toLocaleString();
  return n.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

// Convert {columns, rows} where rows may be arrays or objects → {cols, rowObjs}.
export function normalizeRows(result) {
  const cols = result.columns || [];
  const rows = result.rows || [];
  const rowObjs = rows.map((r) => {
    if (Array.isArray(r)) {
      const o = {};
      cols.forEach((c, i) => { o[c] = r[i]; });
      return o;
    }
    return r;
  });
  return { cols, rows: rowObjs };
}

const ID_RE = /(^id$|_id$|pk|sk|code|uuid|ref|hash|^no$|^num$)/i;
const TIME_RE = /(date|time|month|year|quarter|qtr|week|day|period|fy)/i;

function isNumericCol(rows, col) {
  let seen = 0;
  for (const r of rows) {
    const v = r[col];
    if (v == null || v === '') continue;
    if (Number.isNaN(Number(v))) return false;
    seen++;
  }
  return seen > 0;
}

// Decide whether/how to chart a result. Returns {type, labelCol, numCols} or null.
export function detectChartConfig(cols, rows) {
  if (!rows.length || cols.length < 2) return null;
  const numCols = cols.filter((c) => isNumericCol(rows, c) && !ID_RE.test(c));
  const textCols = cols.filter((c) => !numCols.includes(c));
  if (numCols.length === 0) return null;
  const labelCol = textCols[0] || cols[0];
  // distinct labels required
  const labels = rows.map((r) => r[labelCol]);
  if (new Set(labels).size !== labels.length && rows.length > 1) {
    // duplicate labels → likely detail rows, skip charting
    if (!TIME_RE.test(labelCol)) return null;
  }
  let type;
  if (TIME_RE.test(labelCol)) type = 'line';
  else if (numCols.length === 1 && rows.length <= 6) type = 'doughnut';
  else if (rows.length <= 10) type = 'bar';
  else type = 'barh';
  return { type, labelCol, numCols };
}

const PALETTE = ['#4285F4', '#9B72CB', '#D96570', '#34A853', '#FBBC04', '#FF6D01', '#46BDC6', '#7B61FF'];

export function ChartBlock({ cols, rows, config }) {
  const ref = useRef(null);
  const chartRef = useRef(null);
  useEffect(() => {
    if (!ref.current) return undefined;
    const { type, labelCol, numCols } = config;
    const labels = rows.map((r) => r[labelCol]);
    const isDoughnut = type === 'doughnut';
    const isH = type === 'barh';
    const datasets = numCols.map((c, i) => ({
      label: c,
      data: rows.map((r) => Number(r[c]) || 0),
      backgroundColor: isDoughnut ? PALETTE : PALETTE[i % PALETTE.length],
      borderColor: PALETTE[i % PALETTE.length],
      borderWidth: type === 'line' ? 2 : 0,
      fill: false,
      tension: 0.3,
    }));
    chartRef.current = new Chart(ref.current, {
      type: isDoughnut ? 'doughnut' : type === 'line' ? 'line' : 'bar',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: isH ? 'y' : 'x',
        plugins: { legend: { display: numCols.length > 1 || isDoughnut, labels: { color: '#c7cdd6', font: { size: 11 } } } },
        scales: isDoughnut ? {} : {
          x: { ticks: { color: '#9aa3af', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.06)' } },
          y: { ticks: { color: '#9aa3af', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.06)' } },
        },
      },
    });
    return () => { chartRef.current?.destroy(); chartRef.current = null; };
  }, [cols, rows, config]);
  return <canvas ref={ref} />;
}
