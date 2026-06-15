import { useEffect, useRef } from 'react';
import { marked } from 'marked';
import Chart from 'chart.js/auto';

marked.setOptions({ breaks: true, gfm: true });

export function escHtml(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// A table cell counts as numeric if it's a bare/currency/percent number.
const NUM_CELL_RE = /^[-+]?[$€£₹]?\s*\d[\d,]*(\.\d+)?\s*%?$/;
function isNumericText(t) {
  const s = String(t).trim();
  if (!s || s === '—' || s === '-' || s === 'N/A') return false;
  return NUM_CELL_RE.test(s);
}

// marked + style tables: wrap for horizontal scroll, tag the table, and
// right-align columns whose body cells are mostly numeric (so currency/percent
// columns line up). Falls back to a plain string wrap if DOMParser is absent.
export function renderMarkdown(text) {
  let html;
  try {
    html = marked.parse(String(text || ''));
  } catch {
    return escHtml(text).replace(/\n/g, '<br>');
  }
  if (typeof window === 'undefined' || !window.DOMParser) {
    return html
      .replace(/<table>/g, '<div class="md-table-wrap"><table class="md-chat-table">')
      .replace(/<\/table>/g, '</table></div>');
  }
  try {
    const doc = new DOMParser().parseFromString(html, 'text/html');
    doc.querySelectorAll('table').forEach((table) => {
      table.classList.add('md-chat-table');
      const headCells = [...(table.querySelector('thead tr')?.children || [])];
      const bodyRows = [...table.querySelectorAll('tbody tr')];
      const ncols = headCells.length || (bodyRows[0]?.children.length || 0);
      for (let ci = 0; ci < ncols; ci++) {
        let num = 0, seen = 0;
        bodyRows.forEach((r) => {
          const cell = r.children[ci];
          if (!cell || !cell.textContent.trim()) return;
          seen++;
          if (isNumericText(cell.textContent)) num++;
        });
        if (seen >= 1 && num / seen >= 0.6) {
          headCells[ci]?.classList.add('col-num');
          bodyRows.forEach((r) => r.children[ci]?.classList.add('col-num'));
        }
      }
      const wrap = doc.createElement('div');
      wrap.className = 'md-table-wrap';
      table.parentNode.insertBefore(wrap, table);
      wrap.appendChild(table);
    });
    return doc.body.innerHTML;
  } catch {
    return html
      .replace(/<table>/g, '<div class="md-table-wrap"><table class="md-chat-table">')
      .replace(/<\/table>/g, '</table></div>');
  }
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
// Columns that should NOT be summed in a totals row (they're rates/ratios).
const NO_SUM_RE = /(pct|percent|avg|average|mean|ratio|rate|margin|score|index|yoy|growth|share|%)/i;

export function isNumericCol(rows, col) {
  let seen = 0;
  for (const r of rows) {
    const v = r[col];
    if (v == null || v === '') continue;
    if (Number.isNaN(Number(v))) return false;
    seen++;
  }
  return seen > 0;
}

export function isNum(v) {
  return v != null && v !== '' && !Number.isNaN(Number(v));
}

// Numeric columns excluding identifiers — the columns worth charting / aggregating.
export function metricColumns(cols, rows) {
  return cols.filter((c) => isNumericCol(rows, c) && !ID_RE.test(c));
}

// Any numeric column (incl. IDs) — used for right-aligning table cells.
export function numericColumns(cols, rows) {
  return cols.filter((c) => isNumericCol(rows, c));
}

export function columnRange(rows, col) {
  let min = Infinity, max = -Infinity;
  for (const r of rows) {
    const n = Number(r[col]);
    if (Number.isFinite(n)) { if (n < min) min = n; if (n > max) max = n; }
  }
  return Number.isFinite(min) ? { min, max } : null;
}

export function canSum(col) {
  return !NO_SUM_RE.test(col);
}

// Columns whose name marks them as a prior-period value (for trend deltas).
export const PREV_RE = /(^prev|_prev|previous|^last|_last|prior|^py$|_py$|^ly$|_ly$|baseline|_pp$)/i;

// Fractional change of curr vs base (0.123 = +12.3%); null if not computable.
export function pctDelta(curr, base) {
  const c = Number(curr), b = Number(base);
  if (!Number.isFinite(c) || !Number.isFinite(b) || b === 0) return null;
  return (c - b) / Math.abs(b);
}

// Best-effort: find a "previous period" sibling column for a metric column,
// e.g. revenue → prev_revenue / revenue_ly / revenue_prior.
export function findPriorColumn(metric, cols) {
  const norm = (s) => String(s).toLowerCase().replace(/[_\s%-]/g, '');
  const base = norm(metric);
  if (!base) return null;
  for (const c of cols) {
    if (c === metric || !PREV_RE.test(c)) continue;
    const stripped = norm(String(c).replace(PREV_RE, ''));
    if (stripped && (base.includes(stripped) || stripped.includes(base))) return c;
  }
  return null;
}

// Largest magnitude (max |value|) reached by a numeric column.
function colMagnitude(rows, col) {
  let m = 0;
  for (const r of rows) { const n = Math.abs(Number(r[col]) || 0); if (n > m) m = n; }
  return m;
}

// Ratio between the biggest and smallest series magnitudes (1 = identical scale).
function magnitudeRatio(rows, numCols) {
  const mags = numCols.map((c) => colMagnitude(rows, c)).filter((m) => m > 0);
  if (mags.length < 2) return 1;
  return Math.max(...mags) / Math.min(...mags);
}

// Decide whether/how to chart a result. Returns a config object or null.
export function detectChartConfig(cols, rows) {
  if (!rows.length || cols.length < 2) return null;
  const numCols = metricColumns(cols, rows);
  const textCols = cols.filter((c) => !numCols.includes(c) && !isNumericCol(rows, c));
  if (numCols.length === 0) return null;

  // No categorical dimension but ≥2 metrics → scatter (relationship / correlation).
  if (textCols.length === 0 && numCols.length >= 2 && rows.length >= 4) {
    return { type: 'scatter', xCol: numCols[0], yCol: numCols[1], labelCol: cols[0], numCols };
  }

  const labelCol = textCols[0] || cols[0];
  const labels = rows.map((r) => r[labelCol]);
  const distinct = new Set(labels).size === labels.length;
  const isTime = TIME_RE.test(labelCol);
  // Duplicate labels on a non-time axis → detail rows, not a chart.
  if (!distinct && rows.length > 1 && !isTime) return null;

  const manySeries = numCols.length > 1;
  const ratio = manySeries ? magnitudeRatio(rows, numCols) : 1;

  // Metrics on very different scales → combo (bars on the left axis, the
  // small-magnitude series as a line on the right axis) so a % / ratio next to
  // dollar amounts stays readable instead of collapsing to a flat sliver.
  if (manySeries && ratio >= 25 && rows.length <= 16) {
    return { type: 'combo', labelCol, numCols };
  }

  if (isTime) return { type: 'line', labelCol, numCols };

  if (!manySeries) {
    if (rows.length <= 6) return { type: 'doughnut', labelCol, numCols };
    if (rows.length <= 12) return { type: 'bar', labelCol, numCols };
    return { type: 'barh', labelCol, numCols };
  }
  // Three or more comparable series over categories read as a composition
  // (part-to-whole) → stacked bars; horizontal once there are many categories.
  if (numCols.length >= 3 && rows.length <= 14) {
    return { type: rows.length <= 8 ? 'stacked' : 'stackedh', labelCol, numCols };
  }
  // Two comparable series → grouped bars (easier side-by-side comparison).
  return { type: rows.length <= 8 ? 'bar' : 'barh', labelCol, numCols };
}

// Professional palette — saturated enough to read on a white surface.
const PALETTE = ['#4285F4', '#34A853', '#FBBC04', '#EA4335', '#A142F4', '#24C1E0', '#FF7043', '#5C6BC0'];
const FONT = "'Google Sans', 'Inter', system-ui, sans-serif";

// Light-theme chart ink (matches the app's design tokens).
const INK = '#202124';
const INK_SUB = '#5f6368';
const GRID = 'rgba(60,64,67,0.10)';

function hexToRgba(hex, a) {
  const h = hex.replace('#', '');
  const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

// Compact number for axes/tooltips/labels: 1.2K, 3.4M, 1.1B.
function compactNumber(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return String(v);
  const abs = Math.abs(n);
  if (abs >= 1e9) return (n / 1e9).toFixed(1).replace(/\.0$/, '') + 'B';
  if (abs >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
  if (abs >= 1e3) return (n / 1e3).toFixed(1).replace(/\.0$/, '') + 'K';
  return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

// Directional gradient; falls back to a flat colour until the chart area exists.
function gradientFactory(color, { horizontal = false, area = false } = {}) {
  return (ctx) => {
    const { chartArea, ctx: c } = ctx.chart;
    if (!chartArea) return area ? hexToRgba(color, 0.18) : color;
    const g = horizontal
      ? c.createLinearGradient(chartArea.left, 0, chartArea.right, 0)
      : c.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
    if (area) { g.addColorStop(0, hexToRgba(color, 0.02)); g.addColorStop(1, hexToRgba(color, 0.28)); }
    else { g.addColorStop(0, hexToRgba(color, 0.65)); g.addColorStop(1, color); }
    return g;
  };
}

// Quick descriptive stats for the analytical subtitle (single-metric charts).
function statsSubtitle(rows, numCols) {
  if (numCols.length !== 1) return '';
  const vals = rows.map((r) => Number(r[numCols[0]])).filter(Number.isFinite);
  if (vals.length < 2) return '';
  const total = vals.reduce((a, b) => a + b, 0);
  const avg = total / vals.length;
  return `Total ${compactNumber(total)}   ·   Avg ${compactNumber(avg)}   ·   Max ${compactNumber(Math.max(...vals))}   ·   Min ${compactNumber(Math.min(...vals))}`;
}

// Inline plugin: print the value at the end of each bar (few bars, single series).
const valueLabelPlugin = {
  id: 'valueLabels',
  afterDatasetsDraw(chart, _args, opts) {
    if (!opts || !opts.enabled) return;
    const { ctx } = chart;
    const horiz = chart.options.indexAxis === 'y';
    chart.data.datasets.forEach((ds, di) => {
      const meta = chart.getDatasetMeta(di);
      if (meta.type !== 'bar' || meta.hidden) return;
      meta.data.forEach((el, i) => {
        const v = ds.data[i];
        if (v == null) return;
        ctx.save();
        ctx.fillStyle = INK_SUB;
        ctx.font = `600 10px ${FONT}`;
        if (horiz) {
          ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
          ctx.fillText(compactNumber(v), el.x + 6, el.y);
        } else {
          ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
          ctx.fillText(compactNumber(v), el.x, el.y - 4);
        }
        ctx.restore();
      });
    });
  },
};

// Inline plugin: centre total inside a doughnut.
const centerTextPlugin = {
  id: 'centerText',
  afterDraw(chart, _args, opts) {
    if (!opts || !opts.text) return;
    const { ctx, chartArea } = chart;
    if (!chartArea) return;
    const x = (chartArea.left + chartArea.right) / 2;
    const y = (chartArea.top + chartArea.bottom) / 2;
    ctx.save();
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = INK; ctx.font = `700 22px ${FONT}`;
    ctx.fillText(opts.text, x, y - 7);
    if (opts.sub) {
      ctx.fillStyle = INK_SUB; ctx.font = `500 11px ${FONT}`;
      ctx.fillText(opts.sub, x, y + 14);
    }
    ctx.restore();
  },
};

export function ChartBlock({ cols, rows, config, title }) {
  const ref = useRef(null);
  const chartRef = useRef(null);
  useEffect(() => {
    if (!ref.current) return undefined;
    const { type, labelCol, numCols } = config;
    const labels = rows.map((r) => r[labelCol]);
    const isDoughnut = type === 'doughnut';
    const isStack = type === 'stacked' || type === 'stackedh';
    const isH = type === 'barh' || type === 'stackedh';
    const isLine = type === 'line';
    const isScatter = type === 'scatter';
    const isCombo = type === 'combo';

    const valueTicks = { color: INK_SUB, font: { family: FONT, size: 10 }, callback: (v) => compactNumber(v) };
    const catTicks = { color: INK_SUB, font: { family: FONT, size: 10 }, autoSkip: true, maxRotation: isH ? 0 : 38, minRotation: 0 };
    const axisTitle = (text) => ({ display: true, text, color: INK_SUB, font: { family: FONT, size: 11, weight: '600' } });

    let chartType = 'bar';
    let datasets = [];
    let scales = {};

    if (isScatter) {
      const { xCol, yCol } = config;
      chartType = 'scatter';
      datasets = [{
        label: `${yCol} vs ${xCol}`,
        data: rows.map((r) => ({ x: Number(r[xCol]), y: Number(r[yCol]) })).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y)),
        backgroundColor: hexToRgba(PALETTE[0], 0.55),
        borderColor: PALETTE[0],
        borderWidth: 1,
        pointRadius: 5,
        pointHoverRadius: 8,
      }];
      scales = {
        x: { type: 'linear', title: axisTitle(xCol), ticks: valueTicks, grid: { color: GRID }, border: { display: false } },
        y: { type: 'linear', title: axisTitle(yCol), ticks: valueTicks, grid: { color: GRID }, border: { display: false } },
      };
    } else if (isCombo) {
      // Split series by magnitude: large ones → bars on the left axis, small
      // ones (< 1/10 of the max) → lines on the right axis. Handles 2..N series.
      const maxMag = Math.max(...numCols.map((c) => colMagnitude(rows, c)));
      const barCols = numCols.filter((c) => colMagnitude(rows, c) >= maxMag / 10);
      const lineCols = numCols.filter((c) => !barCols.includes(c));
      chartType = 'bar';
      datasets = [
        ...barCols.map((c, i) => {
          const color = PALETTE[i % PALETTE.length];
          return {
            type: 'bar', label: c, yAxisID: 'y',
            data: rows.map((r) => Number(r[c]) || 0),
            backgroundColor: gradientFactory(color),
            hoverBackgroundColor: color, borderWidth: 0, borderRadius: 6, borderSkipped: false,
            maxBarThickness: 46, categoryPercentage: 0.7, barPercentage: 0.9, order: 2,
          };
        }),
        ...lineCols.map((c, i) => {
          const color = PALETTE[(barCols.length + i) % PALETTE.length];
          return {
            type: 'line', label: c, yAxisID: 'y1',
            data: rows.map((r) => Number(r[c]) || 0),
            borderColor: color, backgroundColor: hexToRgba(color, 0.12),
            borderWidth: 2.5, tension: 0.35, fill: false,
            pointRadius: 3, pointHoverRadius: 6, pointBackgroundColor: color,
            pointBorderColor: '#fff', pointBorderWidth: 1.5, order: 1,
          };
        }),
      ];
      const leftTitle = barCols.length === 1 ? barCols[0] : 'Amount';
      const rightTitle = lineCols.length === 1 ? lineCols[0] : 'Rate / ratio';
      scales = {
        x: { ticks: catTicks, grid: { color: GRID }, border: { display: false } },
        y: { position: 'left', beginAtZero: true, title: axisTitle(leftTitle), ticks: valueTicks, grid: { color: GRID }, border: { display: false } },
        y1: { position: 'right', title: axisTitle(rightTitle), ticks: valueTicks, grid: { drawOnChartArea: false }, border: { display: false } },
      };
    } else {
      chartType = isDoughnut ? 'doughnut' : isLine ? 'line' : 'bar';
      datasets = numCols.map((c, i) => {
        const color = PALETTE[i % PALETTE.length];
        if (isDoughnut) {
          return {
            label: c,
            data: rows.map((r) => Number(r[c]) || 0),
            backgroundColor: PALETTE.map((p) => hexToRgba(p, 0.92)),
            borderColor: '#ffffff',
            borderWidth: 2,
            hoverOffset: 12,
          };
        }
        if (isLine) {
          return {
            label: c,
            data: rows.map((r) => Number(r[c]) || 0),
            borderColor: color,
            backgroundColor: gradientFactory(color, { area: true }),
            borderWidth: 2.5, fill: numCols.length === 1, tension: 0.35,
            pointRadius: 3, pointHoverRadius: 6,
            pointBackgroundColor: color, pointBorderColor: '#fff', pointBorderWidth: 1.5,
          };
        }
        return {
          label: c,
          data: rows.map((r) => Number(r[c]) || 0),
          backgroundColor: gradientFactory(color, { horizontal: isH }),
          hoverBackgroundColor: color,
          borderWidth: 0, borderRadius: 6, borderSkipped: false,
          maxBarThickness: 46, categoryPercentage: 0.72, barPercentage: 0.9,
        };
      });
      if (!isDoughnut) {
        scales = {
          x: { stacked: isStack, beginAtZero: isH, ticks: isH ? valueTicks : catTicks, grid: { color: GRID }, border: { display: false } },
          y: { stacked: isStack, beginAtZero: !isH, ticks: isH ? catTicks : valueTicks, grid: { color: GRID }, border: { display: false } },
        };
      }
    }

    // Single-series few-bar charts get value labels; doughnut gets a centre total.
    const singleSeries = numCols.length === 1;
    const wantValueLabels = (type === 'bar' || type === 'barh') && singleSeries && rows.length <= 12;
    const doughnutTotal = isDoughnut
      ? rows.reduce((a, r) => a + (Number(r[numCols[0]]) || 0), 0)
      : 0;

    const subtitle = isScatter || isCombo ? '' : statsSubtitle(rows, numCols);

    chartRef.current = new Chart(ref.current, {
      type: chartType,
      data: { labels: isScatter ? undefined : labels, datasets },
      plugins: [valueLabelPlugin, centerTextPlugin],
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: isH ? 'y' : 'x',
        cutout: isDoughnut ? '64%' : undefined,
        layout: { padding: { top: 6, right: wantValueLabels && isH ? 34 : 14, bottom: 2, left: 4 } },
        animation: { duration: 700, easing: 'easeOutQuart' },
        interaction: isDoughnut || isScatter ? { mode: 'nearest', intersect: true } : { mode: 'index', intersect: false },
        plugins: {
          valueLabels: { enabled: wantValueLabels },
          centerText: isDoughnut ? { text: compactNumber(doughnutTotal), sub: 'Total' } : {},
          title: title
            ? { display: true, text: title, color: INK, font: { family: FONT, size: 14, weight: '600' }, padding: { top: 2, bottom: subtitle ? 2 : 10 } }
            : { display: false },
          subtitle: subtitle
            ? { display: true, text: subtitle, color: INK_SUB, font: { family: FONT, size: 11, weight: '500' }, padding: { bottom: 12 } }
            : { display: false },
          legend: {
            display: numCols.length > 1 || isDoughnut || isCombo,
            position: isDoughnut ? 'right' : 'top',
            labels: { color: INK, font: { family: FONT, size: 11 }, usePointStyle: true, pointStyle: 'circle', boxWidth: 8, padding: 14 },
          },
          tooltip: {
            backgroundColor: 'rgba(32,33,36,0.96)',
            borderColor: 'rgba(255,255,255,0.12)',
            borderWidth: 1, cornerRadius: 8, padding: 10,
            titleColor: '#ffffff', bodyColor: '#e8eaed',
            titleFont: { family: FONT, size: 12, weight: '600' },
            bodyFont: { family: FONT, size: 12 },
            usePointStyle: true,
            callbacks: {
              label: (item) => {
                if (isScatter) {
                  const { xCol, yCol } = config;
                  return `  ${xCol}: ${formatNumber(item.parsed.x)}, ${yCol}: ${formatNumber(item.parsed.y)}`;
                }
                const val = item.parsed.y ?? item.parsed.x ?? item.parsed;
                if (isDoughnut) {
                  const tot = item.dataset.data.reduce((a, b) => a + (Number(b) || 0), 0);
                  const pct = tot ? ((Number(val) / tot) * 100).toFixed(1) : '0';
                  return `  ${item.label}: ${formatNumber(val)} (${pct}%)`;
                }
                if (isStack) {
                  const idx = item.dataIndex;
                  const tot = item.chart.data.datasets.reduce((a, d) => a + (Number(d.data[idx]) || 0), 0);
                  const pct = tot ? ((Number(val) / tot) * 100).toFixed(1) : '0';
                  return `  ${item.dataset.label}: ${formatNumber(val)} (${pct}%)`;
                }
                return `  ${item.dataset.label}: ${formatNumber(val)}`;
              },
            },
          },
        },
        scales: isDoughnut ? {} : scales,
      },
    });
    return () => { chartRef.current?.destroy(); chartRef.current = null; };
  }, [cols, rows, config, title]);
  return <canvas ref={ref} />;
}
