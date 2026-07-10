import { useState } from 'react';
import {
  renderMarkdown, escHtml, normalizeRows, detectChartConfig, formatNumber, ChartBlock,
  numericColumns, metricColumns, columnRange, canSum, isNum, pctDelta, findPriorColumn,
} from '../lib/render.jsx';
import { exportExcel } from '../api.js';

const TYPE_ICON = {
  line: '📈', doughnut: '🍩', bar: '📊', barh: '📊', stacked: '📊', stackedh: '📊',
  combo: '📊', scatter: '⚬', kpi: '🎯',
};

// Up/down delta pill: green when the value rose, red when it fell.
function TrendBadge({ delta }) {
  if (delta == null) return null;
  const up = delta >= 0;
  return (
    <span className={`trend-badge ${up ? 'trend-up' : 'trend-down'}`}>
      {up ? '▲' : '▼'} {(Math.abs(delta) * 100).toFixed(1)}%
    </span>
  );
}

// Hero KPI for a single metric over time: latest value + Δ vs prev / vs first.
function KpiHero({ rows, metric, labelCol }) {
  const series = rows.map((r) => Number(r[metric])).filter(Number.isFinite);
  if (series.length < 2) return null;
  const latest = series[series.length - 1];
  const dPrev = pctDelta(latest, series[series.length - 2]);
  const dFirst = pctDelta(latest, series[0]);
  const lastLabel = rows[rows.length - 1]?.[labelCol];
  return (
    <div className="kpi-hero">
      <div className="kpi-hero-main">
        <div className="kpi-hero-label">{metric}{lastLabel != null ? ` · ${lastLabel}` : ''}</div>
        <div className="kpi-hero-value">{formatNumber(latest)}</div>
      </div>
      <div className="kpi-hero-deltas">
        {dPrev != null && (
          <div className="kpi-hero-delta"><TrendBadge delta={dPrev} /><span className="kpi-hero-delta-cap">vs prev</span></div>
        )}
        {dFirst != null && (
          <div className="kpi-hero-delta"><TrendBadge delta={dFirst} /><span className="kpi-hero-delta-cap">vs first</span></div>
        )}
      </div>
    </div>
  );
}

// Subtle in-cell magnitude bar for the primary metric column.
function dataBarStyle(v, range) {
  if (!range || range.max === range.min || !isNum(v)) return undefined;
  const pct = Math.max(0, Math.min(1, (Number(v) - range.min) / (range.max - range.min))) * 100;
  return { background: `linear-gradient(90deg, rgba(66,133,244,0.14) ${pct}%, transparent ${pct}%)` };
}

function DataTable({ cols, rows }) {
  const shown = rows.slice(0, 50);
  const numSet = new Set(numericColumns(cols, rows));   // right-align these
  const metrics = metricColumns(cols, rows);            // chartable/aggregatable
  const primary = metrics[0];                           // gets the data bar
  const primaryRange = primary ? columnRange(rows, primary) : null;
  const showTotals = rows.length > 1 && metrics.length > 0;

  return (
    <div className="result-table-wrap">
      <table className="result-table">
        <thead>
          <tr>{cols.map((c) => (
            <th key={c} className={numSet.has(c) ? 'num-cell' : undefined}>{c}</th>
          ))}</tr>
        </thead>
        <tbody>
          {shown.map((r, i) => (
            <tr key={i}>
              {cols.map((c) => {
                const numeric = numSet.has(c);
                const style = c === primary ? dataBarStyle(r[c], primaryRange) : undefined;
                return (
                  <td key={c} className={numeric ? 'num-cell' : undefined} style={style}>
                    {numeric ? (formatNumber(r[c]) || '0') : (r[c] ?? '')}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
        {showTotals && (
          <tfoot>
            <tr>
              {cols.map((c, idx) => {
                if (idx === 0 && !metrics.includes(c)) return <td key={c} className="total-label">Total</td>;
                if (metrics.includes(c) && canSum(c)) {
                  const sum = rows.reduce((a, r) => a + (Number(r[c]) || 0), 0);
                  return <td key={c} className="num-cell total-val">{formatNumber(sum)}</td>;
                }
                return <td key={c} className={numSet.has(c) ? 'num-cell' : undefined}>—</td>;
              })}
            </tr>
          </tfoot>
        )}
      </table>
    </div>
  );
}

function KpiCards({ cols, row }) {
  const nums = cols.filter((c) => isNum(row[c]));
  // Columns that are themselves a prior-period value are used only as the
  // comparison baseline, not shown as their own card.
  const priorCols = new Set(nums.map((c) => findPriorColumn(c, cols)).filter(Boolean));
  const shown = nums.filter((c) => !priorCols.has(c));
  return (
    <div className="kpi-grid">
      {shown.map((c) => {
        const prior = findPriorColumn(c, cols);
        const delta = prior ? pctDelta(row[c], row[prior]) : null;
        return (
          <div className="kpi-card" key={c}>
            <div className="kpi-card-value">{formatNumber(row[c])}</div>
            <div className="kpi-card-label">{c}</div>
            {delta != null && <TrendBadge delta={delta} />}
          </div>
        );
      })}
    </div>
  );
}

function ResultBlock({ result, idx }) {
  const [showData, setShowData] = useState(false);
  const { cols, rows } = normalizeRows(result);
  const rowCount = result.row_count ?? rows.length;
  const config = detectChartConfig(cols, rows);
  const singleRowKpi = rows.length === 1 && cols.filter((c) => !Number.isNaN(Number(rows[0][c]))).length >= 2;

  return (
    <div className="result-block">
      <div className="result-block-header">
        <span className="result-block-icon">{config ? TYPE_ICON[config.type] : '📋'}</span>
        <span className="result-block-title">{result.description || `Result ${idx + 1}`}</span>
        <span className="result-row-badge">{rowCount} rows</span>
      </div>

      {singleRowKpi ? (
        <KpiCards cols={cols} row={rows[0]} />
      ) : config ? (
        <>
          {config.type === 'line' && config.numCols.length === 1 && (
            <KpiHero rows={rows} metric={config.numCols[0]} labelCol={config.labelCol} />
          )}
          <div className="chart-wrap" style={{ height: 340 }}>
            <ChartBlock cols={cols} rows={rows} config={config} title={result.description || ''} />
          </div>
          <div className="result-data-footer">
            <button className="data-toggle-btn" onClick={() => setShowData((s) => !s)}>
              <span className="data-toggle-icon">{showData ? '▼' : '▶'}</span> {showData ? 'Hide' : 'Show'} data
            </button>
          </div>
          {showData && <DataTable cols={cols} rows={rows} />}
        </>
      ) : (
        <DataTable cols={cols} rows={rows} />
      )}
      {rowCount > 50 && <div className="result-row-note">Showing first 50 of {rowCount} rows</div>}
    </div>
  );
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);
  const copy = (e) => {
    e.stopPropagation();
    const done = () => { setCopied(true); setTimeout(() => setCopied(false), 2000); };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done).catch(() => fallback());
    } else {
      fallback();
    }
    function fallback() {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.cssText = 'position:fixed;opacity:0;top:0;left:0';
      document.body.appendChild(ta);
      ta.focus(); ta.select();
      try { document.execCommand('copy'); done(); } catch {}
      document.body.removeChild(ta);
    }
  };
  return (
    <button className="sql-copy-btn" onClick={copy} title="Copy SQL">
      {copied ? (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="20 6 9 17 4 12" /></svg>
      ) : (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="9" y="9" width="13" height="13" rx="2" /><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" /></svg>
      )}
    </button>
  );
}

function SqlDisclosure({ sql, open }) {
  const [show, setShow] = useState(open);
  return (
    <div className="sql-disclosure">
      <button className="sql-toggle" onClick={() => setShow((s) => !s)}>
        <span className="sql-toggle-icon">{show ? '▼' : '▶'}</span> {sql.length} SQL quer{sql.length === 1 ? 'y' : 'ies'}
      </button>
      {show && (
        <div className="sql-block visible">
          {sql.map((q, i) => (
            <div key={i} className="sql-query-wrap">
              {q.query_label && <div className="sql-label">{q.query_label}</div>}
              <div className="sql-pre-wrap">
                <CopyButton text={q.sql} />
                <pre>{q.sql}</pre>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SourcesNote({ sources }) {
  const tables = sources.filter((s) => s.type === 'table');
  const documents = sources.filter((s) => s.type === 'document');
  return (
    <div className="sources-note">
      <span className="sources-label">Sources:</span>
      {tables.map((s, i) => (
        <span key={`t-${i}`} className="source-chip source-chip-table" title="Database table">🗄️ {s.name}</span>
      ))}
      {documents.map((s, i) => (
        <span key={`d-${i}`} className="source-chip source-chip-document" title="Document">📄 {s.name}</span>
      ))}
    </div>
  );
}

export default function Message({ msg, showSQL }) {
  if (msg.role === 'user') {
    return (
      <div className="msg-row user" id={`msg-${msg.id}`}>
        <div className="msg-avatar">U</div>
        <div className="msg-bubble" dangerouslySetInnerHTML={{ __html: escHtml(msg.content).replace(/\n/g, '<br>') }} />
      </div>
    );
  }

  if (msg.error) {
    return (
      <div className="msg-row assistant" id={`ai-msg-${msg.id}`}>
        <div className="msg-avatar">⬡</div>
        <div className="msg-bubble">
          <div className="ai-error">⚠ {msg.error}</div>
        </div>
      </div>
    );
  }

  const results = msg.results || [];
  const sql = msg.sql || [];
  const errors = msg.errors || [];
  const sources = msg.sources || [];
  const hasData = results.some((r) => (r.rows || []).length);

  const doPdf = () => {
    const el = document.getElementById(`ai-msg-${msg.id}`);
    if (!el) { window.print(); return; }
    el.classList.add('print-target');
    document.body.classList.add('printing');
    const cleanup = () => {
      el.classList.remove('print-target');
      document.body.classList.remove('printing');
      window.removeEventListener('afterprint', cleanup);
    };
    window.addEventListener('afterprint', cleanup);
    // Fallback in case afterprint never fires (some browsers/headless).
    setTimeout(cleanup, 60000);
    window.print();
  };

  const doExcel = async () => {
    try {
      const blob = await exportExcel({ title: 'DataChat Insight', results });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'insight.xlsx'; a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch { /* ignore */ }
  };

  return (
    <div className="msg-row assistant" id={`ai-msg-${msg.id}`}>
      <div className="msg-avatar">⬡</div>
      <div className="msg-bubble">
        {msg.content && <div className="md-content" dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }} />}
        {results.map((r, i) => <ResultBlock key={i} result={r} idx={i} />)}
        {errors.length > 0 && (
          <div className="ai-error-notes">
            {errors.map((e, i) => <div key={i} className="ai-error-note">⚠ {e}</div>)}
          </div>
        )}
        {sql.length > 0 && <SqlDisclosure sql={sql} open={showSQL} />}
        {sources.length > 0 && <SourcesNote sources={sources} />}
        {msg.cache_hit && <div className="cache-note">⚡ Served from cache</div>}
        <div className="insight-action-bar">
          <span className="insight-action-label">Export:</span>
          <button className="insight-action-btn" onClick={doPdf}>PDF</button>
          <button className="insight-action-btn" onClick={doExcel} disabled={!hasData}>Excel</button>
        </div>
      </div>
    </div>
  );
}
