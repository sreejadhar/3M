import { useState } from 'react';
import { renderMarkdown, escHtml, normalizeRows, detectChartConfig, formatNumber, ChartBlock } from '../lib/render.jsx';
import { exportExcel } from '../api.js';

const TYPE_ICON = { line: '📈', doughnut: '🍩', bar: '📊', barh: '📊', kpi: '🎯' };

function DataTable({ cols, rows }) {
  const shown = rows.slice(0, 50);
  return (
    <div className="result-table-wrap">
      <table className="result-table">
        <thead>
          <tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {shown.map((r, i) => (
            <tr key={i}>{cols.map((c) => <td key={c}>{formatNumber(r[c]) || (r[c] ?? '')}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function KpiCards({ cols, row }) {
  const nums = cols.filter((c) => row[c] != null && !Number.isNaN(Number(row[c])));
  return (
    <div className="kpi-grid">
      {nums.map((c) => (
        <div className="kpi-card" key={c}>
          <div className="kpi-card-value">{formatNumber(row[c])}</div>
          <div className="kpi-card-label">{c}</div>
        </div>
      ))}
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
          <div className="chart-wrap" style={{ height: 300 }}>
            <ChartBlock cols={cols} rows={rows} config={config} />
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

function SqlDisclosure({ sql, open }) {
  const [show, setShow] = useState(open);
  return (
    <div className="sql-disclosure">
      <button className="sql-toggle" onClick={() => setShow((s) => !s)}>
        <span className="sql-toggle-icon">{show ? '▼' : '▶'}</span> {sql.length} SQL quer{sql.length === 1 ? 'y' : 'ies'}
      </button>
      {show && (
        <div className="sql-block">
          {sql.map((q, i) => (
            <div key={i}>
              {q.query_label && <div className="sql-label">{q.query_label}</div>}
              <pre>{q.sql}</pre>
            </div>
          ))}
        </div>
      )}
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
  const hasData = results.some((r) => (r.rows || []).length);

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
        {msg.cache_hit && <div className="cache-note">⚡ Served from cache</div>}
        <div className="insight-action-bar">
          <span className="insight-action-label">Export:</span>
          <button className="insight-action-btn" onClick={() => window.print()}>PDF</button>
          <button className="insight-action-btn" onClick={doExcel} disabled={!hasData}>Excel</button>
        </div>
      </div>
    </div>
  );
}
