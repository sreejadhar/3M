import { useEffect, useRef, useState } from 'react';
import { useAppState } from '../state.jsx';
import {
  sourceEvents,
  reindexSource,
  enrichTaxonomy,
  classifyPII,
  detectBusiness,
} from '../api/clients.js';
import { dbIcon, statusDotClass, fmtRelTime, fmtNum } from '../lib/utils.js';
import { IconRefresh, IconPlus, IconDatabase, IconOntology, IconLock, IconExcel } from '../components/Icons.jsx';

// step → { cls, label, sub } — drives the .ev-stage colour classes in style.css
function stageMeta(step) {
  const map = {
    extract: { cls: 'extract', label: 'EXTRACT' },
    'extract:table': { cls: 'sub', label: '↳ extract', sub: true },
    discover: { cls: 'discover', label: 'DISCOVER' },
    fd: { cls: 'fd', label: 'FD' },
    'fd:table': { cls: 'sub', label: '↳ fd', sub: true },
    ind: { cls: 'ind', label: 'IND' },
    'ind:pair': { cls: 'sub', label: '↳ ind', sub: true },
    cardinality: { cls: 'cardinality', label: 'CARDINALITY' },
    'cardinality:pair': { cls: 'sub', label: '↳ card', sub: true },
    ontology: { cls: 'ontology', label: 'ONTOLOGY' },
    kg: { cls: 'kg', label: 'KG' },
    taxonomy: { cls: 'taxonomy', label: 'TAXONOMY' },
    complete: { cls: 'complete', label: 'COMPLETE' },
    error: { cls: 'error', label: 'ERROR' },
  };
  return map[step] || { cls: 'info', label: (step || 'info').toUpperCase() };
}

const STATUS_ICON = { running: '⟳', done: '✓', error: '✗', warn: '⚠' };

function evTime(ts) {
  const d = ts ? new Date(ts) : new Date();
  return Number.isNaN(d.getTime()) ? new Date().toLocaleTimeString() : d.toLocaleTimeString();
}

export default function PipelineMonitor() {
  const { sources, activeSourceId, setActiveSourceId, refreshSources, bumpRefresh, setAddSourceOpen, toast } =
    useAppState();
  const [events, setEvents] = useState([]);
  const [sseState, setSseState] = useState('idle'); // idle|connecting|open|closed|error
  const [sseKey, setSseKey] = useState(0); // bump to force-reopen SSE for a new pipeline run
  const esRef = useRef(null);
  const logRef = useRef(null);

  const selected = sources.find((s) => s.id === activeSourceId) || null;
  const [predictedBusiness, setPredictedBusiness] = useState(null); // { business, confidence, method } | null
  const [businessLoading, setBusinessLoading] = useState(false);

  // Fetch the ML-predicted business/industry whenever the selected source changes.
  useEffect(() => {
    setPredictedBusiness(null);
    if (!activeSourceId) return;
    setBusinessLoading(true);
    detectBusiness(activeSourceId)
      .then((r) => setPredictedBusiness(r))
      .catch(() => setPredictedBusiness(null))
      .finally(() => setBusinessLoading(false));
  }, [activeSourceId]);

  // Open / re-open the SSE stream when the selected source changes or sseKey bumps.
  useEffect(() => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
    setEvents([]);
    if (!activeSourceId) {
      setSseState('idle');
      return undefined;
    }
    setSseState('connecting');
    const es = sourceEvents(activeSourceId);
    esRef.current = es;
    es.onopen = () => setSseState('open');
    es.onerror = () => setSseState('error');

    // Debounce refreshSources so bursts of SSE events (e.g. replayed queue on
    // reconnect) don't flood the connection pool with concurrent GET /sources.
    let refreshTimer = null;
    const debouncedRefresh = () => {
      if (refreshTimer) clearTimeout(refreshTimer);
      refreshTimer = setTimeout(() => { refreshTimer = null; refreshSources(); }, 800);
    };

    es.onmessage = (e) => {
      let ev;
      try {
        ev = JSON.parse(e.data);
      } catch {
        return;
      }
      if (ev.type === 'heartbeat') return;
      setEvents((prev) => [...prev, { ...ev, _ts: Date.now() }]);
      if (ev.status === 'done' || ev.status === 'error') debouncedRefresh();
      // The classifier retrains in the background after persist — re-fetch the
      // prediction once that step reports done so the badge reflects it live.
      if (ev.step === 'business-classifier' && ev.status === 'done') {
        detectBusiness(activeSourceId).then(setPredictedBusiness).catch(() => {});
      }
      // Only close the EventSource for a *live* complete — not a replayed one from
      // a prior run, which would prematurely close before this run's events arrive.
      if (ev.step === 'complete' && !ev.is_replay) {
        bumpRefresh();
        refreshSources();
        es.close();
        setSseState('closed');
      }
    };
    return () => {
      if (refreshTimer) clearTimeout(refreshTimer);
      es.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSourceId, sseKey]);

  // Auto-scroll the log to the newest event.
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [events]);

  const act = async (fn, label) => {
    if (!activeSourceId) return;
    try {
      // Reopen SSE before the API call so the fresh queue is ready when the
      // backend starts pushing events (clears any closed state from a prior run).
      setSseKey((k) => k + 1);
      await fn(activeSourceId);
      toast(`${label} started`, 'success');
      refreshSources();
    } catch (e) {
      toast(`${label} failed: ${e.message}`, 'error');
    }
  };

  const dotColor = {
    idle: 'var(--text-2)',
    connecting: 'var(--accent)',
    open: 'var(--green)',
    closed: 'var(--green)',
    error: 'var(--red)',
  }[sseState];

  return (
    <div id="view-pipeline" className="view active">
      {/* ── Left: data sources ── */}
      <div id="pipeline-left">
        <div
          className="panel-header"
          style={{ borderRadius: 0, borderLeft: 'none', borderRight: 'none', borderTop: 'none' }}
        >
          <IconDatabase />
          Data Sources
          <div className="panel-actions">
            <button
              className="btn btn-ghost"
              style={{ padding: '2px 6px', fontSize: 10 }}
              onClick={refreshSources}
            >
              <IconRefresh style={{ width: 11, height: 11 }} />
            </button>
          </div>
        </div>

        <div id="source-list" style={{ overflowY: 'auto', flex: 1 }}>
          {sources.length === 0 ? (
            <div className="empty-state" style={{ height: 200 }}>
              <IconDatabase strokeWidth="1.5" />
              <span>No sources registered</span>
            </div>
          ) : (
            sources.map((s) => (
              <div
                key={s.id}
                className={`source-card ${s.id === activeSourceId ? 'selected' : ''}`}
                onClick={() => setActiveSourceId(s.id)}
              >
                <span className="src-icon">{dbIcon(s.db_type)}</span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="src-name">{s.name}</div>
                  <div className="src-meta">
                    {s.db_type} · {fmtNum(s.table_count || 0)} tables
                    {s.domain ? ` · ${s.domain}` : ''}
                  </div>
                  <div className="src-meta" style={{ display: 'flex', alignItems: 'center', gap: 5, marginTop: 3 }}>
                    <span className={`status-dot ${statusDotClass(s.status)}`} />
                    {s.status} · {fmtRelTime(s.indexed_at)}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>

        <div style={{ padding: '10px 12px', borderTop: '1px solid var(--border)' }}>
          <button className="btn btn-primary" style={{ width: '100%' }} onClick={() => setAddSourceOpen(true)}>
            <IconPlus />
            Add Source
          </button>
        </div>
      </div>

      {/* ── Right: detail + events ── */}
      <div id="pipeline-right" style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {selected && (
          <div
            id="pipeline-detail-header"
            style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)', background: 'var(--bg-2)', flexShrink: 0, display: 'block' }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <div>
                <div style={{ fontWeight: 700, fontSize: 14 }}>{selected.name}</div>
                <div style={{ fontSize: 11, color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>
                  {selected.db_type}
                </div>
              </div>
              <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
                <button className="btn btn-secondary" onClick={() => act(reindexSource, 'Reindex')}>
                  <IconRefresh />
                  Reindex
                </button>
                <button className="btn btn-secondary" onClick={() => act(enrichTaxonomy, 'Enrich Taxonomy')}>
                  <IconOntology />
                  Enrich Taxonomy
                </button>
                <button
                  className="btn btn-secondary"
                  style={{ borderColor: '#ff4d4d', color: '#ff4d4d' }}
                  onClick={() => act(classifyPII, 'Classify PII')}
                >
                  <IconLock />
                  Classify PII
                </button>
              </div>
            </div>
            <div id="detail-src-stats" style={{ display: 'flex', gap: 20, marginTop: 10, fontSize: 11, color: 'var(--text-2)' }}>
              <span>📋 {fmtNum(selected.table_count || 0)} tables</span>
              {selected.domain && <span>🏷️ {selected.domain}</span>}
              {businessLoading ? (
                <span>🤖 detecting business…</span>
              ) : predictedBusiness ? (
                <span title={predictedBusiness.method === 'ml' ? 'ML-predicted' : 'Rule-based (model not trained yet)'}>
                  🤖 {predictedBusiness.business}
                  {predictedBusiness.confidence != null && ` (${Math.round(predictedBusiness.confidence * 100)}%)`}
                </span>
              ) : null}
              <span>
                <span className={`status-dot ${statusDotClass(selected.status)}`} style={{ marginRight: 5 }} />
                {selected.status}
              </span>
              <span>🕐 indexed {fmtRelTime(selected.indexed_at)}</span>
            </div>
          </div>
        )}

        <div
          style={{ padding: '8px 16px 4px', background: 'var(--bg-0)', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}
        >
          <span style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Pipeline Events
          </span>
          <span style={{ width: 7, height: 7, borderRadius: '50%', background: dotColor, display: 'inline-block' }} />
          <button
            className="btn btn-ghost"
            style={{ padding: '2px 6px', fontSize: 10, marginLeft: 'auto' }}
            onClick={() => setEvents([])}
          >
            Clear
          </button>
        </div>

        <div id="event-log" className="event-log" ref={logRef} style={{ flex: 1, overflowY: 'auto' }}>
          {!activeSourceId ? (
            <div className="empty-state" style={{ height: 120 }}>
              <span>Select a source to view pipeline events</span>
            </div>
          ) : events.length === 0 ? (
            <div className="empty-state" style={{ height: 120 }}>
              <span>No events yet — trigger a reindex to watch the pipeline.</span>
            </div>
          ) : (
            events.map((ev, i) => {
              const meta = stageMeta(ev.step || ev.stage);
              const sIcon = STATUS_ICON[ev.status] || '';
              return (
                <div key={i} className={`event-row ${meta.sub ? 'event-row-sub' : ''}`}>
                  <span className="ev-time">{evTime(ev.ts || ev._ts)}</span>
                  <span className={`ev-stage ${meta.cls}`}>{meta.label}</span>
                  {sIcon && <span className={`ev-status ev-${ev.status}`}>{sIcon}</span>}
                  <span className="ev-msg">{ev.message || ''}</span>
                  {ev.detail && <div className="ev-detail">{ev.detail}</div>}
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
