import { useEffect, useRef, useState } from 'react';
import { useAppState } from '../state.jsx';
import { generateSourceGlossary, sourceEvents } from '../api/clients.js';
import { fmtNum } from '../lib/utils.js';
import { IconGlossary, IconDatabase } from '../components/Icons.jsx';

// Ordered progress stages — matches the step names pushed by
// generate_glossary_for_source's progress_cb in glossary_generate.py
// (glossary:normalize -> glossary:cross_source -> glossary:llm_generate -> done).
const STAGES = [
  { key: 'normalize', label: 'Normalize columns' },
  { key: 'cross_source', label: 'Match against other sources' },
  { key: 'llm_generate', label: 'Generate remaining terms (LLM)' },
  { key: 'done', label: 'Done' },
];

function StageDot({ state }) {
  const cls = state === 'done' ? 'badge-green' : state === 'running' ? 'badge-amber' : 'badge-gray';
  const icon = state === 'done' ? '✓' : state === 'running' ? '⟳' : '○';
  return <span className={`badge ${cls}`} style={{ minWidth: 20, textAlign: 'center' }}>{icon}</span>;
}

export default function DataGlossary() {
  const { sources, toast } = useAppState();
  const [runningSourceId, setRunningSourceId] = useState(null);
  const [stageState, setStageState] = useState({}); // stage key -> 'pending'|'running'|'done'
  const [log, setLog] = useState([]); // raw event log for the running source
  const [summary, setSummary] = useState(null);
  const esRef = useRef(null);
  const logRef = useRef(null);

  useEffect(() => {
    return () => { if (esRef.current) esRef.current.close(); };
  }, []);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log]);

  const resetProgress = () => {
    const init = {};
    STAGES.forEach((s) => { init[s.key] = 'pending'; });
    setStageState(init);
    setLog([]);
    setSummary(null);
  };

  const stageKeyFromStep = (step) => {
    if (!step) return null;
    if (step.startsWith('glossary:')) return step.slice('glossary:'.length);
    if (step === 'glossary') return 'done';
    return null;
  };

  const handleGenerate = async (sourceId, sourceName) => {
    if (esRef.current) esRef.current.close();
    resetProgress();
    setRunningSourceId(sourceId);

    const es = sourceEvents(sourceId);
    esRef.current = es;
    es.onmessage = (e) => {
      let ev;
      try { ev = JSON.parse(e.data); } catch { return; }
      if (ev.type === 'heartbeat') return;
      // This view only cares about glossary-namespaced events on the shared
      // per-source event bus — anything else (reindex, enrich-taxonomy, PII
      // classification) belongs to Pipeline Monitor, not here.
      const stageKey = stageKeyFromStep(ev.step);
      if (stageKey == null) return;

      setLog((prev) => [...prev, { ...ev, _ts: Date.now() }]);
      setStageState((prev) => ({ ...prev, [stageKey]: ev.status === 'error' ? 'error' : ev.status }));

      if (ev.step === 'glossary' && ev.status === 'done') {
        setSummary(ev.message);
        toast(ev.message || 'Business glossary generated', 'success');
      }
      if (ev.step === 'glossary' && ev.status === 'error') {
        toast(ev.message || 'Business glossary generation failed', 'error');
      }
      if (ev.step === 'glossary-complete') {
        es.close();
      }
    };
    es.onerror = () => es.close();

    try {
      await generateSourceGlossary(sourceId);
      toast(`Business glossary generation started for ${sourceName}`, 'success');
    } catch (e) {
      toast(`Generate glossary failed: ${e.message}`, 'error');
    }
  };

  const runningSource = sources.find((s) => s.id === runningSourceId);

  return (
    <div id="view-dataglossary" className="view active" style={{ display: 'flex', flexDirection: 'row', height: '100%' }}>
      <div style={{ width: 340, display: 'flex', flexDirection: 'column', borderRight: '1px solid var(--border)' }}>
        <div className="panel-header" style={{ borderRadius: 0, border: 'none', borderBottom: '1px solid var(--border)' }}>
          <IconDatabase />
          Data Sources
        </div>
        <div style={{ overflowY: 'auto', flex: 1 }}>
          {sources.length === 0 ? (
            <div className="empty-state" style={{ height: 200 }}>
              <IconDatabase strokeWidth="1.5" />
              <span>No sources registered</span>
            </div>
          ) : (
            sources.map((s) => (
              <div key={s.id} className={`source-card ${s.id === runningSourceId ? 'selected' : ''}`}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="src-name">{s.name}</div>
                  <div className="src-meta">{s.db_type} · {fmtNum(s.table_count || 0)} tables</div>
                </div>
                <button
                  className="btn btn-secondary"
                  style={{ padding: '4px 8px', fontSize: 11 }}
                  disabled={runningSourceId === s.id && summary == null}
                  onClick={() => handleGenerate(s.id, s.name)}
                >
                  {runningSourceId === s.id && summary == null ? (
                    <span className="spinner" style={{ width: 11, height: 11 }} />
                  ) : (
                    <IconGlossary style={{ width: 11, height: 11 }} />
                  )}
                  Generate
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {!runningSourceId ? (
          <div className="empty-state">
            <IconGlossary strokeWidth="1.5" style={{ width: 36, height: 36, opacity: 0.3 }} />
            Click Generate next to a source to discover and govern its business glossary
          </div>
        ) : (
          <>
            <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)', background: 'var(--bg-2)' }}>
              <div style={{ fontWeight: 700, fontSize: 14 }}>
                Progress for {runningSource?.name || runningSourceId}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 10 }}>
                {STAGES.map((s) => (
                  <div key={s.key} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
                    <StageDot state={stageState[s.key] || 'pending'} />
                    <span>{s.label}</span>
                  </div>
                ))}
              </div>
              {summary && (
                <div style={{ marginTop: 10, fontSize: 12, color: 'var(--text-2)' }}>{summary}</div>
              )}
            </div>
            <div ref={logRef} style={{ flex: 1, overflow: 'auto', padding: '8px 16px', fontFamily: 'var(--font-mono)', fontSize: 11 }}>
              {log.length === 0 ? (
                <div className="dim">Waiting for progress events…</div>
              ) : (
                log.map((ev, i) => (
                  <div key={i} style={{ padding: '3px 0', borderBottom: '1px solid var(--border)' }}>
                    <span className="dim">{new Date(ev._ts).toLocaleTimeString()}</span>{' '}
                    <span style={{ fontWeight: 600 }}>{ev.step}</span>{' '}
                    <span className={ev.status === 'error' ? 'dim' : ''}>{ev.message}</span>
                  </div>
                ))
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
