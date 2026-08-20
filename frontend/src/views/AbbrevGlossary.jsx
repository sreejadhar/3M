import { useEffect, useRef, useState } from 'react';
import { useAppState } from '../state.jsx';
import {
  generateSourceAbbrevGlossary,
  listAbbrevGlossarySources,
  listAbbrevGlossaryTerms,
  getAbbrevGlossaryTerm,
  createAbbrevGlossaryTerm,
  updateAbbrevGlossaryTerm,
  approveAbbrevGlossaryTerm,
  rejectAbbrevGlossaryTerm,
  sourceEvents,
} from '../api/clients.js';
import { fmtNum } from '../lib/utils.js';
import { IconGlossary, IconDatabase, IconPlus } from '../components/Icons.jsx';

// Ordered progress stages — matches the step names pushed by
// generate_abbreviations_for_source's progress_cb in abbrev_glossary_generate.py
// (abbrev-glossary:scan -> abbrev-glossary:canonical -> abbrev-glossary:llm_generate
// -> abbrev-glossary:kg-enrich -> done). Mirrors DataGlossary.jsx exactly.
const STAGES = [
  { key: 'scan', label: 'Scan for abbreviation candidates' },
  { key: 'canonical', label: 'Match against known abbreviations' },
  { key: 'llm_generate', label: 'Resolve full forms (LLM)' },
  { key: 'kg-enrich', label: 'Enrich knowledge graph' },
  { key: 'done', label: 'Done' },
];

function StageDot({ state }) {
  const cls = state === 'done' ? 'badge-green' : state === 'running' ? 'badge-amber' : 'badge-gray';
  const icon = state === 'done' ? '✓' : state === 'running' ? '⟳' : '○';
  return <span className={`badge ${cls}`} style={{ minWidth: 20, textAlign: 'center' }}>{icon}</span>;
}

const EMPTY_ADD_FORM = { abbreviation: '', full_form: '', definition: '', domain: '', steward: '' };

function StatusBadge({ status }) {
  const cls = status === 'approved' ? 'badge-green' : status === 'deprecated' ? 'badge-gray' : 'badge-amber';
  return <span className={`badge ${cls}`}>{status}</span>;
}

export default function AbbrevGlossary() {
  const { sources, toast } = useAppState();
  const [runningSourceId, setRunningSourceId] = useState(null);
  const [stageState, setStageState] = useState({});
  const [log, setLog] = useState([]);
  const [summary, setSummary] = useState(null);
  const [terms, setTerms] = useState([]);
  const [termsLoading, setTermsLoading] = useState(false);
  const [sourceCounts, setSourceCounts] = useState({});
  const esRef = useRef(null);
  const logRef = useRef(null);

  // Edit modal state
  const [editId, setEditId] = useState(null);
  const [editForm, setEditForm] = useState(null);
  const [editBusy, setEditBusy] = useState(false);

  // Add modal state
  const [addOpen, setAddOpen] = useState(false);
  const [addForm, setAddForm] = useState(EMPTY_ADD_FORM);
  const [addBusy, setAddBusy] = useState(false);

  const loadSourceCounts = () => {
    listAbbrevGlossarySources()
      .then((rows) => {
        const map = {};
        (Array.isArray(rows) ? rows : []).forEach((r) => { map[r.source_id] = r.term_count; });
        setSourceCounts(map);
      })
      .catch(() => {});
  };

  useEffect(() => {
    loadSourceCounts();
    return () => { if (esRef.current) esRef.current.close(); };
  }, []);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log]);

  const loadTerms = (sourceId) => {
    setTermsLoading(true);
    listAbbrevGlossaryTerms({ sourceId })
      .then((r) => setTerms(Array.isArray(r) ? r : []))
      .catch(() => setTerms([]))
      .finally(() => setTermsLoading(false));
  };

  useEffect(() => {
    if (runningSourceId) loadTerms(runningSourceId);
  }, [runningSourceId]);

  const resetProgress = () => {
    const init = {};
    STAGES.forEach((s) => { init[s.key] = 'pending'; });
    setStageState(init);
    setLog([]);
    setSummary(null);
  };

  const stageKeyFromStep = (step) => {
    if (!step) return null;
    if (step.startsWith('abbrev-glossary:')) return step.slice('abbrev-glossary:'.length);
    if (step === 'abbrev-glossary') return 'done';
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
      const stageKey = stageKeyFromStep(ev.step);
      if (stageKey == null) return;

      setLog((prev) => [...prev, { ...ev, _ts: Date.now() }]);
      setStageState((prev) => ({ ...prev, [stageKey]: ev.status === 'error' ? 'error' : ev.status }));

      if (ev.step === 'abbrev-glossary' && ev.status === 'done') {
        setSummary(ev.message);
        toast(ev.message || 'Abbreviation glossary generated', 'success');
        loadTerms(sourceId);
      }
      if (ev.step === 'abbrev-glossary' && ev.status === 'error') {
        toast(ev.message || 'Abbreviation glossary generation failed', 'error');
      }
      if (ev.step === 'abbrev-glossary-complete') {
        es.close();
        loadTerms(sourceId);
        loadSourceCounts();
      }
    };
    es.onerror = () => es.close();

    try {
      await generateSourceAbbrevGlossary(sourceId);
      toast(`Abbreviation glossary generation started for ${sourceName}`, 'success');
    } catch (e) {
      toast(`Generate abbreviation glossary failed: ${e.message}`, 'error');
    }
  };

  const handleApprove = async (termId) => {
    try {
      await approveAbbrevGlossaryTerm(termId);
      toast('Term approved', 'success');
      loadTerms(runningSourceId);
      loadSourceCounts();
    } catch (e) {
      toast(`Approve failed: ${e.message}`, 'error');
    }
  };

  const handleReject = async (termId) => {
    try {
      await rejectAbbrevGlossaryTerm(termId);
      toast('Term rejected', 'info');
      loadTerms(runningSourceId);
      loadSourceCounts();
    } catch (e) {
      toast(`Reject failed: ${e.message}`, 'error');
    }
  };

  const openEdit = async (termId) => {
    setEditId(termId);
    setEditForm(null);
    try {
      const t = await getAbbrevGlossaryTerm(termId);
      setEditForm({
        abbreviation: t.abbreviation || '',
        full_form: t.full_form || '',
        definition: t.definition || '',
        domain: t.domain || '',
        steward: t.steward || '',
      });
    } catch (e) {
      toast(`Failed to load term: ${e.message}`, 'error');
      setEditId(null);
    }
  };

  const closeEdit = () => { if (!editBusy) { setEditId(null); setEditForm(null); } };

  const setEditField = (k, v) => setEditForm((f) => ({ ...f, [k]: v }));

  const saveEdit = async () => {
    if (!editForm) return;
    if (!editForm.abbreviation.trim() || !editForm.full_form.trim()) {
      toast('Abbreviation and full form are required', 'warn');
      return;
    }
    setEditBusy(true);
    try {
      await updateAbbrevGlossaryTerm(editId, {
        abbreviation: editForm.abbreviation.trim(),
        full_form: editForm.full_form.trim(),
        definition: editForm.definition.trim(),
        domain: editForm.domain.trim(),
        steward: editForm.steward.trim(),
      });
      toast('Term updated', 'success');
      setEditId(null);
      setEditForm(null);
      loadTerms(runningSourceId);
      loadSourceCounts();
    } catch (e) {
      toast(`Update failed: ${e.message}`, 'error');
    } finally {
      setEditBusy(false);
    }
  };

  const openAdd = () => {
    setAddForm(EMPTY_ADD_FORM);
    setAddOpen(true);
  };

  const closeAdd = () => { if (!addBusy) setAddOpen(false); };

  const setAddField = (k, v) => setAddForm((f) => ({ ...f, [k]: v }));

  const saveAdd = async () => {
    if (!addForm.abbreviation.trim() || !addForm.full_form.trim()) {
      toast('Abbreviation and full form are required', 'warn');
      return;
    }
    setAddBusy(true);
    try {
      await createAbbrevGlossaryTerm({
        source_id: runningSourceId,
        abbreviation: addForm.abbreviation.trim(),
        full_form: addForm.full_form.trim(),
        definition: addForm.definition.trim(),
        domain: addForm.domain.trim(),
        steward: addForm.steward.trim(),
      });
      toast('Term added', 'success');
      setAddOpen(false);
      loadTerms(runningSourceId);
      loadSourceCounts();
    } catch (e) {
      toast(`Add failed: ${e.message}`, 'error');
    } finally {
      setAddBusy(false);
    }
  };

  const runningSource = sources.find((s) => s.id === runningSourceId);

  return (
    <div id="view-abbrevglossary" className="view active" style={{ display: 'flex', flexDirection: 'row', height: '100%' }}>
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
                <div style={{ flex: 1, minWidth: 0, cursor: 'pointer' }} onClick={() => setRunningSourceId(s.id)}>
                  <div className="src-name">
                    {s.name}
                    {sourceCounts[s.id] > 0 && (
                      <span className="badge badge-green" style={{ marginLeft: 6, fontSize: 10 }}>
                        {fmtNum(sourceCounts[s.id])} term{sourceCounts[s.id] === 1 ? '' : 's'}
                      </span>
                    )}
                  </div>
                  <div className="src-meta">{s.db_type} · {fmtNum(s.table_count || 0)} tables</div>
                </div>
                <button
                  className="btn btn-secondary"
                  style={{ padding: '4px 8px', fontSize: 11 }}
                  disabled={runningSourceId === s.id && summary == null && log.length > 0}
                  onClick={() => handleGenerate(s.id, s.name)}
                >
                  {runningSourceId === s.id && summary == null && log.length > 0 ? (
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

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        {!runningSourceId ? (
          <div className="empty-state">
            <IconGlossary strokeWidth="1.5" style={{ width: 36, height: 36, opacity: 0.3 }} />
            Select a source, then click Generate to discover its abbreviation glossary
          </div>
        ) : (
          <>
            <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)', background: 'var(--bg-2)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                <div style={{ fontWeight: 700, fontSize: 14 }}>
                  Abbreviation Glossary — {runningSource?.name || runningSourceId}
                </div>
                <button className="btn btn-primary" style={{ fontSize: 11, padding: '4px 10px' }} onClick={openAdd}>
                  <IconPlus style={{ width: 11, height: 11 }} /> Add Term
                </button>
              </div>
              {log.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 10 }}>
                  {STAGES.map((s) => (
                    <div key={s.key} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
                      <StageDot state={stageState[s.key] || 'pending'} />
                      <span>{s.label}</span>
                    </div>
                  ))}
                </div>
              )}
              {summary && (
                <div style={{ marginTop: 10, fontSize: 12, color: 'var(--text-2)' }}>{summary}</div>
              )}
            </div>

            <div style={{ flex: 1, overflowY: 'auto', padding: 14 }}>
              {termsLoading ? (
                <div className="empty-state"><span className="spinner" /></div>
              ) : terms.length === 0 ? (
                <div className="empty-state">
                  <IconGlossary strokeWidth="1.5" style={{ width: 32, height: 32, opacity: 0.3 }} />
                  No abbreviation terms yet for this source — click Generate.
                </div>
              ) : (
                <table className="data-table" style={{ width: '100%', fontSize: 12 }}>
                  <thead>
                    <tr>
                      <th>Abbreviation</th>
                      <th>Full Form</th>
                      <th>Definition</th>
                      <th>Status</th>
                      <th>Confidence</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {terms.map((t) => (
                      <tr key={t.term_id}>
                        <td style={{ fontWeight: 600 }}>{t.abbreviation}</td>
                        <td>{t.full_form}</td>
                        <td style={{ color: 'var(--text-2)' }}>{t.definition}</td>
                        <td><StatusBadge status={t.status} /></td>
                        <td>{t.confidence != null ? t.confidence.toFixed(2) : ''}</td>
                        <td style={{ whiteSpace: 'nowrap' }}>
                          <button className="btn btn-ghost" style={{ fontSize: 11, padding: '2px 6px' }} onClick={() => openEdit(t.term_id)}>Edit</button>
                          {t.status !== 'approved' && (
                            <button className="btn btn-ghost" style={{ fontSize: 11, padding: '2px 6px' }} onClick={() => handleApprove(t.term_id)}>Approve</button>
                          )}
                          {t.status !== 'deprecated' && (
                            <button className="btn btn-ghost" style={{ fontSize: 11, padding: '2px 6px', color: 'var(--red)' }} onClick={() => handleReject(t.term_id)}>Reject</button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>

            {log.length > 0 && (
              <div ref={logRef} style={{ maxHeight: 140, overflow: 'auto', padding: '8px 16px', fontFamily: 'var(--font-mono)', fontSize: 11, borderTop: '1px solid var(--border)' }}>
                {log.map((ev, i) => (
                  <div key={i} style={{ padding: '3px 0', borderBottom: '1px solid var(--border)' }}>
                    <span className="dim">{new Date(ev._ts).toLocaleTimeString()}</span>{' '}
                    <span style={{ fontWeight: 600 }}>{ev.step}</span>{' '}
                    <span className={ev.status === 'error' ? 'dim' : ''}>{ev.message}</span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>

      {editId && (
        <div
          style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          onMouseDown={(e) => e.target === e.currentTarget && closeEdit()}
        >
          <div className="modal" style={{ width: 480, maxHeight: '90vh', overflowY: 'auto' }}>
            <div className="modal-title">
              <IconGlossary />
              Edit Abbreviation Term
            </div>

            {!editForm ? (
              <div className="empty-state"><span className="spinner" /></div>
            ) : (
              <>
                <div className="form-row">
                  <div className="form-grid">
                    <div>
                      <label>Abbreviation *</label>
                      <input value={editForm.abbreviation} onChange={(e) => setEditField('abbreviation', e.target.value)} placeholder="e.g. CLM" />
                    </div>
                    <div>
                      <label>Full Form *</label>
                      <input value={editForm.full_form} onChange={(e) => setEditField('full_form', e.target.value)} placeholder="e.g. Claim" />
                    </div>
                  </div>
                </div>

                <div className="form-row">
                  <label>Definition</label>
                  <textarea rows="2" value={editForm.definition} onChange={(e) => setEditField('definition', e.target.value)} placeholder="Human-readable definition" />
                </div>

                <div className="form-row">
                  <div className="form-grid">
                    <div>
                      <label>Domain</label>
                      <input value={editForm.domain} onChange={(e) => setEditField('domain', e.target.value)} placeholder="Finance" />
                    </div>
                    <div>
                      <label>Steward</label>
                      <input value={editForm.steward} onChange={(e) => setEditField('steward', e.target.value)} placeholder="Team / person" />
                    </div>
                  </div>
                </div>

                <div className="modal-actions">
                  <button className="btn btn-ghost" onClick={closeEdit} disabled={editBusy}>Cancel</button>
                  <button className="btn btn-primary" onClick={saveEdit} disabled={editBusy}>
                    {editBusy ? 'Saving…' : 'Save Changes'}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {addOpen && (
        <div
          style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          onMouseDown={(e) => e.target === e.currentTarget && closeAdd()}
        >
          <div className="modal" style={{ width: 480, maxHeight: '90vh', overflowY: 'auto' }}>
            <div className="modal-title">
              <IconPlus />
              Add Abbreviation Term — {runningSource?.name || runningSourceId}
            </div>

            <div className="form-row">
              <div className="form-grid">
                <div>
                  <label>Abbreviation *</label>
                  <input value={addForm.abbreviation} onChange={(e) => setAddField('abbreviation', e.target.value)} placeholder="e.g. CLM" />
                </div>
                <div>
                  <label>Full Form *</label>
                  <input value={addForm.full_form} onChange={(e) => setAddField('full_form', e.target.value)} placeholder="e.g. Claim" />
                </div>
              </div>
            </div>

            <div className="form-row">
              <label>Definition</label>
              <textarea rows="2" value={addForm.definition} onChange={(e) => setAddField('definition', e.target.value)} placeholder="Human-readable definition" />
            </div>

            <div className="form-row">
              <div className="form-grid">
                <div>
                  <label>Domain</label>
                  <input value={addForm.domain} onChange={(e) => setAddField('domain', e.target.value)} placeholder="Finance" />
                </div>
                <div>
                  <label>Steward</label>
                  <input value={addForm.steward} onChange={(e) => setAddField('steward', e.target.value)} placeholder="Team / person" />
                </div>
              </div>
            </div>

            <div className="modal-actions">
              <button className="btn btn-ghost" onClick={closeAdd} disabled={addBusy}>Cancel</button>
              <button className="btn btn-primary" onClick={saveAdd} disabled={addBusy}>
                {addBusy ? 'Saving…' : 'Add Term'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
