import { useEffect, useMemo, useState } from 'react';
import { useAppState } from '../state.jsx';
import {
  listGlossaryTerms,
  getGlossaryTerm,
  createGlossaryTerm,
  updateGlossaryTerm,
  deleteGlossaryTerm,
  addGlossarySynonym,
  removeGlossarySynonym,
  upsertGlossaryThreshold,
} from '../api/clients.js';
import { IconGlossary, IconSearch, IconPlus, IconRefresh } from '../components/Icons.jsx';

// Ports the legacy tech_ui Business Glossary (app.js §Business Glossary) to React.
// Backend: /metadata/glossary/* (terms, synonyms, thresholds). Purely additive —
// no shared styles touched; reuses .modal/.form-row/.btn/.search-* from theme.css.

const DIRECTIONS = [
  ['higher_is_better', 'Higher is better'],
  ['lower_is_better', 'Lower is better'],
];

const EMPTY_FORM = {
  name: '', domain: '', definition: '', formula: '', sql_hint: '', owner: '', approved: true,
};
const EMPTY_THR = {
  threshold_red: '', threshold_amber: '', benchmark_value: '',
  benchmark_source: '', unit: '', direction: 'higher_is_better',
};

export default function Glossary() {
  const { toast, refreshTick } = useAppState();
  const [terms, setTerms] = useState(null); // null = loading
  const [search, setSearch] = useState('');
  const [domainFilter, setDomainFilter] = useState('');

  // modal state
  const [modalOpen, setModalOpen] = useState(false);
  const [editId, setEditId] = useState(null);
  const [form, setForm] = useState(EMPTY_FORM);
  const [synonyms, setSynonyms] = useState([]); // [{synonym, domain_scope, synonym_id|null}]
  const [newSyn, setNewSyn] = useState('');
  const [thr, setThr] = useState(EMPTY_THR);
  const [busy, setBusy] = useState(false);

  const load = () => {
    setTerms(null);
    listGlossaryTerms()
      .then((r) => setTerms(Array.isArray(r) ? r : []))
      .catch(() => setTerms([]));
  };

  useEffect(load, [refreshTick]);

  // Distinct domains for the filter dropdown (derived from loaded terms).
  const domains = useMemo(() => {
    const set = new Set((terms || []).map((t) => t.domain).filter(Boolean));
    return Array.from(set).sort();
  }, [terms]);

  // Client-side filter over name / definition / formula / synonyms (matches the
  // server's search surface) plus the domain dropdown.
  const shown = useMemo(() => {
    const list = terms || [];
    const q = search.trim().toLowerCase();
    return list.filter((t) => {
      if (domainFilter && t.domain !== domainFilter) return false;
      if (!q) return true;
      const hay = [
        t.name, t.definition, t.formula,
        ...(t.synonyms || []).map((s) => s.synonym),
      ].join(' ').toLowerCase();
      return hay.includes(q);
    });
  }, [terms, search, domainFilter]);

  const setField = (k, v) => setForm((f) => ({ ...f, [k]: v }));
  const setThrField = (k, v) => setThr((t) => ({ ...t, [k]: v }));

  const openCreate = () => {
    setEditId(null);
    setForm(EMPTY_FORM);
    setSynonyms([]);
    setThr(EMPTY_THR);
    setNewSyn('');
    setModalOpen(true);
  };

  const openEdit = async (termId) => {
    setEditId(termId);
    setForm(EMPTY_FORM);
    setSynonyms([]);
    setThr(EMPTY_THR);
    setNewSyn('');
    setModalOpen(true);
    try {
      const t = await getGlossaryTerm(termId);
      setForm({
        name: t.name || '', domain: t.domain || '', definition: t.definition || '',
        formula: t.formula || '', sql_hint: t.sql_hint || '', owner: t.owner || '',
        approved: !!t.approved,
      });
      setSynonyms((t.synonyms || []).map((s) => ({ ...s })));
      const x = t.threshold;
      if (x) {
        setThr({
          threshold_red: x.threshold_red ?? '', threshold_amber: x.threshold_amber ?? '',
          benchmark_value: x.benchmark_value ?? '', benchmark_source: x.benchmark_source || '',
          unit: x.unit || '', direction: x.direction || 'higher_is_better',
        });
      }
    } catch (e) {
      toast(`Failed to load term: ${e.message}`, 'error');
    }
  };

  const close = () => { if (!busy) setModalOpen(false); };

  const addSyn = () => {
    const v = newSyn.trim();
    if (!v) return;
    setSynonyms((s) => [...s, { synonym: v, domain_scope: '', synonym_id: null }]);
    setNewSyn('');
  };
  const removeSyn = (idx) => setSynonyms((s) => s.filter((_, i) => i !== idx));

  const save = async () => {
    const name = form.name.trim();
    if (!name) { toast('Term name is required', 'warn'); return; }
    setBusy(true);
    try {
      const body = {
        name,
        definition: form.definition.trim(),
        formula: form.formula.trim(),
        sql_hint: form.sql_hint.trim(),
        domain: form.domain.trim(),
        owner: form.owner.trim(),
        approved: form.approved,
      };
      const saved = editId
        ? await updateGlossaryTerm(editId, body)
        : await createGlossaryTerm(body);
      const savedId = saved.term_id;

      // Sync synonyms: delete removed, add new (same logic as legacy submitGlossaryTerm).
      const existingIds = (saved.synonyms || []).map((s) => s.synonym_id);
      const keepIds = synonyms.filter((s) => s.synonym_id).map((s) => s.synonym_id);
      for (const sid of existingIds) {
        if (!keepIds.includes(sid)) {
          await removeGlossarySynonym(sid).catch(() => {});
        }
      }
      for (const s of synonyms) {
        if (!s.synonym_id) {
          await addGlossarySynonym(savedId, s.synonym, s.domain_scope || '').catch(() => {});
        }
      }

      // Upsert threshold only if any numeric value was provided.
      const { threshold_red: red, threshold_amber: amber, benchmark_value: bmark } = thr;
      if (red !== '' || amber !== '' || bmark !== '') {
        await upsertGlossaryThreshold(savedId, {
          threshold_red: red !== '' ? parseFloat(red) : null,
          threshold_amber: amber !== '' ? parseFloat(amber) : null,
          benchmark_value: bmark !== '' ? parseFloat(bmark) : null,
          benchmark_source: thr.benchmark_source || '',
          direction: thr.direction || 'higher_is_better',
          unit: thr.unit || '',
        }).catch(() => {});
      }

      toast(editId ? 'Term updated' : 'Term created', 'success');
      setModalOpen(false);
      load();
    } catch (e) {
      toast(`Save failed: ${e.message}`, 'error');
    } finally {
      setBusy(false);
    }
  };

  const del = async () => {
    if (!editId) return;
    if (!window.confirm(`Delete business term "${form.name}"? This cannot be undone.`)) return;
    setBusy(true);
    try {
      await deleteGlossaryTerm(editId);
      toast('Term deleted', 'info');
      setModalOpen(false);
      load();
    } catch (e) {
      toast(`Delete failed: ${e.message}`, 'error');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div id="view-glossary" className="view active">
      {/* Toolbar */}
      <div style={{ padding: '8px 14px', background: 'var(--bg-2)', borderBottom: '1px solid var(--border)', display: 'flex', gap: 8, alignItems: 'center', flexShrink: 0 }}>
        <span style={{ fontWeight: 600, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
          <IconGlossary width="15" height="15" /> Business Glossary
        </span>
        <span style={{ fontSize: 11, color: 'var(--text-2)' }}>Domain terms, synonyms, thresholds &amp; benchmarks</span>
        <div className="search-wrap" style={{ marginLeft: 'auto' }}>
          <IconSearch />
          <input
            className="search-input"
            placeholder="Search terms, synonyms…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <select
          className="search-input"
          style={{ width: 160 }}
          value={domainFilter}
          onChange={(e) => setDomainFilter(e.target.value)}
        >
          <option value="">All domains</option>
          {domains.map((d) => <option key={d} value={d}>{d}</option>)}
        </select>
        <button className="btn btn-secondary" onClick={load} title="Refresh"><IconRefresh /></button>
        <button className="btn btn-primary" onClick={openCreate}><IconPlus /> Add Term</button>
      </div>

      {/* Term grid */}
      <div style={{ flex: 1, overflowY: 'auto', padding: 14 }}>
        {terms === null ? (
          <div className="empty-state"><span className="spinner" /></div>
        ) : shown.length === 0 ? (
          <div className="empty-state">
            <IconGlossary strokeWidth="1.5" style={{ width: 36, height: 36, opacity: 0.3 }} />
            {terms.length === 0 ? 'No business terms yet — add one to get started.' : 'No terms match your filters.'}
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
            {shown.map((t) => (
              <div
                key={t.term_id}
                onClick={() => openEdit(t.term_id)}
                style={{ cursor: 'pointer', background: 'var(--bg-2)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: 12 }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontWeight: 600, fontSize: 13 }}>
                      {t.name}
                      {t.threshold && <span style={{ fontSize: 10, color: 'var(--accent)', marginLeft: 6 }}>⚡ thresholds</span>}
                    </div>
                    {t.domain && <div style={{ fontSize: 10, color: 'var(--accent)', marginTop: 1 }}>{t.domain}</div>}
                  </div>
                  {!t.approved && <span className="badge badge-amber">draft</span>}
                </div>
                {t.definition && <div style={{ fontSize: 12, color: 'var(--text-2)', marginTop: 6, lineHeight: 1.4 }}>{t.definition}</div>}
                {t.formula && <div style={{ fontSize: 11, color: 'var(--text-2)', marginTop: 4, fontStyle: 'italic' }}>{t.formula}</div>}
                {(t.synonyms || []).length > 0 && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 8 }}>
                    {t.synonyms.map((s) => (
                      <span key={s.synonym_id} style={{ fontSize: 10, background: 'var(--bg-1)', border: '1px solid var(--border)', borderRadius: 10, padding: '1px 7px' }}>
                        {s.synonym}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Add / Edit modal — overlay styled inline to avoid colliding with #modal-overlay */}
      {modalOpen && (
        <div
          style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          onMouseDown={(e) => e.target === e.currentTarget && close()}
        >
          <div className="modal" style={{ width: 540, maxHeight: '90vh', overflowY: 'auto' }}>
            <div className="modal-title">
              <IconGlossary />
              {editId ? 'Edit Business Term' : 'Add Business Term'}
            </div>

            <div className="form-row">
              <div className="form-grid">
                <div>
                  <label>Term Name *</label>
                  <input value={form.name} onChange={(e) => setField('name', e.target.value)} placeholder="e.g. Gross Margin" />
                </div>
                <div>
                  <label>Domain</label>
                  <input list="glossary-domains" value={form.domain} onChange={(e) => setField('domain', e.target.value)} placeholder="Finance" />
                  <datalist id="glossary-domains">
                    {domains.map((d) => <option key={d} value={d} />)}
                  </datalist>
                </div>
              </div>
            </div>

            <div className="form-row">
              <label>Definition</label>
              <textarea rows="2" value={form.definition} onChange={(e) => setField('definition', e.target.value)} placeholder="Human-readable definition" />
            </div>

            <div className="form-row">
              <label>Formula (natural language)</label>
              <input value={form.formula} onChange={(e) => setField('formula', e.target.value)} placeholder="(Revenue − COGS) / Revenue" />
            </div>

            <div className="form-row">
              <label>SQL Hint (optional)</label>
              <input value={form.sql_hint} onChange={(e) => setField('sql_hint', e.target.value)} placeholder="Optional SQL fragment for the planner" />
            </div>

            <div className="form-row">
              <div className="form-grid">
                <div>
                  <label>Owner</label>
                  <input value={form.owner} onChange={(e) => setField('owner', e.target.value)} placeholder="Team / person" />
                </div>
                <div style={{ display: 'flex', alignItems: 'flex-end', gap: 8, paddingBottom: 8 }}>
                  <input id="g-approved" type="checkbox" style={{ width: 'auto' }} checked={form.approved} onChange={(e) => setField('approved', e.target.checked)} />
                  <label htmlFor="g-approved" style={{ margin: 0, textTransform: 'none', letterSpacing: 0 }}>Approved</label>
                </div>
              </div>
            </div>

            {/* Synonyms */}
            <div className="form-row">
              <label>Synonyms</label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 6 }}>
                {synonyms.length === 0 && <span style={{ fontSize: 11, color: 'var(--text-2)' }}>None yet</span>}
                {synonyms.map((s, i) => (
                  <span key={s.synonym_id || `new-${i}`} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 11, background: 'var(--bg-1)', border: '1px solid var(--border)', borderRadius: 10, padding: '2px 8px' }}>
                    {s.synonym}
                    <button onClick={() => removeSyn(i)} style={{ background: 'none', border: 'none', color: 'var(--text-2)', cursor: 'pointer', fontSize: 13, padding: 0, lineHeight: 1 }}>×</button>
                  </span>
                ))}
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <input
                  value={newSyn}
                  onChange={(e) => setNewSyn(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addSyn(); } }}
                  placeholder="Add a synonym and press Enter"
                />
                <button className="btn btn-secondary" onClick={addSyn} type="button">Add</button>
              </div>
            </div>

            {/* Thresholds / benchmarks */}
            <div className="form-row">
              <label>Thresholds &amp; Benchmark (optional)</label>
              <div className="form-grid">
                <div>
                  <label style={{ fontSize: 10 }}>Red below</label>
                  <input type="number" step="any" value={thr.threshold_red} onChange={(e) => setThrField('threshold_red', e.target.value)} />
                </div>
                <div>
                  <label style={{ fontSize: 10 }}>Amber below</label>
                  <input type="number" step="any" value={thr.threshold_amber} onChange={(e) => setThrField('threshold_amber', e.target.value)} />
                </div>
              </div>
              <div className="form-grid" style={{ marginTop: 8 }}>
                <div>
                  <label style={{ fontSize: 10 }}>Benchmark value</label>
                  <input type="number" step="any" value={thr.benchmark_value} onChange={(e) => setThrField('benchmark_value', e.target.value)} />
                </div>
                <div>
                  <label style={{ fontSize: 10 }}>Unit</label>
                  <input value={thr.unit} onChange={(e) => setThrField('unit', e.target.value)} placeholder="%, $, days…" />
                </div>
              </div>
              <div className="form-grid" style={{ marginTop: 8 }}>
                <div>
                  <label style={{ fontSize: 10 }}>Benchmark source</label>
                  <input value={thr.benchmark_source} onChange={(e) => setThrField('benchmark_source', e.target.value)} placeholder="e.g. industry report" />
                </div>
                <div>
                  <label style={{ fontSize: 10 }}>Direction</label>
                  <select value={thr.direction} onChange={(e) => setThrField('direction', e.target.value)}>
                    {DIRECTIONS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                  </select>
                </div>
              </div>
            </div>

            <div className="modal-actions">
              {editId && (
                <button className="btn btn-ghost" style={{ marginRight: 'auto', color: 'var(--red)' }} onClick={del} disabled={busy}>
                  Delete
                </button>
              )}
              <button className="btn btn-ghost" onClick={close} disabled={busy}>Cancel</button>
              <button className="btn btn-primary" onClick={save} disabled={busy}>
                {busy ? 'Saving…' : (editId ? 'Save Changes' : 'Create Term')}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
