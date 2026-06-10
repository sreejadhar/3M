import { useEffect, useMemo, useState } from 'react';
import { useAppState } from '../state.jsx';
import {
  listKpis, getKpi, createKpi, updateKpi, deleteKpi,
  compileKpi, listKpiVersions, rollbackKpiVersion, activateKpi,
  listEntities, getEntity,
} from '../api/clients.js';
import { IconKpi, IconSearch, IconPlus, IconRefresh } from '../components/Icons.jsx';

// Ports the legacy tech_ui "KPI Formula Registry" (app.js §KPI) to React.
// Backend: orchestrator /kpis/* (list/create/patch/delete/compile/versions/
// rollback/activate). Additive only — reuses existing theme.css classes.

const STATUSES = ['draft', 'active', 'deprecated'];
const STATUS_BADGE = { active: 'badge-green', draft: 'badge-amber', deprecated: 'badge-gray' };
const DIR_ARROW = { up: '↑', down: '↓' };

const EMPTY = {
  name: '', category: '', source_id: '', description: '',
  nl_formula: '', sql_expression: '', unit: '', direction: 'up', status: 'draft',
  change_note: '',
};

export default function Kpi() {
  const { sources, toast, refreshTick } = useAppState();
  const [kpis, setKpis] = useState(null); // null = loading
  const [search, setSearch] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');

  // modal
  const [open, setOpen] = useState(false);
  const [editId, setEditId] = useState(null);
  const [form, setForm] = useState(EMPTY);
  const [busy, setBusy] = useState(false);
  const [compiling, setCompiling] = useState(false);
  const [compileMsg, setCompileMsg] = useState('');
  const [guardrail, setGuardrail] = useState('');
  // version history
  const [showHistory, setShowHistory] = useState(false);
  const [versions, setVersions] = useState(null);

  const load = () => {
    setKpis(null);
    listKpis({ category: categoryFilter, status: statusFilter })
      .then((r) => setKpis(Array.isArray(r) ? r : []))
      .catch(() => setKpis([]));
  };

  // Reload when filters change (server-side category/status filter, like legacy).
  useEffect(load, [categoryFilter, statusFilter, refreshTick]);

  const categories = useMemo(() => {
    const set = new Set((kpis || []).map((k) => k.category).filter(Boolean));
    return Array.from(set).sort();
  }, [kpis]);

  const shown = useMemo(() => {
    const q = search.trim().toLowerCase();
    return (kpis || []).filter((k) => {
      if (!q) return true;
      return [k.name, k.description, k.category, k.nl_formula]
        .join(' ').toLowerCase().includes(q);
    });
  }, [kpis, search]);

  const setField = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  const reset = () => {
    setForm(EMPTY);
    setGuardrail('');
    setCompileMsg('');
    setShowHistory(false);
    setVersions(null);
  };

  const openCreate = () => { setEditId(null); reset(); setOpen(true); };

  const openEdit = async (id) => {
    setEditId(id); reset(); setOpen(true);
    try {
      const k = await getKpi(id);
      setForm({
        name: k.name || '', category: k.category || '', source_id: k.source_id || '',
        description: k.description || '', nl_formula: k.nl_formula || '',
        sql_expression: k.sql_expression || '', unit: k.unit || '',
        direction: k.direction || 'up', status: k.status || 'draft', change_note: '',
      });
    } catch (e) {
      toast(`Failed to load KPI: ${e.message}`, 'error');
    }
  };

  const close = () => { if (!busy && !compiling) setOpen(false); };

  // Build a column-context string from the selected source's schema (legacy parity).
  const buildColumnContext = async (sourceId) => {
    if (!sourceId) return '(No schema available — write formula referencing likely column names)';
    try {
      const entities = (await listEntities(sourceId)) || [];
      const lines = [];
      for (const ent of entities.slice(0, 15)) {
        const full = await getEntity(ent.metadata_id).catch(() => null);
        if (!full) continue;
        const tbl = full.table_name || ent.table_name || ent.entity_name || '';
        for (const a of (full.attributes || []).slice(0, 30)) {
          const col = a.column_name || a.attribute_name;
          lines.push(`${tbl}.${col} (${a.data_type || 'TEXT'})`);
        }
      }
      return lines.join('\n') || '(No schema available — write formula referencing likely column names)';
    } catch {
      return '(No schema available — write formula referencing likely column names)';
    }
  };

  const doCompile = async () => {
    const nl = form.nl_formula.trim();
    if (!nl) { toast('Enter a natural language formula first', 'warn'); return; }
    if (!form.name.trim()) { toast('KPI name is required before compiling', 'warn'); return; }
    setCompiling(true);
    setCompileMsg('Fetching schema…');
    try {
      const columnContext = await buildColumnContext(form.source_id);
      setCompileMsg('Calling LLM…');
      // Ensure the KPI exists (compile is keyed by id), creating/patching as needed.
      let id = editId;
      if (!id) {
        const res = await createKpi({
          name: form.name.trim(), nl_formula: nl, source_id: form.source_id || '',
          category: form.category || '', description: form.description.trim() || '',
          unit: form.unit.trim() || '', direction: form.direction || 'up', status: 'draft',
        });
        const k = res.kpi || res;
        id = k.kpi_id;
        setEditId(id);
      } else {
        await updateKpi(id, { nl_formula: nl, source_id: form.source_id || '' });
      }
      const result = await compileKpi(id, columnContext);
      const expr = result.sql_expression || '';
      setField('sql_expression', expr);
      setCompileMsg(expr ? '✓ Compiled successfully' : '⚠ LLM returned empty — check formula');
      if (expr) toast('Formula compiled', 'success');
    } catch (e) {
      setCompileMsg(`Compile failed: ${e.message}`);
      toast(`Compile failed: ${e.message}`, 'error');
    } finally {
      setCompiling(false);
    }
  };

  const save = async () => {
    const name = form.name.trim();
    if (!name) { toast('KPI name is required', 'warn'); return; }
    setBusy(true);
    setGuardrail('');
    const body = {
      name, description: form.description.trim(), category: form.category,
      source_id: form.source_id, nl_formula: form.nl_formula.trim(),
      sql_expression: form.sql_expression.trim(), unit: form.unit.trim(),
      direction: form.direction, status: form.status,
      change_note: form.change_note.trim(),
    };
    try {
      const res = editId ? await updateKpi(editId, body) : await createKpi(body);
      const warnings = res.warnings || [];
      toast(editId ? 'KPI updated' : 'KPI created', 'success');
      setOpen(false);
      if (warnings.length) setTimeout(() => warnings.forEach((w) => toast(`⚠ ${w.message}`, 'warn')), 200);
      load();
    } catch (e) {
      // Guardrail validation (422) comes back as the error message.
      setGuardrail(e.message);
    } finally {
      setBusy(false);
    }
  };

  const del = async () => {
    if (!editId) return;
    if (!window.confirm(`Delete KPI "${form.name}"? This cannot be undone.`)) return;
    setBusy(true);
    try {
      await deleteKpi(editId);
      toast('KPI deleted', 'info');
      setOpen(false);
      load();
    } catch (e) {
      toast(`Delete failed: ${e.message}`, 'error');
    } finally {
      setBusy(false);
    }
  };

  const activate = async () => {
    if (!editId) return;
    setGuardrail('');
    setBusy(true);
    try {
      await activateKpi(editId);
      toast('KPI activated ✓', 'success');
      setField('status', 'active');
      load();
    } catch (e) {
      setGuardrail(e.message);
    } finally {
      setBusy(false);
    }
  };

  const toggleHistory = async () => {
    const next = !showHistory;
    setShowHistory(next);
    if (next && editId) {
      setVersions(null);
      try { setVersions((await listKpiVersions(editId)) || []); }
      catch { setVersions([]); }
    }
  };

  const rollback = async (versionNum) => {
    if (!editId) return;
    if (!window.confirm(`Rollback to version ${versionNum}? The current state will be saved as a new version.`)) return;
    try {
      const r = await rollbackKpiVersion(editId, versionNum);
      setForm((f) => ({
        ...f,
        name: r.name || '', category: r.category || '', source_id: r.source_id || '',
        description: r.description || '', nl_formula: r.nl_formula || '',
        sql_expression: r.sql_expression || '', unit: r.unit || '',
        direction: r.direction || 'up', status: r.status || 'draft',
      }));
      toast(`Rolled back to v${versionNum}`, 'info');
      setVersions((await listKpiVersions(editId)) || []);
      load();
    } catch (e) {
      toast(`Rollback failed: ${e.message}`, 'error');
    }
  };

  const canActivate = editId && form.status !== 'active' && form.sql_expression.trim();

  return (
    <div id="view-kpi" className="view active">
      {/* Toolbar */}
      <div style={{ padding: '8px 14px', background: 'var(--bg-2)', borderBottom: '1px solid var(--border)', display: 'flex', gap: 8, alignItems: 'center', flexShrink: 0 }}>
        <span style={{ fontWeight: 600, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
          <IconKpi width="15" height="15" /> KPI Formula Registry
        </span>
        <span style={{ fontSize: 11, color: 'var(--text-2)' }}>Business metrics, NL formulas &amp; compiled SQL</span>
        <div className="search-wrap" style={{ marginLeft: 'auto' }}>
          <IconSearch />
          <input className="search-input" placeholder="Search KPIs…" value={search} onChange={(e) => setSearch(e.target.value)} />
        </div>
        <select className="search-input" style={{ width: 150 }} value={categoryFilter} onChange={(e) => setCategoryFilter(e.target.value)}>
          <option value="">All categories</option>
          {categories.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <select className="search-input" style={{ width: 130 }} value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
          <option value="">All statuses</option>
          {STATUSES.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <button className="btn btn-secondary" onClick={load} title="Refresh"><IconRefresh /></button>
        <button className="btn btn-primary" onClick={openCreate}><IconPlus /> Add KPI</button>
      </div>

      {/* Card grid */}
      <div style={{ flex: 1, overflowY: 'auto', padding: 14 }}>
        {kpis === null ? (
          <div className="empty-state"><span className="spinner" /></div>
        ) : shown.length === 0 ? (
          <div className="empty-state">
            <IconKpi strokeWidth="1.5" style={{ width: 36, height: 36, opacity: 0.3 }} />
            {(kpis || []).length === 0 ? 'No KPIs yet — add one to get started.' : 'No KPIs match your filters.'}
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 12 }}>
            {shown.map((k) => (
              <div
                key={k.kpi_id}
                onClick={() => openEdit(k.kpi_id)}
                style={{ cursor: 'pointer', background: 'var(--bg-2)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: 12 }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontWeight: 600, fontSize: 13 }}>
                      {k.name}
                      {!k.sql_expression && <span style={{ fontSize: 10, color: 'var(--accent)', marginLeft: 6 }}>⚠ no SQL</span>}
                    </div>
                    {k.category && <div style={{ fontSize: 10, color: 'var(--accent)', marginTop: 1 }}>{k.category}</div>}
                  </div>
                  <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexShrink: 0 }}>
                    <span className={`badge ${STATUS_BADGE[k.status] || 'badge-gray'}`}>{k.status}</span>
                    <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-2)' }}>{DIR_ARROW[k.direction] || '↑'} {k.unit || ''}</span>
                  </div>
                </div>
                {k.description && <div style={{ fontSize: 12, color: 'var(--text-2)', marginTop: 6, lineHeight: 1.4 }}>{k.description}</div>}
                {k.nl_formula && <div style={{ fontSize: 11, color: 'var(--text-2)', marginTop: 4, fontStyle: 'italic' }}>{k.nl_formula}</div>}
                {k.sql_expression && (
                  <div style={{ fontSize: 11, color: 'var(--accent)', marginTop: 4, fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={k.sql_expression}>
                    {k.sql_expression}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Add / Edit modal */}
      {open && (
        <div
          style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          onMouseDown={(e) => e.target === e.currentTarget && close()}
        >
          <div className="modal" style={{ width: 580, maxHeight: '92vh', overflowY: 'auto' }}>
            <div className="modal-title">
              <IconKpi />
              {editId ? 'Edit KPI' : 'Add KPI'}
            </div>

            {guardrail && (
              <div style={{ background: 'rgba(248,81,73,0.12)', border: '1px solid var(--red)', color: 'var(--red)', borderRadius: 'var(--radius)', padding: '8px 10px', fontSize: 12, marginBottom: 12, whiteSpace: 'pre-wrap' }}>
                ⛔ {guardrail}
              </div>
            )}

            <div className="form-row">
              <div className="form-grid">
                <div>
                  <label>KPI Name *</label>
                  <input value={form.name} onChange={(e) => setField('name', e.target.value)} placeholder="e.g. Net Revenue Growth" />
                </div>
                <div>
                  <label>Category</label>
                  <input list="kpi-categories" value={form.category} onChange={(e) => setField('category', e.target.value)} placeholder="Sales, Finance…" />
                  <datalist id="kpi-categories">{categories.map((c) => <option key={c} value={c} />)}</datalist>
                </div>
              </div>
            </div>

            <div className="form-row">
              <div className="form-grid">
                <div>
                  <label>Source</label>
                  <select value={form.source_id} onChange={(e) => setField('source_id', e.target.value)}>
                    <option value="">— any source —</option>
                    {sources.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
                  </select>
                </div>
                <div>
                  <label>Status</label>
                  <select value={form.status} onChange={(e) => setField('status', e.target.value)}>
                    {STATUSES.map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                </div>
              </div>
            </div>

            <div className="form-row">
              <label>Description</label>
              <textarea rows="2" value={form.description} onChange={(e) => setField('description', e.target.value)} placeholder="What this metric measures" />
            </div>

            <div className="form-row">
              <label>Natural-Language Formula</label>
              <textarea rows="2" value={form.nl_formula} onChange={(e) => setField('nl_formula', e.target.value)} placeholder="e.g. sum of revenue this year minus last year, divided by last year" />
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 6 }}>
                <button className="btn btn-secondary" type="button" onClick={doCompile} disabled={compiling}>
                  ▶ {compiling ? 'Compiling…' : 'Compile to SQL'}
                </button>
                {compileMsg && <span style={{ fontSize: 11, color: 'var(--text-2)' }}>{compileMsg}</span>}
              </div>
            </div>

            <div className="form-row">
              <label>SQL Expression</label>
              <textarea
                rows="2"
                value={form.sql_expression}
                onChange={(e) => setField('sql_expression', e.target.value)}
                placeholder="Compiled SQL (or write/edit manually)"
                style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}
              />
            </div>

            <div className="form-row">
              <div className="form-grid">
                <div>
                  <label>Unit</label>
                  <input value={form.unit} onChange={(e) => setField('unit', e.target.value)} placeholder="%, $, units…" />
                </div>
                <div>
                  <label>Direction</label>
                  <select value={form.direction} onChange={(e) => setField('direction', e.target.value)}>
                    <option value="up">↑ Higher is better</option>
                    <option value="down">↓ Lower is better</option>
                  </select>
                </div>
              </div>
            </div>

            {editId && (
              <div className="form-row">
                <label>Change Note (for version history)</label>
                <input value={form.change_note} onChange={(e) => setField('change_note', e.target.value)} placeholder="What changed and why" />
              </div>
            )}

            {/* Version history */}
            {editId && (
              <div className="form-row">
                <button className="btn btn-ghost" type="button" style={{ padding: '2px 6px', fontSize: 11 }} onClick={toggleHistory}>
                  {showHistory ? 'Hide History' : 'History'}
                </button>
                {showHistory && (
                  <div style={{ marginTop: 8, border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '4px 10px', maxHeight: 180, overflowY: 'auto' }}>
                    {versions === null ? (
                      <span style={{ color: 'var(--text-2)', fontSize: 12 }}>Loading…</span>
                    ) : versions.length === 0 ? (
                      <span style={{ color: 'var(--text-2)', fontSize: 12 }}>No previous versions.</span>
                    ) : (
                      versions.map((v) => (
                        <div key={v.version_num} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '5px 0', borderBottom: '1px solid var(--border)' }}>
                          <div style={{ minWidth: 0 }}>
                            <span style={{ color: 'var(--accent)', fontWeight: 600 }}>v{v.version_num}</span>
                            <span style={{ color: 'var(--text-2)', marginLeft: 6, fontSize: 11 }}>
                              {(v.created_at || '').substring(0, 16).replace('T', ' ')}
                              {v.changed_by ? ` by ${v.changed_by}` : ''}
                              {v.change_note ? ` — ${v.change_note}` : ''}
                            </span>
                            <div style={{ color: 'var(--text-2)', fontSize: 11, marginTop: 2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                              {v.status} · {(v.sql_expression || '').substring(0, 60)}{(v.sql_expression || '').length > 60 ? '…' : ''}
                            </div>
                          </div>
                          <button className="btn btn-secondary" style={{ fontSize: 11, padding: '2px 8px', flexShrink: 0, marginLeft: 8 }} onClick={() => rollback(v.version_num)}>
                            Rollback
                          </button>
                        </div>
                      ))
                    )}
                  </div>
                )}
              </div>
            )}

            <div className="modal-actions">
              {editId && (
                <button className="btn btn-ghost" style={{ marginRight: 'auto', color: 'var(--red)' }} onClick={del} disabled={busy}>
                  Delete
                </button>
              )}
              {canActivate && (
                <button className="btn btn-secondary" style={{ borderColor: 'var(--green)', color: 'var(--green)' }} onClick={activate} disabled={busy}>
                  Activate
                </button>
              )}
              <button className="btn btn-ghost" onClick={close} disabled={busy || compiling}>Cancel</button>
              <button className="btn btn-primary" onClick={save} disabled={busy || compiling}>
                {busy ? 'Saving…' : (editId ? 'Save Changes' : 'Create KPI')}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
