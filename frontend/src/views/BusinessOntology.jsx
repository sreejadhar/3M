import { useEffect, useMemo, useState } from 'react';
import { useAppState } from '../state.jsx';
import { useAuth } from '../auth.jsx';
import {
  listBizGlossaryTerms,
  approveBizGlossaryTerm,
  rejectBizGlossaryTerm,
  listBusinessOntologySources,
  generateBusinessOntology,
  getBusinessOntologyDraft,
  getBusinessOntologyTermKgLinks,
  saveBusinessOntologyDraftTtl,
  updateBusinessOntologyTerm,
  deleteBusinessOntologyTerm,
  addBusinessOntologyRelation,
  saveBusinessOntologyVersion,
  listBusinessOntologyVersions,
  getBusinessOntologyVersion,
  restoreBusinessOntologyVersion,
} from '../api/clients.js';
import { IconOntology, IconRefresh } from '../components/Icons.jsx';

// Business Ontology (SKOS+OWL) — generated per data source from that
// source's governed Business Glossary (draft/candidate/approved terms),
// distinct from the per-source structural ontology in OntologyViewer.jsx.
// Only sources with an already-generated glossary are offered (see
// listBusinessOntologySources -> GET /business-ontology/sources). Mirrors
// Glossary.jsx's card/edit-modal pattern for terms and OntologyViewer.jsx's
// raw-TTL textarea for power users.

const STATUS_ORDER = ['draft', 'candidate', 'approved'];
const STATUS_LABEL = { draft: 'Draft', candidate: 'Candidate', approved: 'Approved' };
const RELATIONSHIP_TYPES = ['broader', 'narrower', 'related', 'synonym'];

const EMPTY_FORM = { preferred_name: '', definition: '', domain: '', steward: '' };

export default function BusinessOntology() {
  const { toast, refreshTick } = useAppState();
  const { isAdmin } = useAuth();

  const [ontologySources, setOntologySources] = useState(null); // null = loading; [] = none generated yet
  const [sourceId, setSourceId] = useState('');
  const [terms, setTerms] = useState(null); // null = loading
  const [draft, setDraft] = useState(null);
  const [versions, setVersions] = useState([]);
  const [tab, setTab] = useState('terms');
  const [statusFilter, setStatusFilter] = useState('all');
  const [busy, setBusy] = useState(false);

  const [editingTerm, setEditingTerm] = useState(null); // term dict | null
  const [form, setForm] = useState(EMPTY_FORM);
  const [relTarget, setRelTarget] = useState('');
  const [relType, setRelType] = useState('related');
  const [kgLinks, setKgLinks] = useState(null); // null = loading, [] = none

  const [versionModalOpen, setVersionModalOpen] = useState(false);
  const [versionLabel, setVersionLabel] = useState('');
  const [previewVersion, setPreviewVersion] = useState(null);

  const [ttlText, setTtlText] = useState('');

  const loadSources = () => {
    setOntologySources(null);
    listBusinessOntologySources()
      .then((list) => {
        const arr = Array.isArray(list) ? list : [];
        setOntologySources(arr);
        setSourceId((cur) => (cur && arr.some((s) => s.source_id === cur)) ? cur : (arr[0]?.source_id || ''));
      })
      .catch((e) => {
        setOntologySources([]);
        toast(`Failed to load indexed sources: ${e.message}`, 'error');
      });
  };

  useEffect(loadSources, [refreshTick]);

  const load = () => {
    if (!sourceId) { setTerms([]); setDraft(null); setTtlText(''); setVersions([]); return; }
    setTerms(null);
    Promise.all([
      listBizGlossaryTerms({ sourceId }),
      getBusinessOntologyDraft(sourceId),
      listBusinessOntologyVersions(sourceId),
    ])
      .then(([t, d, v]) => {
        setTerms(Array.isArray(t) ? t : []);
        setDraft(d || null);
        setTtlText(d?.ttl_content || '');
        setVersions(Array.isArray(v) ? v : []);
      })
      .catch((e) => {
        setTerms([]);
        toast(`Failed to load Business Ontology: ${e.message}`, 'error');
      });
  };

  useEffect(load, [sourceId, refreshTick]);

  const grouped = useMemo(() => {
    const list = terms || [];
    const g = { draft: [], candidate: [], approved: [] };
    for (const t of list) {
      if (statusFilter !== 'all' && t.status !== statusFilter) continue;
      if (g[t.status]) g[t.status].push(t);
    }
    return g;
  }, [terms, statusFilter]);

  const refreshDraft = async () => {
    if (!sourceId) return;
    try {
      const d = await getBusinessOntologyDraft(sourceId);
      setDraft(d);
      setTtlText(d?.ttl_content || '');
    } catch (e) {
      toast(`Failed to refresh draft: ${e.message}`, 'error');
    }
  };

  const regenerate = async () => {
    if (!sourceId) return;
    setBusy(true);
    try {
      await generateBusinessOntology(sourceId);
      toast('Business ontology regenerated from the glossary', 'success');
      load();
    } catch (e) {
      toast(`Regenerate failed: ${e.message}`, 'error');
    } finally {
      setBusy(false);
    }
  };

  const openEdit = (term) => {
    setEditingTerm(term);
    setForm({
      preferred_name: term.preferred_name || '',
      definition: term.definition || '',
      domain: term.domain || '',
      steward: term.steward || '',
    });
    setRelTarget('');
    setRelType('related');
    setKgLinks(null);
    getBusinessOntologyTermKgLinks(sourceId, term.term_id)
      .then((links) => setKgLinks(Array.isArray(links) ? links : []))
      .catch(() => setKgLinks([]));
  };
  const closeEdit = () => { if (!busy) setEditingTerm(null); };
  const setField = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  const saveTerm = async () => {
    if (!editingTerm) return;
    setBusy(true);
    try {
      await updateBusinessOntologyTerm(sourceId, editingTerm.term_id, {
        preferred_name: form.preferred_name.trim(),
        definition: form.definition.trim(),
        domain: form.domain.trim(),
        steward: form.steward.trim(),
      });
      toast('Term updated — ontology regenerated', 'success');
      setEditingTerm(null);
      load();
    } catch (e) {
      toast(`Save failed: ${e.message}`, 'error');
    } finally {
      setBusy(false);
    }
  };

  const approveTerm = async (term) => {
    setBusy(true);
    try {
      await approveBizGlossaryTerm(term.term_id);
      toast('Term approved', 'success');
      load();
    } catch (e) {
      toast(`Approve failed: ${e.message}`, 'error');
    } finally {
      setBusy(false);
    }
  };

  const rejectTerm = async (term) => {
    setBusy(true);
    try {
      await rejectBizGlossaryTerm(term.term_id);
      toast('Term rejected', 'info');
      load();
    } catch (e) {
      toast(`Reject failed: ${e.message}`, 'error');
    } finally {
      setBusy(false);
    }
  };

  const deleteTerm = async (term) => {
    if (!window.confirm(`Delete business ontology term "${term.preferred_name}"? This deprecates it.`)) return;
    setBusy(true);
    try {
      await deleteBusinessOntologyTerm(sourceId, term.term_id);
      toast('Term deleted', 'info');
      setEditingTerm(null);
      load();
    } catch (e) {
      toast(`Delete failed: ${e.message}`, 'error');
    } finally {
      setBusy(false);
    }
  };

  const addRelation = async () => {
    if (!editingTerm || !relTarget) return;
    setBusy(true);
    try {
      await addBusinessOntologyRelation(sourceId, editingTerm.term_id, relTarget, relType);
      toast('Relation added — ontology regenerated', 'success');
      setRelTarget('');
      refreshDraft();
    } catch (e) {
      toast(`Add relation failed: ${e.message}`, 'error');
    } finally {
      setBusy(false);
    }
  };

  const saveDraftTtl = async () => {
    if (!sourceId) return;
    setBusy(true);
    try {
      const d = await saveBusinessOntologyDraftTtl(sourceId, ttlText);
      setDraft(d);
      toast('Draft saved', 'success');
    } catch (e) {
      toast(`Save failed: ${e.message}`, 'error');
    } finally {
      setBusy(false);
    }
  };

  const openSaveVersion = () => { setVersionLabel(''); setVersionModalOpen(true); };
  const saveVersion = async () => {
    if (!sourceId) return;
    setBusy(true);
    try {
      await saveBusinessOntologyVersion(sourceId, versionLabel.trim());
      toast('Version saved', 'success');
      setVersionModalOpen(false);
      const v = await listBusinessOntologyVersions(sourceId);
      setVersions(Array.isArray(v) ? v : []);
    } catch (e) {
      toast(`Save version failed: ${e.message}`, 'error');
    } finally {
      setBusy(false);
    }
  };

  const openVersionPreview = async (versionId) => {
    try {
      const v = await getBusinessOntologyVersion(sourceId, versionId);
      setPreviewVersion(v);
    } catch (e) {
      toast(`Failed to load version: ${e.message}`, 'error');
    }
  };

  const restoreVersion = async (versionId) => {
    if (!window.confirm('Restore this version as the current draft?')) return;
    setBusy(true);
    try {
      const d = await restoreBusinessOntologyVersion(sourceId, versionId);
      setDraft(d);
      setTtlText(d?.ttl_content || '');
      setPreviewVersion(null);
      toast('Version restored to draft', 'success');
    } catch (e) {
      toast(`Restore failed: ${e.message}`, 'error');
    } finally {
      setBusy(false);
    }
  };

  if (ontologySources === null) {
    return (
      <div id="view-business-ontology" className="view active">
        <div className="empty-state"><span className="spinner" /></div>
      </div>
    );
  }

  if (ontologySources.length === 0) {
    return (
      <div id="view-business-ontology" className="view active">
        <div className="empty-state">
          <IconOntology strokeWidth="1.5" style={{ width: 36, height: 36, opacity: 0.3 }} />
          No indexed data source has a generated Business Glossary yet.
          <div style={{ fontSize: 11, color: 'var(--text-2)', marginTop: 6 }}>
            Generate a glossary for a source from "Business Glossary (Discovery)" first — a Business
            Ontology can then be built from its terms.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div id="view-business-ontology" className="view active">
      <div style={{ padding: '8px 14px', background: 'var(--bg-2)', borderBottom: '1px solid var(--border)', display: 'flex', gap: 8, alignItems: 'center', flexShrink: 0 }}>
        <span style={{ fontWeight: 600, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
          <IconOntology width="15" height="15" /> Business Ontology
        </span>
        <select
          className="search-input"
          style={{ width: 220 }}
          value={sourceId}
          onChange={(e) => setSourceId(e.target.value)}
        >
          {ontologySources.map((s) => (
            <option key={s.source_id} value={s.source_id}>{s.source_name} ({s.term_count} terms)</option>
          ))}
        </select>
        {draft && (
          <span style={{ fontSize: 11, color: 'var(--text-2)', marginLeft: 4 }}>
            {draft.triple_count} triples · updated {draft.updated_at ? new Date(draft.updated_at).toLocaleString() : '—'}
          </span>
        )}
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <button className="btn btn-secondary" onClick={loadSources} title="Refresh"><IconRefresh /></button>
          <button className="btn btn-secondary" onClick={regenerate} disabled={busy}>Regenerate</button>
          {isAdmin && (
            <button className="btn btn-primary" onClick={openSaveVersion} disabled={busy}>Save Version</button>
          )}
        </div>
      </div>

      <div style={{ display: 'flex', gap: 4, padding: '8px 14px 0', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
        {[['terms', 'Terms'], ['ttl', 'Raw TTL'], ['versions', 'Versions']].map(([id, label]) => (
          <button
            key={id}
            className={`btn ${tab === id ? 'btn-primary' : 'btn-ghost'}`}
            style={{ borderRadius: '6px 6px 0 0' }}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === 'terms' && (
        <div style={{ flex: 1, overflowY: 'auto', padding: 14 }}>
          <div style={{ marginBottom: 12 }}>
            <select className="search-input" style={{ width: 200 }} value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
              <option value="all">All statuses</option>
              {STATUS_ORDER.map((s) => <option key={s} value={s}>{STATUS_LABEL[s]}</option>)}
            </select>
          </div>
          {terms === null ? (
            <div className="empty-state"><span className="spinner" /></div>
          ) : (
            STATUS_ORDER.filter((s) => statusFilter === 'all' || statusFilter === s).map((status) => (
              <div key={status} style={{ marginBottom: 20 }}>
                <div style={{ fontWeight: 600, fontSize: 12, marginBottom: 8, color: 'var(--text-2)' }}>
                  {STATUS_LABEL[status]} ({grouped[status].length})
                </div>
                {grouped[status].length === 0 ? (
                  <div style={{ fontSize: 11, color: 'var(--text-2)' }}>No terms.</div>
                ) : (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
                    {grouped[status].map((t) => (
                      <div key={t.term_id} style={{ background: 'var(--bg-2)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: 12 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
                          <div style={{ minWidth: 0, fontWeight: 600, fontSize: 13, cursor: isAdmin ? 'pointer' : 'default' }}
                               onClick={() => isAdmin && openEdit(t)}>
                            {t.preferred_name}
                          </div>
                          <span className={`badge ${status === 'approved' ? 'badge-green' : status === 'candidate' ? 'badge-amber' : ''}`}>{STATUS_LABEL[status]}</span>
                        </div>
                        {t.domain && <div style={{ fontSize: 10, color: 'var(--accent)', marginTop: 2 }}>{t.domain}</div>}
                        {t.definition && <div style={{ fontSize: 12, color: 'var(--text-2)', marginTop: 6, lineHeight: 1.4 }}>{t.definition}</div>}
                        {isAdmin && (
                          <div style={{ display: 'flex', gap: 6, marginTop: 10 }}>
                            <button className="btn btn-ghost" style={{ fontSize: 11, padding: '2px 8px' }} onClick={() => openEdit(t)}>Edit</button>
                            {status !== 'approved' && (
                              <button className="btn btn-ghost" style={{ fontSize: 11, padding: '2px 8px' }} onClick={() => approveTerm(t)}>Approve</button>
                            )}
                            <button className="btn btn-ghost" style={{ fontSize: 11, padding: '2px 8px' }} onClick={() => rejectTerm(t)}>Reject</button>
                            <button className="btn btn-ghost" style={{ fontSize: 11, padding: '2px 8px', color: 'var(--red)', marginLeft: 'auto' }} onClick={() => deleteTerm(t)}>Delete</button>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      )}

      {tab === 'ttl' && (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 14, minHeight: 0 }}>
          <textarea
            spellCheck={false}
            value={ttlText}
            onChange={(e) => setTtlText(e.target.value)}
            readOnly={!isAdmin}
            style={{ flex: 1, width: '100%', resize: 'none', fontFamily: 'monospace', fontSize: 12, background: 'var(--bg-2)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', color: 'var(--text-0)', padding: 10 }}
            placeholder="Business ontology TTL will appear here after generation…"
          />
          {isAdmin && (
            <div style={{ marginTop: 8, display: 'flex', justifyContent: 'flex-end' }}>
              <button className="btn btn-primary" onClick={saveDraftTtl} disabled={busy}>Save Draft</button>
            </div>
          )}
        </div>
      )}

      {tab === 'versions' && (
        <div style={{ flex: 1, overflowY: 'auto', padding: 14 }}>
          {versions.length === 0 ? (
            <div className="empty-state">No versions saved yet.</div>
          ) : (
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ textAlign: 'left', color: 'var(--text-2)' }}>
                  <th style={{ padding: '4px 8px' }}>#</th>
                  <th style={{ padding: '4px 8px' }}>Label</th>
                  <th style={{ padding: '4px 8px' }}>Triples</th>
                  <th style={{ padding: '4px 8px' }}>Created By</th>
                  <th style={{ padding: '4px 8px' }}>Created At</th>
                  <th style={{ padding: '4px 8px' }} />
                </tr>
              </thead>
              <tbody>
                {versions.map((v) => (
                  <tr key={v.version_id} style={{ borderTop: '1px solid var(--border)', cursor: 'pointer' }} onClick={() => openVersionPreview(v.version_id)}>
                    <td style={{ padding: '6px 8px' }}>v{v.version_number} {v.is_current ? <span className="badge badge-green" style={{ marginLeft: 4 }}>current</span> : null}</td>
                    <td style={{ padding: '6px 8px' }}>{v.label || <span style={{ color: 'var(--text-2)' }}>—</span>}</td>
                    <td style={{ padding: '6px 8px' }}>{v.triple_count}</td>
                    <td style={{ padding: '6px 8px' }}>{v.created_by || '—'}</td>
                    <td style={{ padding: '6px 8px' }}>{new Date(v.created_at).toLocaleString()}</td>
                    <td style={{ padding: '6px 8px' }}>
                      {isAdmin && (
                        <button className="btn btn-ghost" style={{ fontSize: 11 }} onClick={(e) => { e.stopPropagation(); restoreVersion(v.version_id); }}>
                          Restore
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {previewVersion && (
            <div
              style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
              onMouseDown={(e) => e.target === e.currentTarget && setPreviewVersion(null)}
            >
              <div className="modal" style={{ width: 700, maxHeight: '85vh', overflowY: 'auto' }}>
                <div className="modal-title">
                  v{previewVersion.version_number}{previewVersion.label ? ` — ${previewVersion.label}` : ''}
                </div>
                <textarea
                  readOnly
                  spellCheck={false}
                  value={previewVersion.ttl_content}
                  style={{ width: '100%', height: 400, fontFamily: 'monospace', fontSize: 11, background: 'var(--bg-1)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', color: 'var(--text-0)', padding: 10 }}
                />
                <div className="modal-actions">
                  {isAdmin && (
                    <button className="btn btn-primary" style={{ marginRight: 'auto' }} onClick={() => restoreVersion(previewVersion.version_id)}>
                      Restore this version
                    </button>
                  )}
                  <button className="btn btn-ghost" onClick={() => setPreviewVersion(null)}>Close</button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Term edit modal (admin-only, opened from Terms tab) */}
      {editingTerm && (
        <div
          style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          onMouseDown={(e) => e.target === e.currentTarget && closeEdit()}
        >
          <div className="modal" style={{ width: 540, maxHeight: '90vh', overflowY: 'auto' }}>
            <div className="modal-title">
              <IconOntology />
              Edit Business Concept
            </div>

            <div className="form-row">
              <label>Preferred Label</label>
              <input value={form.preferred_name} onChange={(e) => setField('preferred_name', e.target.value)} />
            </div>
            <div className="form-row">
              <label>Definition</label>
              <textarea rows="3" value={form.definition} onChange={(e) => setField('definition', e.target.value)} />
            </div>
            <div className="form-row">
              <div className="form-grid">
                <div>
                  <label>Domain</label>
                  <input value={form.domain} onChange={(e) => setField('domain', e.target.value)} />
                </div>
                <div>
                  <label>Steward</label>
                  <input value={form.steward} onChange={(e) => setField('steward', e.target.value)} />
                </div>
              </div>
            </div>

            <div className="form-row">
              <label>Enriches KG</label>
              {kgLinks === null ? (
                <div style={{ fontSize: 11, color: 'var(--text-2)' }}>Loading…</div>
              ) : kgLinks.length === 0 ? (
                <div style={{ fontSize: 11, color: 'var(--text-2)' }}>
                  Not linked to any graph nodes in this source's KG yet.
                </div>
              ) : (
                <div style={{ fontSize: 11, color: 'var(--text-2)' }}>
                  Linked to {kgLinks.length} graph node{kgLinks.length === 1 ? '' : 's'} in this source's KG
                  {' '}({kgLinks.map((l) => l.target_node_id.split('/').pop()).join(', ')})
                </div>
              )}
            </div>

            <div className="form-row">
              <label>Relations (broader / narrower / related / synonym)</label>
              <div style={{ display: 'flex', gap: 8 }}>
                <select
                  className="search-input"
                  style={{ flex: 1 }}
                  value={relTarget}
                  onChange={(e) => setRelTarget(e.target.value)}
                >
                  <option value="">— select term —</option>
                  {(terms || [])
                    .filter((t) => t.term_id !== editingTerm.term_id)
                    .map((t) => <option key={t.term_id} value={t.term_id}>{t.preferred_name}</option>)}
                </select>
                <select className="search-input" style={{ width: 120 }} value={relType} onChange={(e) => setRelType(e.target.value)}>
                  {RELATIONSHIP_TYPES.map((r) => <option key={r} value={r}>{r}</option>)}
                </select>
                <button className="btn btn-secondary" onClick={addRelation} disabled={!relTarget || busy}>Add</button>
              </div>
            </div>

            <div className="modal-actions">
              <button className="btn btn-ghost" style={{ marginRight: 'auto', color: 'var(--red)' }} onClick={() => deleteTerm(editingTerm)} disabled={busy}>
                Delete
              </button>
              <button className="btn btn-ghost" onClick={closeEdit} disabled={busy}>Cancel</button>
              <button className="btn btn-primary" onClick={saveTerm} disabled={busy}>
                {busy ? 'Saving…' : 'Save Changes'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Save Version modal */}
      {versionModalOpen && (
        <div
          style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          onMouseDown={(e) => e.target === e.currentTarget && setVersionModalOpen(false)}
        >
          <div className="modal" style={{ width: 400 }}>
            <div className="modal-title">Save Version</div>
            <div className="form-row">
              <label>Label</label>
              <input value={versionLabel} onChange={(e) => setVersionLabel(e.target.value)} placeholder="e.g. Q3 governance review" />
            </div>
            <div className="modal-actions">
              <button className="btn btn-ghost" onClick={() => setVersionModalOpen(false)} disabled={busy}>Cancel</button>
              <button className="btn btn-primary" onClick={saveVersion} disabled={busy}>
                {busy ? 'Saving…' : 'Save Version'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
