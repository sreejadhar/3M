import { useEffect, useRef, useState, useCallback } from 'react';
import { IconDocuments, IconSearch, IconPlus, IconRefresh } from '../components/Icons.jsx';
import {
  docListSources, docCreateSource, docStartIndex, docGetJob,
  docListAssets, docGetAsset, docSearch, docQuery,
} from '../api/clients.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

const DOC_TYPE_BADGE = {
  report: 'badge-blue', policy: 'badge-purple', contract: 'badge-cyan',
  presentation: 'badge-amber', research: 'badge-blue', manual: 'badge-gray',
  correspondence: 'badge-green', other: 'badge-gray',
};
const SENS_BADGE = {
  public: 'badge-green', internal: 'badge-cyan',
  confidential: 'badge-amber', restricted: 'badge-red',
};
const SRC_TYPE_BADGE = {
  local: 'badge-gray', s3: 'badge-blue', gcs: 'badge-cyan',
  azure: 'badge-blue', gdrive: 'badge-green', sharepoint: 'badge-purple',
};

function fmtDate(iso) {
  if (!iso) return '—';
  try { return new Date(iso).toLocaleString(); } catch { return iso; }
}
function relTime(iso) {
  if (!iso) return null;
  const s = Math.floor((Date.now() - new Date(iso)) / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

const EMPTY_SOURCE = { name: '', source_type: 'local', path: '', bucket: '',
  prefix: '', folder_id: '', domain: '' };

// ── Icons used only here ──────────────────────────────────────────────────────
const S = (p) => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" {...p} />;
const IconShield = (p) => (
  <S {...p}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></S>
);
const IconFile = (p) => (
  <S {...p}>
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
  </S>
);
const IconClose = (p) => (
  <S {...p}><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></S>
);
const IconQuery = (p) => (
  <S {...p}><circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" /></S>
);
const IconLink = (p) => (
  <S {...p}><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" /><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" /></S>
);

// ── Add Source modal ──────────────────────────────────────────────────────────
function AddSourceModal({ onClose, onCreated }) {
  const [form, setForm] = useState(EMPTY_SOURCE);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));

  const submit = async () => {
    if (!form.name.trim()) { setErr('Name is required'); return; }
    setBusy(true); setErr('');
    try {
      const conn = {};
      if (form.source_type === 'local') conn.path = form.path;
      else if (form.source_type === 's3') { conn.bucket = form.bucket; if (form.prefix) conn.prefix = form.prefix; }
      else if (form.source_type === 'gdrive') conn.folder_id = form.folder_id;
      const src = await docCreateSource({ name: form.name.trim(), source_type: form.source_type, connection: conn, domain: form.domain.trim() });
      onCreated(src);
    } catch (e) { setErr(e.message); }
    finally { setBusy(false); }
  };

  return (
    <div className="modal-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal-box" style={{ width: 460 }}>
        <div className="modal-header">
          <span>Add Document Source</span>
          <button className="btn btn-ghost" style={{ padding: '2px 6px' }} onClick={onClose}><IconClose /></button>
        </div>
        <div className="modal-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div className="form-row">
            <label>Source Name</label>
            <input className="form-input" value={form.name} onChange={e => set('name', e.target.value)} placeholder="e.g. PNC Policy Docs" />
          </div>
          <div className="form-row">
            <label>Type</label>
            <select className="form-input" value={form.source_type} onChange={e => set('source_type', e.target.value)}>
              <option value="local">Local Folder</option>
              <option value="s3">Amazon S3</option>
              <option value="gdrive">Google Drive</option>
            </select>
          </div>
          {form.source_type === 'local' && (
            <div className="form-row">
              <label>Folder Path</label>
              <input className="form-input" value={form.path} onChange={e => set('path', e.target.value)} placeholder="C:/Documents/PNC or /data/docs" />
            </div>
          )}
          {form.source_type === 's3' && (
            <>
              <div className="form-row">
                <label>Bucket</label>
                <input className="form-input" value={form.bucket} onChange={e => set('bucket', e.target.value)} placeholder="my-docs-bucket" />
              </div>
              <div className="form-row">
                <label>Prefix (optional)</label>
                <input className="form-input" value={form.prefix} onChange={e => set('prefix', e.target.value)} placeholder="pnc/policies/" />
              </div>
            </>
          )}
          {form.source_type === 'gdrive' && (
            <div className="form-row">
              <label>Folder ID</label>
              <input className="form-input" value={form.folder_id} onChange={e => set('folder_id', e.target.value)} placeholder="Google Drive folder ID" />
            </div>
          )}
          <div className="form-row">
            <label>Domain Hint (optional)</label>
            <input className="form-input" value={form.domain} onChange={e => set('domain', e.target.value)} placeholder="e.g. Banking, Insurance, CPG" />
          </div>
          {err && <div className="err-msg">{err}</div>}
        </div>
        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" onClick={submit} disabled={busy}>
            {busy ? 'Creating…' : 'Create Source'}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Asset detail drawer ───────────────────────────────────────────────────────
function AssetDrawer({ assetId, onClose }) {
  const [asset, setAsset] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!assetId) return;
    setLoading(true);
    docGetAsset(assetId)
      .then(setAsset)
      .catch(() => setAsset(null))
      .finally(() => setLoading(false));
  }, [assetId]);

  if (!assetId) return null;

  const piiEntities = asset?.pii_entities || [];
  const topics = asset?.topics || [];
  const entities = asset?.named_entities || {};
  const timeRefs = asset?.time_references || [];

  return (
    <div className="di-drawer">
      <div className="di-drawer-header">
        <IconFile style={{ width: 14, height: 14, flexShrink: 0 }} />
        <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {loading ? 'Loading…' : (asset?.title || asset?.file_name || assetId)}
        </span>
        <button className="btn btn-ghost" style={{ padding: '2px 4px' }} onClick={onClose}><IconClose /></button>
      </div>
      {loading ? (
        <div style={{ padding: 20, color: 'var(--text-2)', textAlign: 'center' }}>Loading…</div>
      ) : !asset ? (
        <div style={{ padding: 20, color: 'var(--red)' }}>Failed to load asset.</div>
      ) : (
        <div className="di-drawer-body">
          {/* Badges row */}
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 12 }}>
            <span className={`badge ${DOC_TYPE_BADGE[asset.doc_type] || 'badge-gray'}`}>{asset.doc_type || 'other'}</span>
            <span className={`badge ${SENS_BADGE[asset.sensitivity] || 'badge-gray'}`}>{asset.sensitivity || 'internal'}</span>
            {asset.language && <span className="badge badge-gray">{asset.language?.toUpperCase()}</span>}
            {asset.pii_risk && (
              <span className="badge badge-red" style={{ gap: 4 }}>
                <IconShield style={{ width: 10, height: 10 }} /> PII
              </span>
            )}
          </div>

          {/* Summary */}
          {asset.summary && (
            <div className="di-section">
              <div className="di-section-title">Summary</div>
              <p style={{ color: 'var(--text-0)', lineHeight: 1.6, fontSize: 12 }}>{asset.summary}</p>
            </div>
          )}

          {/* Topics */}
          {topics.length > 0 && (
            <div className="di-section">
              <div className="di-section-title">Topics</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                {topics.map((t, i) => <span key={i} className="badge badge-blue">{t}</span>)}
              </div>
            </div>
          )}

          {/* Named entities */}
          {Object.entries(entities).some(([, v]) => v?.length > 0) && (
            <div className="di-section">
              <div className="di-section-title">Named Entities</div>
              {Object.entries(entities).map(([kind, vals]) =>
                vals?.length > 0 ? (
                  <div key={kind} style={{ marginBottom: 6 }}>
                    <span style={{ fontSize: 10, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{kind}</span>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 3 }}>
                      {vals.map((v, i) => <span key={i} className="badge badge-gray">{v}</span>)}
                    </div>
                  </div>
                ) : null
              )}
            </div>
          )}

          {/* Time references */}
          {timeRefs.length > 0 && (
            <div className="di-section">
              <div className="di-section-title">Time References</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                {timeRefs.map((t, i) => <span key={i} className="badge badge-cyan">{t}</span>)}
              </div>
            </div>
          )}

          {/* PII entities */}
          <div className="di-section">
            <div className="di-section-title" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <IconShield style={{ width: 12, height: 12, color: piiEntities.length > 0 ? 'var(--red)' : 'var(--green)' }} />
              PII Scan
              <span className={`badge ${piiEntities.length > 0 ? 'badge-red' : 'badge-green'}`} style={{ marginLeft: 2 }}>
                {piiEntities.length > 0 ? `${piiEntities.length} entity${piiEntities.length > 1 ? 'ies' : ''}` : 'Clean'}
              </span>
            </div>
            {piiEntities.length > 0 ? (
              <table className="data-table" style={{ marginTop: 6 }}>
                <thead>
                  <tr>
                    <th>Type</th>
                    <th>Masked Value</th>
                    <th>Confidence</th>
                    <th>Source</th>
                  </tr>
                </thead>
                <tbody>
                  {piiEntities.map((e, i) => (
                    <tr key={i}>
                      <td><span className="badge badge-red">{e.type}</span></td>
                      <td className="mono">{e.masked_value}</td>
                      <td>{e.confidence != null ? `${Math.round(e.confidence * 100)}%` : '—'}</td>
                      <td><span className={`badge ${e.source === 'regex' ? 'badge-amber' : 'badge-purple'}`}>{e.source || '—'}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p style={{ color: 'var(--text-2)', fontSize: 12, marginTop: 4 }}>No PII detected in this document.</p>
            )}
          </div>

          {/* File meta */}
          <div className="di-section">
            <div className="di-section-title">File Info</div>
            <div className="di-meta-grid">
              <span className="di-meta-key">File</span><span className="di-meta-val mono">{asset.file_name}</span>
              <span className="di-meta-key">Indexed</span><span className="di-meta-val">{fmtDate(asset.updated_at)}</span>
              {asset.page_count != null && <><span className="di-meta-key">Pages</span><span className="di-meta-val">{asset.page_count}</span></>}
              {asset.domain && <><span className="di-meta-key">Domain</span><span className="di-meta-val">{asset.domain}</span></>}
              {asset.ocr_used && <><span className="di-meta-key">OCR</span><span className="di-meta-val">Yes {asset.ocr_confidence != null ? `(${Math.round(asset.ocr_confidence * 100)}% confidence)` : ''}</span></>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main view ─────────────────────────────────────────────────────────────────
export default function DocumentIntelligence() {
  const [sources, setSources] = useState([]);
  const [selectedSrc, setSelectedSrc] = useState(null);
  const [showAdd, setShowAdd] = useState(false);

  // assets tab
  const [assets, setAssets] = useState(null);
  const [selectedAsset, setSelectedAsset] = useState(null);
  const [loadingAssets, setLoadingAssets] = useState(false);

  // search tab
  const [searchQ, setSearchQ] = useState('');
  const [searchResults, setSearchResults] = useState(null);
  const [searching, setSearching] = useState(false);

  // query tab
  const [queryQ, setQueryQ] = useState('');
  const [queryResult, setQueryResult] = useState(null);
  const [querying, setQuerying] = useState(false);

  // tabs
  const [tab, setTab] = useState('assets');

  // running jobs polling
  const [runningJobs, setRunningJobs] = useState({}); // sourceId → jobId
  const pollRef = useRef(null);

  const loadSources = useCallback(() => {
    docListSources()
      .then(r => setSources(Array.isArray(r) ? r : []))
      .catch(() => setSources([]));
  }, []);

  useEffect(() => { loadSources(); }, [loadSources]);

  // auto-select first source
  useEffect(() => {
    if (!selectedSrc && sources.length > 0) setSelectedSrc(sources[0].id);
  }, [sources, selectedSrc]);

  // load assets when source changes
  useEffect(() => {
    if (!selectedSrc) { setAssets([]); return; }
    setLoadingAssets(true);
    setSelectedAsset(null);
    docListAssets(selectedSrc)
      .then(r => setAssets(Array.isArray(r) ? r : []))
      .catch(() => setAssets([]))
      .finally(() => setLoadingAssets(false));
  }, [selectedSrc]);

  // poll running jobs
  useEffect(() => {
    if (Object.keys(runningJobs).length === 0) return;
    pollRef.current = setInterval(async () => {
      const updates = {};
      let changed = false;
      for (const [srcId, jobId] of Object.entries(runningJobs)) {
        try {
          const job = await docGetJob(jobId);
          if (job.status !== 'running') {
            changed = true;
            loadSources();
            if (srcId === selectedSrc) {
              docListAssets(srcId).then(r => setAssets(Array.isArray(r) ? r : [])).catch(() => {});
            }
          } else {
            updates[srcId] = jobId;
          }
        } catch { /* ignore */ }
      }
      if (changed) setRunningJobs(updates);
    }, 3000);
    return () => clearInterval(pollRef.current);
  }, [runningJobs, selectedSrc, loadSources]);

  const handleIndex = async (srcId) => {
    try {
      const job = await docStartIndex(srcId);
      setRunningJobs(j => ({ ...j, [srcId]: job.job_id }));
    } catch (e) { alert(`Index failed: ${e.message}`); }
  };

  const handleSearch = async () => {
    if (!searchQ.trim()) return;
    setSearching(true); setSearchResults(null);
    try {
      const r = await docSearch(searchQ.trim());
      setSearchResults(r.results || []);
    } catch { setSearchResults([]); }
    finally { setSearching(false); }
  };

  const handleQuery = async () => {
    if (!queryQ.trim()) return;
    setQuerying(true); setQueryResult(null);
    try {
      const r = await docQuery(queryQ.trim());
      setQueryResult(r);
    } catch (e) { setQueryResult({ error: e.message }); }
    finally { setQuerying(false); }
  };

  const activeSrc = sources.find(s => s.id === selectedSrc);
  const totalDocs = assets?.length ?? 0;
  const piiDocs = assets?.filter(a => a.pii_risk).length ?? 0;

  return (
    <div className="di-root">
      {/* ── Left: sources panel ─────────────────────────────────────── */}
      <div className="di-left">
        <div className="panel-header" style={{ justifyContent: 'space-between' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <IconDocuments /> Document Sources
          </span>
          <button className="btn btn-primary" style={{ padding: '3px 8px', fontSize: 11 }} onClick={() => setShowAdd(true)}>
            <IconPlus /> Add
          </button>
        </div>
        <div className="di-sources-list">
          {sources.length === 0 ? (
            <div style={{ padding: '20px 14px', color: 'var(--text-2)', fontSize: 12, textAlign: 'center' }}>
              No document sources yet.<br />Click <strong>Add</strong> to register one.
            </div>
          ) : sources.map(s => {
            const isRunning = !!runningJobs[s.id];
            return (
              <div
                key={s.id}
                className={`source-card ${selectedSrc === s.id ? 'selected' : ''}`}
                onClick={() => setSelectedSrc(s.id)}
              >
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="src-name">{s.name}</div>
                  <div className="src-meta" style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 3 }}>
                    <span className={`badge ${SRC_TYPE_BADGE[s.source_type] || 'badge-gray'}`}>{s.source_type}</span>
                    {s.domain && <span className="badge badge-purple">{s.domain}</span>}
                  </div>
                  <div className="src-meta" style={{ marginTop: 4 }}>
                    {s.last_indexed_at ? `Indexed ${relTime(s.last_indexed_at)}` : 'Never indexed'}
                  </div>
                </div>
                <button
                  className={`btn ${isRunning ? 'btn-secondary' : 'btn-ghost'}`}
                  style={{ fontSize: 11, padding: '3px 8px', flexShrink: 0 }}
                  disabled={isRunning}
                  onClick={e => { e.stopPropagation(); handleIndex(s.id); }}
                  title="Trigger indexing"
                >
                  {isRunning ? (
                    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <span className="status-dot indexing" /> Running
                    </span>
                  ) : 'Index'}
                </button>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Right: content ──────────────────────────────────────────── */}
      <div className="di-right">
        {/* Header */}
        <div className="di-right-header">
          <div>
            <span className="page-title">{activeSrc ? activeSrc.name : 'Document Intelligence'}</span>
            {activeSrc && (
              <span className="page-sub" style={{ marginLeft: 10 }}>
                {totalDocs} documents{piiDocs > 0 ? ` · ${piiDocs} with PII` : ''}
              </span>
            )}
          </div>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
            <button className="btn btn-ghost" onClick={loadSources} title="Refresh">
              <IconRefresh />
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="di-tabs">
          {[['assets', IconFile, 'Assets'], ['search', IconSearch, 'Search'], ['query', IconQuery, 'Cross-Modal Query']].map(([id, Icon, label]) => (
            <button key={id} className={`di-tab ${tab === id ? 'active' : ''}`} onClick={() => setTab(id)}>
              <Icon style={{ width: 13, height: 13 }} /> {label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="di-tab-content">

          {/* ASSETS TAB */}
          {tab === 'assets' && (
            <div className="di-assets-pane">
              <div className="di-assets-table-wrap">
                {!selectedSrc ? (
                  <div className="di-empty">Select a document source from the left panel.</div>
                ) : loadingAssets ? (
                  <div className="di-empty">Loading documents…</div>
                ) : assets?.length === 0 ? (
                  <div className="di-empty">No documents indexed yet. Click <strong>Index</strong> on the source to start.</div>
                ) : (
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Title</th>
                        <th>Type</th>
                        <th>Sensitivity</th>
                        <th>PII</th>
                        <th>Domain</th>
                        <th>Indexed</th>
                      </tr>
                    </thead>
                    <tbody>
                      {assets.map(a => (
                        <tr
                          key={a.id}
                          style={{ cursor: 'pointer' }}
                          onClick={() => setSelectedAsset(a.id === selectedAsset ? null : a.id)}
                        >
                          <td>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                              <IconFile style={{ width: 13, height: 13, flexShrink: 0, color: 'var(--text-2)' }} />
                              <div>
                                <div style={{ fontWeight: 600 }}>{a.title || a.file_name}</div>
                                {a.summary && <div className="dim" style={{ fontSize: 11, maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{a.summary}</div>}
                              </div>
                            </div>
                          </td>
                          <td><span className={`badge ${DOC_TYPE_BADGE[a.doc_type] || 'badge-gray'}`}>{a.doc_type || '—'}</span></td>
                          <td><span className={`badge ${SENS_BADGE[a.sensitivity] || 'badge-gray'}`}>{a.sensitivity || '—'}</span></td>
                          <td>
                            {a.pii_risk ? (
                              <span className="badge badge-red" style={{ gap: 4 }}>
                                <IconShield style={{ width: 10, height: 10 }} /> Yes
                              </span>
                            ) : (
                              <span className="badge badge-green">Clean</span>
                            )}
                          </td>
                          <td className="muted">{a.domain || '—'}</td>
                          <td className="dim">{relTime(a.updated_at) || '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
              <AssetDrawer assetId={selectedAsset} onClose={() => setSelectedAsset(null)} />
            </div>
          )}

          {/* SEARCH TAB */}
          {tab === 'search' && (
            <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 12, height: '100%', overflow: 'hidden' }}>
              <div style={{ display: 'flex', gap: 8 }}>
                <input
                  className="form-input"
                  style={{ flex: 1 }}
                  placeholder="Search title, summary, topics…"
                  value={searchQ}
                  onChange={e => setSearchQ(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleSearch()}
                />
                <button className="btn btn-primary" onClick={handleSearch} disabled={searching}>
                  <IconSearch style={{ width: 13, height: 13 }} />
                  {searching ? 'Searching…' : 'Search'}
                </button>
              </div>
              <div style={{ flex: 1, overflow: 'auto' }}>
                {searchResults === null && !searching && (
                  <div className="di-empty">Enter a query and press Search.</div>
                )}
                {searchResults?.length === 0 && (
                  <div className="di-empty">No results found.</div>
                )}
                {searchResults?.length > 0 && (
                  <table className="data-table">
                    <thead>
                      <tr><th>Title</th><th>Type</th><th>Sensitivity</th><th>PII</th><th>Domain</th></tr>
                    </thead>
                    <tbody>
                      {searchResults.map(a => (
                        <tr key={a.id} style={{ cursor: 'pointer' }} onClick={() => { setTab('assets'); setSelectedSrc(a.source_id); setSelectedAsset(a.id); }}>
                          <td>
                            <div style={{ fontWeight: 600 }}>{a.title || a.file_name}</div>
                            {a.summary && <div className="dim" style={{ fontSize: 11 }}>{a.summary.slice(0, 120)}…</div>}
                          </td>
                          <td><span className={`badge ${DOC_TYPE_BADGE[a.doc_type] || 'badge-gray'}`}>{a.doc_type || '—'}</span></td>
                          <td><span className={`badge ${SENS_BADGE[a.sensitivity] || 'badge-gray'}`}>{a.sensitivity || '—'}</span></td>
                          <td>{a.pii_risk ? <span className="badge badge-red">PII</span> : <span className="badge badge-green">Clean</span>}</td>
                          <td className="muted">{a.domain || '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          )}

          {/* CROSS-MODAL QUERY TAB */}
          {tab === 'query' && (
            <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 12, height: '100%', overflow: 'hidden' }}>
              <div style={{ background: 'var(--bg-3)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '10px 14px', fontSize: 12, color: 'var(--text-1)', display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                <IconLink style={{ width: 14, height: 14, flexShrink: 0, marginTop: 1, color: 'var(--cyan)' }} />
                <span>Cross-modal query finds document context that enriches structured data answers. Ask a business question and get relevant document passages back.</span>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <input
                  className="form-input"
                  style={{ flex: 1 }}
                  placeholder="e.g. What are the key underwriting risks in our PNC portfolio?"
                  value={queryQ}
                  onChange={e => setQueryQ(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleQuery()}
                />
                <button className="btn btn-primary" onClick={handleQuery} disabled={querying}>
                  {querying ? 'Querying…' : 'Query'}
                </button>
              </div>
              <div style={{ flex: 1, overflow: 'auto' }}>
                {queryResult === null && !querying && (
                  <div className="di-empty">Ask a question to retrieve relevant document context.</div>
                )}
                {queryResult?.error && (
                  <div className="err-msg">{queryResult.error}</div>
                )}
                {queryResult?.doc_context && (
                  <div style={{ background: 'var(--bg-3)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: 14 }}>
                    <div style={{ fontSize: 11, color: 'var(--text-2)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      {queryResult.doc_count} document{queryResult.doc_count !== 1 ? 's' : ''} matched
                    </div>
                    <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'var(--font-sans)', fontSize: 12, color: 'var(--text-0)', lineHeight: 1.7 }}>
                      {queryResult.doc_context}
                    </pre>
                  </div>
                )}
                {queryResult?.doc_count === 0 && (
                  <div className="di-empty">No relevant documents found for this query.</div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Add source modal */}
      {showAdd && (
        <AddSourceModal
          onClose={() => setShowAdd(false)}
          onCreated={(src) => { setShowAdd(false); loadSources(); setSelectedSrc(src.id); }}
        />
      )}
    </div>
  );
}
