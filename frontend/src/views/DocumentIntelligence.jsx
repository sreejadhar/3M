import { useEffect, useRef, useState } from 'react';
import { useAppState } from '../state.jsx';
import { IconDocuments, IconPlus, IconRefresh } from '../components/Icons.jsx';
import {
  docListSources, docCreateSource, docDeleteSource, docStartIndex, docListAssets,
  docGetAsset, docUploadDocument,
} from '../api/clients.js';

const STEP_ICON = { pending: '○', running: '◐', done: '●', error: '✖', skipped: '⊘' };
const STEP_COLOR = {
  pending: 'var(--text-2)', running: 'var(--amber, #f59e0b)',
  done: 'var(--green, #22c55e)', error: 'var(--red, #f87171)', skipped: 'var(--text-2)',
};

const CONNECTORS = [
  ['local', 'Local Filesystem'],
  ['s3', 'Amazon S3'],
  ['gdrive', 'Google Drive'],
  ['sharepoint', 'SharePoint'],
  ['onedrive', 'OneDrive'],
];

const CONNECTOR_FIELDS = {
  local:      [],
  s3:         [['bucket', 'Bucket'], ['prefix', 'Prefix (optional)'], ['region', 'Region (optional)'],
               ['aws_access_key_id', 'Access key ID'], ['aws_secret_access_key', 'Secret access key', 'password']],
  gdrive:     [['folder_id', 'Folder ID'], ['access_token', 'OAuth access token', 'password']],
  sharepoint: [['site_id', 'Site ID'], ['drive_id', 'Drive ID (optional)'], ['access_token', 'Graph access token', 'password']],
  onedrive:   [['user_id', 'User (email or "me")'], ['access_token', 'Graph access token', 'password']],
};

const STATUS_LABEL = {
  ready:    ['● Ready', 'var(--green, #22c55e)'],
  indexing: ['◐ Indexing…', 'var(--amber, #f59e0b)'],
  error:    ['✖ Error', 'var(--red, #f87171)'],
  idle:     ['○ Not indexed', 'var(--text-2)'],
};

function statusView(status) {
  return STATUS_LABEL[status] || STATUS_LABEL.idle;
}

export default function DocumentIntelligence() {
  const { toast } = useAppState();
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedId, setSelectedId] = useState(null);
  const [assets, setAssets] = useState([]);
  const [assetsLoading, setAssetsLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const pollRef = useRef(null);
  const assetPollRef = useRef(null);
  const fileInputRef = useRef(null);
  const selected = sources.find((s) => s.source_id === selectedId);

  async function refresh() {
    try {
      const data = await docListSources();
      setSources(data);
    } catch (e) {
      toast(`Failed to load sources: ${e.message}`, 'error');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    return () => clearInterval(pollRef.current);
  }, []);

  useEffect(() => {
    const anyIndexing = sources.some((s) => s.status === 'indexing');
    clearInterval(pollRef.current);
    if (anyIndexing) {
      pollRef.current = setInterval(refresh, 3000);
    }
    return () => clearInterval(pollRef.current);
  }, [sources]);

  async function openAssets(source) {
    setSelectedId(source.source_id);
    setAssetsLoading(true);
    try {
      const data = await docListAssets(source.source_id);
      setAssets(data);
    } catch (e) {
      toast(`Failed to load assets: ${e.message}`, 'error');
      setAssets([]);
    } finally {
      setAssetsLoading(false);
    }
  }

  // While the selected source is still indexing, re-fetch its full asset
  // list periodically — a bulk reindex enumerates + processes files one at a
  // time, so newly-appearing files (and their processing_status flipping
  // from 'none' to 'running') wouldn't otherwise be picked up.
  useEffect(() => {
    if (!selected || selected.status !== 'indexing') return;
    const t = setInterval(() => docListAssets(selected.source_id).then(setAssets).catch(() => {}), 3000);
    return () => clearInterval(t);
  }, [selected?.source_id, selected?.status]);

  // Poll assets that are still processing (extract/embed/tag), updating just
  // those rows in place so the pipeline steps update live in the table.
  useEffect(() => {
    const pending = assets.filter((a) => a.processing_status === 'running');
    clearInterval(assetPollRef.current);
    if (pending.length === 0) return;
    assetPollRef.current = setInterval(async () => {
      const updates = await Promise.all(pending.map((a) => docGetAsset(a.asset_id).catch(() => null)));
      setAssets((prev) => prev.map((a) => updates.find((u) => u && u.asset_id === a.asset_id) || a));
    }, 1500);
    return () => clearInterval(assetPollRef.current);
  }, [assets]);

  async function uploadFile(file) {
    if (!selected) return;
    setUploading(true);
    try {
      const asset = await docUploadDocument(selected.source_id, file);
      toast(`Uploaded "${file.name}" — processing started`, 'info');
      setAssets((prev) => {
        const rest = prev.filter((a) => a.asset_id !== asset.asset_id);
        return [asset, ...rest];
      });
    } catch (e) {
      toast(`Upload failed: ${e.message}`, 'error');
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }

  async function reindex(source) {
    try {
      await docStartIndex(source.source_id);
      toast(`Indexing started for "${source.name}"`, 'info');
      refresh();
    } catch (e) {
      toast(`Reindex failed: ${e.message}`, 'error');
    }
  }

  async function remove(source) {
    if (!window.confirm(`Delete source "${source.name}"? This removes its indexed assets too.`)) return;
    try {
      await docDeleteSource(source.source_id);
      toast(`Deleted "${source.name}"`, 'info');
      if (selectedId === source.source_id) { setSelectedId(null); setAssets([]); }
      refresh();
    } catch (e) {
      toast(`Delete failed: ${e.message}`, 'error');
    }
  }

  return (
    <div id="view-documents" className="view active">
      <div className="panel-header">
        <IconDocuments width="16" height="16" />
        Document Intelligence
        <div className="panel-actions">
          <button className="btn btn-secondary" onClick={refresh} title="Refresh">
            <IconRefresh width="14" height="14" />
          </button>
          <button className="btn btn-primary" onClick={() => setModalOpen(true)}>
            <IconPlus width="14" height="14" />
            Add Source
          </button>
        </div>
      </div>

      <div style={{ display: 'flex', height: 'calc(100% - 44px)' }}>
        {/* ── Source list ─────────────────────────────────────────── */}
        <div style={{ width: 340, borderRight: '1px solid var(--border)', overflowY: 'auto' }}>
          {loading && <div className="empty-state" style={{ fontSize: 13 }}>Loading…</div>}
          {!loading && sources.length === 0 && (
            <div className="empty-state" style={{ fontSize: 13, padding: 24, textAlign: 'center' }}>
              No document sources yet.<br />Click <b>Add Source</b> to connect one.
            </div>
          )}
          {sources.map((s) => {
            const [label, color] = statusView(s.status);
            const connectorLabel = CONNECTORS.find((c) => c[0] === s.connector_type)?.[1] || s.connector_type;
            return (
              <div
                key={s.source_id}
                onClick={() => openAssets(s)}
                style={{
                  padding: '12px 16px',
                  borderBottom: '1px solid var(--border)',
                  cursor: 'pointer',
                  background: selectedId === s.source_id ? 'var(--bg-1)' : 'transparent',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                  <span style={{ fontWeight: 600, fontSize: 13 }}>{s.name}</span>
                  <span className="badge" style={{ fontSize: 10 }}>{connectorLabel}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                  <span style={{ fontSize: 11, color }}>{label}</span>
                  <span style={{ fontSize: 11, color: 'var(--text-2)' }}>{s.table_count} files</span>
                </div>
                <div style={{ marginTop: 6, display: 'flex', gap: 8 }}>
                  <button
                    className="btn btn-ghost"
                    style={{ fontSize: 11, padding: '2px 8px' }}
                    disabled={s.status === 'indexing'}
                    onClick={(e) => { e.stopPropagation(); reindex(s); }}
                  >
                    Reindex
                  </button>
                  <button
                    className="btn btn-ghost"
                    style={{ fontSize: 11, padding: '2px 8px', color: 'var(--red, #f87171)' }}
                    onClick={(e) => { e.stopPropagation(); remove(s); }}
                  >
                    Delete
                  </button>
                </div>
                {s.status === 'error' && s.error_message && (
                  <div style={{ fontSize: 10, color: 'var(--red, #f87171)', marginTop: 4 }}>
                    {s.error_message}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* ── Asset list ──────────────────────────────────────────── */}
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {!selected && (
            <div className="empty-state" style={{ fontSize: 13 }}>
              <IconDocuments width="36" height="36" strokeWidth="1.5" style={{ opacity: 0.25 }} />
              Select a source to view its indexed files.
            </div>
          )}

          {selected && (
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '8px 16px', borderBottom: '1px solid var(--border)',
            }}>
              <span style={{ fontSize: 12, color: 'var(--text-2)' }}>
                Upload a document to run it through text extraction, topic tagging, named entity recognition, PII detection & semantic embeddings — the relevant database(s) are then detected automatically and cross-modal links are inferred.
              </span>
              <div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".txt,.md,.csv,.html,.htm,.rst,.docx,.doc,.pdf,.pptx,.ppt,.xlsx,.xls"
                  style={{ display: 'none' }}
                  onChange={(e) => e.target.files[0] && uploadFile(e.target.files[0])}
                />
                <button
                  className="btn btn-primary"
                  disabled={uploading}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <IconPlus width="14" height="14" />
                  {uploading ? 'Uploading…' : 'Upload Document'}
                </button>
              </div>
            </div>
          )}

          {selected && !assetsLoading && assets.length > 0 && (
            <ProcessingSummary assets={assets} sourceStatus={selected.status} />
          )}

          {selected && assetsLoading && (
            <div className="empty-state" style={{ fontSize: 13 }}>Loading files…</div>
          )}
          {selected && !assetsLoading && assets.length === 0 && (
            <div className="empty-state" style={{ fontSize: 13 }}>
              No files yet. Click <b>Reindex</b> to enumerate the source, or <b>Upload Document</b> to add one directly.
            </div>
          )}
          {selected && !assetsLoading && assets.length > 0 && (
            <table className="data-table">
              <thead>
                <tr>
                  <th>File</th>
                  <th>Size</th>
                  <th>Processing</th>
                  <th>Topics</th>
                  <th>Entities</th>
                  <th>PII</th>
                  <th>Database Links</th>
                </tr>
              </thead>
              <tbody>
                {assets.map((a) => (
                  <tr key={a.asset_id}>
                    <td>{a.file_name}</td>
                    <td>{fmtBytes(a.size_bytes)}</td>
                    <td>
                      {a.processing_steps && a.processing_steps.length > 0 ? (
                        <div style={{ display: 'flex', gap: 10 }}>
                          {a.processing_steps.map((s) => (
                            <span
                              key={s.key}
                              title={s.detail || s.label}
                              style={{ fontSize: 11, color: STEP_COLOR[s.status], whiteSpace: 'nowrap' }}
                            >
                              {STEP_ICON[s.status]} {s.label}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <span style={{ fontSize: 11, color: 'var(--text-2)' }}>— indexed only —</span>
                      )}
                    </td>
                    <td>
                      {(a.topics || []).length === 0
                        ? <span style={{ color: 'var(--text-2)' }}>—</span>
                        : (
                          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                            {a.topics.map((t) => (
                              <span key={t} className="badge" style={{ fontSize: 10 }}>{t}</span>
                            ))}
                          </div>
                        )}
                    </td>
                    <td>
                      {(a.entities || []).length === 0
                        ? <span style={{ color: 'var(--text-2)' }}>—</span>
                        : (
                          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                            {a.entities.slice(0, 8).map((e, i) => (
                              <span key={`${e.text}-${i}`} className="badge" style={{ fontSize: 10 }}>
                                {e.type}: {e.text}
                              </span>
                            ))}
                            {a.entities.length > 8 && (
                              <span style={{ fontSize: 10, color: 'var(--text-2)' }}>
                                +{a.entities.length - 8} more
                              </span>
                            )}
                          </div>
                        )}
                    </td>
                    <td>
                      {(a.pii_findings || []).length === 0
                        ? <span style={{ color: 'var(--text-2)' }}>—</span>
                        : (
                          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                            {a.pii_findings.map((p, i) => (
                              <span
                                key={`${p.type}-${i}`}
                                className="badge"
                                style={{ fontSize: 10, color: 'var(--red, #f87171)', borderColor: 'var(--red, #f87171)' }}
                              >
                                ⚠ {p.type}: {p.masked}
                              </span>
                            ))}
                          </div>
                        )}
                    </td>
                    <td>
                      {(a.xref_links || []).length === 0
                        ? <span style={{ color: 'var(--text-2)' }}>—</span>
                        : (
                          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                            {a.xref_links.map((l, i) => (
                              <span
                                key={`${l.mention}-${i}`}
                                className="badge"
                                title={`Auto-detected: "${l.mention}" (${l.mention_type}) → ${l.source_name || l.source_id || 'database'}: ${l.matched_table}${l.matched_column ? '.' + l.matched_column : ''} — confidence ${l.confidence} (${l.basis})`}
                                style={{ fontSize: 10 }}
                              >
                                {l.source_name ? `[${l.source_name}] ` : ''}{l.mention} → {l.matched_table}{l.matched_column ? `.${l.matched_column}` : ''}
                              </span>
                            ))}
                          </div>
                        )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {modalOpen && (
        <AddDocSourceModal
          onClose={() => setModalOpen(false)}
          onCreated={() => { setModalOpen(false); refresh(); }}
        />
      )}
    </div>
  );
}

function ProcessingSummary({ assets, sourceStatus }) {
  const counts = { none: 0, running: 0, done: 0, error: 0, skipped: 0 };
  for (const a of assets) {
    const s = a.processing_status || 'none';
    counts[s] = (counts[s] || 0) + 1;
  }
  const stillWorking = sourceStatus === 'indexing' || counts.running > 0;

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 14, padding: '8px 16px',
      borderBottom: '1px solid var(--border)', fontSize: 12, background: 'var(--bg-1)',
    }}>
      <strong style={{ fontSize: 12 }}>
        {stillWorking ? '◐ Processing…' : '● All files processed'}
      </strong>
      {counts.done > 0 && <span style={{ color: STEP_COLOR.done }}>● {counts.done} done</span>}
      {counts.running > 0 && <span style={{ color: STEP_COLOR.running }}>◐ {counts.running} running</span>}
      {counts.error > 0 && <span style={{ color: STEP_COLOR.error }}>✖ {counts.error} error</span>}
      {counts.skipped > 0 && <span style={{ color: 'var(--text-2)' }}>⊘ {counts.skipped} skipped</span>}
      {counts.none > 0 && <span style={{ color: 'var(--text-2)' }}>○ {counts.none} not processed</span>}
    </div>
  );
}

function fmtBytes(n) {
  if (n == null) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function AddDocSourceModal({ onClose, onCreated }) {
  const { toast } = useAppState();
  const [name, setName] = useState('');
  const [connectorType, setConnectorType] = useState('local');
  const [config, setConfig] = useState({});
  const [busy, setBusy] = useState(false);
  const [localFiles, setLocalFiles] = useState([]);
  const localFileInputRef = useRef(null);

  const fields = CONNECTOR_FIELDS[connectorType] || [];

  function onTypeChange(t) {
    setConnectorType(t);
    setConfig({});
    setLocalFiles([]);
  }

  function addLocalFiles(fileList) {
    const picked = Array.from(fileList);
    setLocalFiles((prev) => {
      const key = (f) => `${f.name}:${f.size}`;
      const seen = new Set(prev.map(key));
      return [...prev, ...picked.filter((f) => !seen.has(key(f)))];
    });
  }

  function removeLocalFile(index) {
    setLocalFiles((prev) => prev.filter((_, i) => i !== index));
  }

  async function submit() {
    if (!name.trim()) {
      toast('Source name is required', 'warn');
      return;
    }
    if (connectorType === 'local' && localFiles.length === 0) {
      toast('Choose at least one file to upload', 'warn');
      return;
    }
    setBusy(true);
    try {
      if (connectorType === 'local') {
        const created = await docCreateSource({ name: name.trim(), connector_type: connectorType, config: {} });
        let failed = 0;
        for (const file of localFiles) {
          try {
            await docUploadDocument(created.source_id, file);
          } catch (e) {
            failed += 1;
            toast(`"${file.name}" failed to upload: ${e.message}`, 'error');
          }
        }
        const okCount = localFiles.length - failed;
        toast(`Source "${name.trim()}" added — ${okCount} file(s) uploaded${failed ? `, ${failed} failed` : ''}`,
          failed ? 'warn' : 'success');
      } else {
        const created = await docCreateSource({ name: name.trim(), connector_type: connectorType, config });
        await docStartIndex(created.source_id);
        toast(`Source "${name.trim()}" added — indexing started`, 'success');
      }
      onCreated();
    } catch (e) {
      toast(`Add source failed: ${e.message}`, 'error');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div id="modal-overlay" className="open" onMouseDown={(e) => e.target.id === 'modal-overlay' && onClose()}>
      <div className="modal">
        <div className="modal-title">
          <IconDocuments width="18" height="18" />
          Connect Document Source
        </div>

        <div className="form-row">
          <label>Source Name</label>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Contracts Repository" />
        </div>

        <div className="form-row">
          <label>Connector</label>
          <select value={connectorType} onChange={(e) => onTypeChange(e.target.value)}>
            {CONNECTORS.map(([v, label]) => (
              <option key={v} value={v}>{label}</option>
            ))}
          </select>
        </div>

        {connectorType === 'local' && (
          <div className="form-row">
            <label>Files</label>
            <div>
              <input
                ref={localFileInputRef}
                type="file"
                multiple
                accept=".txt,.md,.csv,.html,.htm,.rst,.docx,.doc,.pdf,.pptx,.ppt,.xlsx,.xls"
                style={{ display: 'none' }}
                onChange={(e) => { addLocalFiles(e.target.files); e.target.value = ''; }}
              />
              <button type="button" className="btn btn-secondary" onClick={() => localFileInputRef.current?.click()}>
                Upload Files
              </button>
              {localFiles.length > 0 && (
                <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {localFiles.map((f, i) => (
                    <div key={`${f.name}-${f.size}-${i}`} style={{
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                      fontSize: 12, padding: '4px 8px', background: 'var(--bg-1)', borderRadius: 'var(--radius)',
                    }}>
                      <span>{f.name} <span style={{ color: 'var(--text-2)' }}>({fmtBytes(f.size)})</span></span>
                      <button
                        type="button"
                        className="btn btn-ghost"
                        style={{ fontSize: 11, padding: '0 6px', color: 'var(--red, #f87171)' }}
                        onClick={() => removeLocalFile(i)}
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {fields.map(([key, label, type]) => (
          <div className="form-row" key={key}>
            <label>{label}</label>
            <input
              type={type === 'password' ? 'password' : 'text'}
              value={config[key] || ''}
              onChange={(e) => setConfig((c) => ({ ...c, [key]: e.target.value }))}
            />
          </div>
        ))}

        <div className="modal-actions">
          <button className="btn btn-ghost" onClick={onClose} disabled={busy}>Cancel</button>
          <button className="btn btn-primary" onClick={submit} disabled={busy}>
            <IconPlus width="14" height="14" />
            {busy ? 'Working…' : (connectorType === 'local' ? 'Connect & Upload' : 'Connect & Index')}
          </button>
        </div>
      </div>
    </div>
  );
}

