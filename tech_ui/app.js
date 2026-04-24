'use strict';

// ── API base — same origin (served through tech_ui_server.py proxy) ───────────
const API = '/api';

// ── State ──────────────────────────────────────────────────────────────────────
let _sources      = [];       // [{id, name, status, db_type, ...}]
let _selectedSrc  = null;     // source id currently selected in pipeline view
let _sseConn      = null;     // EventSource for pipeline events
let _kgNetwork    = null;     // vis.js network instance
let _catalogEntities = [];    // md_entities for catalog view
let _catalogAttrs    = [];    // attributes for selected table
let _currentView  = 'pipeline';

// ─────────────────────────────────────────────────────────────────────────────
// Navigation
// ─────────────────────────────────────────────────────────────────────────────
function switchView(view) {
  _currentView = view;
  document.querySelectorAll('.nav-item').forEach(el => el.classList.toggle('active', el.dataset.view === view));
  document.querySelectorAll('.view').forEach(el => el.classList.toggle('active', el.id === `view-${view}`));

  const titles = {
    pipeline:   ['Pipeline Monitor',   'All data sources'],
    graph:      ['Graph Explorer',     'Knowledge graph visualisation'],
    catalog:    ['Schema Catalog',     'Tables, columns & taxonomy'],
    ontology:   ['Ontology Viewer',    'OWL/Turtle ontology content'],
    redundancy: ['Redundancies',       'Cross-source schema overlap'],
    sql:        ['SQL Console',        'Query any connected source'],
    cdc:        ['Change Log',         'Structural schema change events'],
    documents:  ['Document Intelligence', 'Unstructured data — semantic fingerprints & cross-modal links'],
  };
  const [title, sub] = titles[view] || ['', ''];
  document.getElementById('topbar-title').textContent = title;
  document.getElementById('topbar-sub').textContent = sub;

  if (view === 'graph')      onViewGraph();
  if (view === 'catalog')    onViewCatalog();
  if (view === 'ontology')   onViewOntology();
  if (view === 'redundancy') loadRedundancies();
  if (view === 'sql')        onViewSQL();
  if (view === 'cdc')        loadCDC();
  if (view === 'documents')  loadDocSources();
}

function refreshCurrentView() { switchView(_currentView); }

// ─────────────────────────────────────────────────────────────────────────────
// Toast notifications
// ─────────────────────────────────────────────────────────────────────────────
function toast(msg, type = 'info', duration = 3500) {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), duration);
}

// ─────────────────────────────────────────────────────────────────────────────
// API helpers
// ─────────────────────────────────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const r = await fetch(API + path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try { const b = await r.json(); detail = b.detail || b.error || detail; } catch {}
    throw new Error(detail);
  }
  return r.json();
}

// Timestamps: API returns Unix seconds (float). JS Date needs milliseconds.
function _toMs(val) {
  if (!val) return null;
  const n = Number(val);
  // If it looks like Unix seconds (< year 3000 in seconds), multiply by 1000
  return n < 9999999999 ? n * 1000 : n;
}
function _fmtTime(val) {
  const ms = _toMs(val);
  if (!ms) return '—';
  try { return new Date(ms).toLocaleString(); } catch { return String(val); }
}
function _fmtRelTime(val) {
  const ms = _toMs(val);
  if (!ms) return '';
  const s = Math.floor((Date.now() - ms) / 1000);
  if (s < 0)     return 'just now';
  if (s < 60)    return `${s}s ago`;
  if (s < 3600)  return `${Math.floor(s/60)}m ago`;
  if (s < 86400) return `${Math.floor(s/3600)}h ago`;
  return `${Math.floor(s/86400)}d ago`;
}
function _esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ─────────────────────────────────────────────────────────────────────────────
// Source management
// ─────────────────────────────────────────────────────────────────────────────
async function loadSources() {
  try {
    _sources = await apiFetch('/sources');
    renderSourceList();
    populateSourceSelects();
    document.getElementById('badge-sources').textContent = _sources.length;
  } catch (e) { toast('Failed to load sources: ' + e.message, 'error'); }
}

function renderSourceList() {
  const el = document.getElementById('source-list');
  if (!_sources.length) {
    el.innerHTML = `<div class="empty-state" style="height:200px">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="width:32px;height:32px;opacity:0.3"><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>
      <span>No sources registered</span></div>`;
    return;
  }
  el.innerHTML = _sources.map(s => {
    // API uses "ready" for a fully indexed source
    const statusClass = s.status === 'ready' ? 'indexed' : s.status === 'indexing' ? 'indexing' : s.status === 'error' ? 'error' : 'pending';
    const dbIcon = _dbIcon(s.db_type);
    const sel = s.id === _selectedSrc ? 'selected' : '';
    return `<div class="source-card ${sel}" onclick="selectSource('${_esc(s.id)}')">
      <div class="src-icon">${dbIcon}</div>
      <div style="flex:1;min-width:0;">
        <div class="src-name">${_esc(s.name)}</div>
        <div class="src-meta">${_esc(s.db_type)} · ${s.table_count ?? 0} tables${s.domain && s.domain !== 'Other' ? ` · <span style="color:var(--accent)">${_esc(s.domain)}</span>` : ''}</div>
      </div>
      <div style="display:flex;flex-direction:column;align-items:flex-end;gap:4px;">
        <span class="status-dot ${statusClass}"></span>
        <span style="font-size:10px;color:var(--text-2)">${_fmtRelTime(s.indexed_at)}</span>
      </div>
    </div>`;
  }).join('');
}

function _dbIcon(dbType) {
  const t = (dbType || '').toLowerCase();
  const colors = { postgres:'#336791', redshift:'#cc2264', mysql:'#4479a1', snowflake:'#29b5e8', bigquery:'#4285f4', sqlite:'#003b57', csv:'#3fb950', oracle:'#f80000', sqlserver:'#cc2264' };
  const col = colors[t] || '#8b949e';
  const letter = t[0]?.toUpperCase() || 'D';
  return `<div style="width:28px;height:28px;border-radius:6px;background:${col}22;border:1px solid ${col}44;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:${col}">${letter}</div>`;
}

function populateSourceSelects() {
  const opts = _sources.map(s => `<option value="${_esc(s.id)}">${_esc(s.name)}</option>`).join('');
  const placeholder = '<option value="">— select source —</option>';
  ['global-source-select','graph-source-select','ontology-source-select','sql-source-select'].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      const cur = el.value;
      el.innerHTML = placeholder + opts;
      if (cur) el.value = cur;
    }
  });
}

async function selectSource(sourceId) {
  _selectedSrc = sourceId;
  renderSourceList();

  // Sync global selector
  document.getElementById('global-source-select').value = sourceId;

  const src = _sources.find(s => s.id === sourceId);
  if (!src) return;

  // Show detail header
  const hdr = document.getElementById('pipeline-detail-header');
  hdr.style.display = 'block';
  document.getElementById('detail-src-name').textContent = src.name;
  document.getElementById('detail-src-conn').textContent = `${src.db_type}`;

  const stats = [];
  if (src.table_count) stats.push(`<span><b>${src.table_count}</b> tables</span>`);
  // Domain — always shown, always editable (pencil icon)
  const domainDisplay = (src.domain && src.domain !== 'Other') ? _esc(src.domain) : '<span style="color:var(--text-2);font-style:italic">None</span>';
  stats.push(`<span>Domain:&nbsp;<b id="domain-label" style="color:var(--accent)">${domainDisplay}</b>&nbsp;<button title="Edit domain" onclick="editDomain('${_esc(src.id)}')" style="background:none;border:none;cursor:pointer;padding:0 2px;color:var(--text-2);font-size:11px;vertical-align:middle">✎</button></span>`);
  if (src.status) stats.push(`<span>Status: <b style="color:${src.status==='ready'?'var(--green)':src.status==='indexing'?'var(--accent)':'var(--red)'}">${src.status}</b></span>`);
  if (src.indexed_at) stats.push(`<span>Last indexed: <b>${_fmtTime(src.indexed_at)}</b></span>`);
  document.getElementById('detail-src-stats').innerHTML = stats.join('&nbsp;&nbsp;·&nbsp;&nbsp;');

  // Subscribe to SSE
  startSSE(sourceId);
}

function editDomain(sourceId) {
  const src = _sources.find(s => s.id === sourceId);
  if (!src) return;
  const current = (src.domain && src.domain !== 'Other') ? src.domain : '';
  const label = document.getElementById('domain-label');
  if (!label) return;
  // Replace the label with an inline input + save/cancel
  label.outerHTML = `<span id="domain-edit-wrap" style="display:inline-flex;align-items:center;gap:4px">
    <input id="domain-input" type="text" value="${_esc(current)}"
      placeholder="e.g. CPG/FP&amp;A"
      style="font-size:11px;padding:1px 5px;border:1px solid var(--accent);border-radius:4px;background:var(--bg-0);color:var(--text-1);width:140px;"
      onkeydown="if(event.key==='Enter')saveDomain('${_esc(sourceId)}');if(event.key==='Escape')selectSource('${_esc(sourceId)}')"
    />
    <button onclick="saveDomain('${_esc(sourceId)}')" style="background:var(--accent);border:none;border-radius:4px;color:#fff;padding:1px 7px;font-size:11px;cursor:pointer">Save</button>
    <button onclick="selectSource('${_esc(sourceId)}')" style="background:none;border:1px solid var(--border);border-radius:4px;color:var(--text-2);padding:1px 6px;font-size:11px;cursor:pointer">✕</button>
  </span>`;
  document.getElementById('domain-input')?.focus();
}

async function saveDomain(sourceId) {
  const input = document.getElementById('domain-input');
  if (!input) return;
  const newDomain = input.value.trim();
  try {
    const updated = await apiFetch(`/sources/${encodeURIComponent(sourceId)}`, {
      method: 'PATCH',
      body: JSON.stringify({ domain: newDomain || 'Other' }),
    });
    // Update local state
    const idx = _sources.findIndex(s => s.id === sourceId);
    if (idx !== -1) _sources[idx] = { ..._sources[idx], ...updated };
    toast(`Domain updated: ${updated.domain || 'Other'}`, 'success', 3000);
    // Re-render the detail header and source list
    await selectSource(sourceId);
    renderSourceList();
  } catch (e) {
    toast('Failed to save domain: ' + e.message, 'error');
  }
}

function onGlobalSourceChange(sourceId) {
  if (!sourceId) return;
  // Sync all view-specific selectors
  ['graph-source-select','ontology-source-select','sql-source-select'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = sourceId;
  });
  if (_currentView === 'pipeline') selectSource(sourceId);
  else if (_currentView === 'catalog') loadCatalog(sourceId);
  else if (_currentView === 'graph') loadGraph(sourceId);
  else if (_currentView === 'ontology') loadOntology(sourceId);
  else selectSource(sourceId);
}

// ─────────────────────────────────────────────────────────────────────────────
// SSE — pipeline events
// ─────────────────────────────────────────────────────────────────────────────
function startSSE(sourceId) {
  if (_sseConn) { _sseConn.close(); _sseConn = null; }
  clearEventLog();

  const indicator = document.getElementById('sse-indicator');
  indicator.style.background = 'var(--accent)';
  indicator.title = 'Connecting…';

  _sseConn = new EventSource(`/api/sources/${encodeURIComponent(sourceId)}/index-events`);

  _sseConn.onopen = () => {
    indicator.style.background = 'var(--green)';
    indicator.title = 'Connected';
  };

  _sseConn.onmessage = e => {
    try {
      const ev = JSON.parse(e.data);

      // Skip heartbeats and other non-display events
      if (ev.type === 'heartbeat' || (!ev.step && !ev.stage && !ev.message)) return;

      appendEvent(ev);

      // Refresh source card when a step completes or errors
      if (ev.status === 'done' || ev.status === 'error') loadSources();

      // Indexing finished — close so EventSource doesn't reconnect and replay
      if (ev.step === 'complete') {
        _sseConn.close();
        _sseConn = null;
        indicator.style.background = ev.status === 'error' ? 'var(--red)' : 'var(--green)';
        indicator.title = ev.status === 'error' ? 'Failed' : 'Done';
      }
    } catch {}
  };

  _sseConn.onerror = () => {
    indicator.style.background = 'var(--text-2)';
    indicator.title = 'Disconnected';
  };
}

// Step → display label + colour class
const _STEP_META = {
  // coarse orchestrator steps
  extract:           { label: 'EXTRACT',     cls: 'extract' },
  ontology:          { label: 'ONTOLOGY',    cls: 'ontology' },
  kg:                { label: 'KG',          cls: 'kg' },
  taxonomy:          { label: 'TAXONOMY',    cls: 'taxonomy' },
  complete:          { label: 'COMPLETE',    cls: 'complete' },
  // fine-grained pipeline steps
  discover:          { label: 'DISCOVER',    cls: 'discover' },
  'extract:table':   { label: '  ↳ table',   cls: 'sub' },
  fd:                { label: 'FD',          cls: 'fd' },
  'fd:table':        { label: '  ↳ fd',      cls: 'sub' },
  ind:               { label: 'IND',         cls: 'ind' },
  'ind:pair':        { label: '  ↳ ind',     cls: 'sub' },
  cardinality:       { label: 'CARDINALITY', cls: 'cardinality' },
  'cardinality:pair':{ label: '  ↳ card',    cls: 'sub' },
  error:             { label: 'ERROR',       cls: 'error' },
};

function appendEvent(ev) {
  const log = document.getElementById('event-log');
  const empty = log.querySelector('.empty-state');
  if (empty) empty.remove();

  const stepKey  = ev.step || ev.stage || ev.type || 'info';
  const meta     = _STEP_META[stepKey] || { label: stepKey.toUpperCase(), cls: 'info' };
  const isSub    = stepKey.includes(':');

  // Status badge
  const statusIcon = { running:'⟳', done:'✓', error:'✗', warn:'⚠' }[ev.status] || '';
  const statusCls  = { running:'ev-running', done:'ev-done', error:'ev-error', warn:'ev-warn' }[ev.status] || '';

  const row = document.createElement('div');
  row.className = 'event-row' + (isSub ? ' event-row-sub' : '');
  row.innerHTML = `
    <span class="ev-time">${new Date().toLocaleTimeString()}</span>
    <span class="ev-stage ${meta.cls}">${_esc(meta.label)}</span>
    ${statusIcon ? `<span class="ev-status ${statusCls}">${statusIcon}</span>` : ''}
    <span class="ev-msg">${_esc(ev.message || '')}</span>
    ${ev.detail ? `<div class="ev-detail">${_esc(ev.detail)}</div>` : ''}
  `;
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
}

function clearEventLog() {
  document.getElementById('event-log').innerHTML = `<div class="empty-state" style="height:80px"><span>No events yet</span></div>`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Reindex / enrich
// ─────────────────────────────────────────────────────────────────────────────
async function reindexSelected() {
  if (!_selectedSrc) { toast('Select a source first', 'warn'); return; }
  const btn = document.getElementById('btn-reindex');
  btn.disabled = true;
  try {
    await apiFetch(`/sources/${_selectedSrc}/reindex`, { method: 'POST' });
    toast('Reindex started', 'success');
    startSSE(_selectedSrc);
    await loadSources();
  } catch (e) { toast('Reindex failed: ' + e.message, 'error'); }
  finally { btn.disabled = false; }
}

async function enrichTaxonomy() {
  if (!_selectedSrc) { toast('Select a source first', 'warn'); return; }
  try {
    await apiFetch(`/metadata/sources/${_selectedSrc}/enrich-taxonomy`, { method: 'POST' });
    toast('Taxonomy enrichment queued', 'success');
  } catch (e) { toast('Enrich failed: ' + e.message, 'error'); }
}

async function classifyPII() {
  if (!_selectedSrc) { toast('Select a source first', 'warn'); return; }
  try {
    await apiFetch(`/metadata/sources/${_selectedSrc}/classify-pii`, { method: 'POST' });
    toast('PII classification queued — reload table to see results', 'success');
  } catch (e) { toast('PII classify failed: ' + e.message, 'error'); }
}

let _piiFilterActive = false;
function togglePIIFilter() {
  _piiFilterActive = !_piiFilterActive;
  const btn = document.getElementById('btn-pii-filter');
  if (_piiFilterActive) {
    btn.style.background = '#ff4d4d';
    btn.style.color = '#fff';
    renderCatalogColumns(_catalogAttrs.filter(a => a.pii_flag === 'PII'));
  } else {
    btn.style.background = 'transparent';
    btn.style.color = '#ff4d4d';
    renderCatalogColumns(_catalogAttrs);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Add Source Modal
// ─────────────────────────────────────────────────────────────────────────────
function openAddSourceModal() {
  document.getElementById('modal-overlay').classList.add('open');
}
function closeAddSourceModal() {
  document.getElementById('modal-overlay').classList.remove('open');
}
function closeModal(e) {
  if (e.target === document.getElementById('modal-overlay')) closeAddSourceModal();
}

function updateConnFields() {
  const t = document.getElementById('src-type').value;
  const isFile = t === 'sqlite' || t === 'csv';
  document.getElementById('conn-fields').style.display = isFile ? 'none' : '';
  document.getElementById('file-field').style.display = isFile ? '' : 'none';
  // Update default port
  const ports = { postgres:5432, redshift:5439, mysql:3306, oracle:1521, sqlserver:1433, snowflake:443, bigquery:443 };
  const portEl = document.getElementById('src-port');
  if (ports[t] && portEl) portEl.value = ports[t];
}

async function testConnection() {
  const type = document.getElementById('src-type').value;
  if (type === 'sqlite' || type === 'csv') { toast('File sources do not require a connection test', 'info'); return; }
  const payload = {
    db_type: type,
    connection: {
      host: document.getElementById('src-host').value,
      port: parseInt(document.getElementById('src-port').value) || 5432,
      database: document.getElementById('src-db').value,
      schema: document.getElementById('src-schema').value || 'public',
      username: document.getElementById('src-user').value,
      password: document.getElementById('src-password').value,
    }
  };
  try {
    const r = await apiFetch('/sources/test-connection', { method: 'POST', body: JSON.stringify(payload) });
    if (r.ok === false) throw new Error(r.error || 'Connection failed');
    toast(r.message || 'Connection successful', 'success');
  } catch (e) { toast('Connection failed: ' + e.message, 'error'); }
}

async function addSource() {
  const name = document.getElementById('src-name').value.trim();
  const type = document.getElementById('src-type').value;
  if (!name) { toast('Source name is required', 'warn'); return; }
  const isFile = type === 'sqlite' || type === 'csv';

  if (isFile) {
    const file = document.getElementById('src-file').files[0];
    if (!file) { toast('Select a file to upload', 'warn'); return; }

    // Step 1: upload the file to get a stored path
    const fd = new FormData();
    fd.append('file', file);
    let uploadInfo;
    try {
      const r = await fetch(`${API}/sources/upload-file`, { method: 'POST', body: fd });
      if (!r.ok) { const b = await r.json(); throw new Error(b.detail || r.statusText); }
      uploadInfo = await r.json();
    } catch (e) { toast('Upload failed: ' + e.message, 'error'); return; }

    // Step 2: register the source with the returned file path
    const payload = {
      name,
      db_type: uploadInfo.db_type || type,
      connection: { file_path: uploadInfo.path, uploaded: true },
      auto_index: true,
    };
    try {
      const r = await apiFetch('/sources', { method: 'POST', body: JSON.stringify(payload) });
      toast(`Source "${name}" registered and indexing started`, 'success');
      closeAddSourceModal();
      await loadSources();
      if (r.id) selectSource(r.id);
    } catch (e) { toast('Failed to register source: ' + e.message, 'error'); }
    return;
  }

  const payload = {
    name,
    db_type: type,
    connection: {
      host: document.getElementById('src-host').value,
      port: parseInt(document.getElementById('src-port').value) || 5432,
      database: document.getElementById('src-db').value,
      schema: document.getElementById('src-schema').value || 'public',
      username: document.getElementById('src-user').value,
      password: document.getElementById('src-password').value,
    },
    auto_index: true,
  };
  try {
    const r = await apiFetch('/sources', { method: 'POST', body: JSON.stringify(payload) });
    toast(`Source "${name}" registered and indexing started`, 'success');
    closeAddSourceModal();
    await loadSources();
    if (r.id) selectSource(r.id);
  } catch (e) { toast('Failed to add source: ' + e.message, 'error'); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph Explorer
// ─────────────────────────────────────────────────────────────────────────────
function onViewGraph() {
  const sel = document.getElementById('graph-source-select');
  const global = document.getElementById('global-source-select').value;
  if (!global) return;
  sel.value = global;
  // Always load — idempotent; ensures first-visit and source-change both render
  loadGraph(global);
}

async function loadGraph(sourceId) {
  if (!sourceId) return;
  document.getElementById('graph-empty').style.display = 'none';
  document.getElementById('graph-info-panel').classList.remove('visible');

  try {
    const data = await apiFetch(`/sources/${sourceId}/graph`);
    const nodes = data.nodes || [];
    const edges = data.edges || [];

    document.getElementById('g-node-count').textContent = nodes.length;
    document.getElementById('g-edge-count').textContent = edges.length;

    if (!nodes.length) {
      document.getElementById('graph-empty').style.display = 'flex';
      document.getElementById('graph-empty').innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="width:36px;height:36px;opacity:0.3"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/></svg>
        No graph data yet — trigger a reindex to build the knowledge graph`;
      return;
    }

    // Transform nodes for vis.js
    const visNodes = nodes.map(n => ({
      id: n.id,
      label: n.label,
      title: n.title || n.label,
      color: {
        background: '#1c2230',
        border: '#f59e0b',
        highlight: { background: '#21273a', border: '#fbbf24' },
        hover: { background: '#21273a', border: '#fcd34d' },
      },
      font: { color: '#e6edf3', size: 12, face: 'JetBrains Mono, Consolas, monospace' },
      size: n.size || 22,
      borderWidth: 2,
      shadow: { enabled: true, color: 'rgba(245,158,11,0.2)', size: 8 },
    }));

    const visEdges = edges.map(e => ({
      from: e.from,
      to: e.to,
      label: e.label || '',
      title: e.title || e.label,
      color: { color: '#3a4a6a', highlight: '#f59e0b', hover: '#58a6ff' },
      font: {
        color: '#e6edf3',
        size: 11,
        face: 'system-ui, sans-serif',
        background: 'rgba(9,14,26,0.82)',
        strokeWidth: 0,
        align: 'middle',
      },
      arrows: { to: { enabled: true, scaleFactor: 0.7 } },
      smooth: { type: 'curvedCW', roundness: 0.2 },
      width: 1.5,
    }));

    const container = document.getElementById('kg-vis');
    if (_kgNetwork) { _kgNetwork.destroy(); _kgNetwork = null; }

    _kgNetwork = new vis.Network(container, { nodes: visNodes, edges: visEdges }, {
      physics: {
        enabled: true,
        solver: 'forceAtlas2Based',
        forceAtlas2Based: { gravitationalConstant: -120, centralGravity: 0.005, springLength: 200, springConstant: 0.04 },
        stabilization: { iterations: 150 },
      },
      interaction: { hover: true, tooltipDelay: 200, navigationButtons: false, keyboard: false },
      layout: { randomSeed: 42 },
    });

    _kgNetwork.on('click', params => {
      if (!params.nodes.length) { document.getElementById('graph-info-panel').classList.remove('visible'); return; }
      const nodeId = params.nodes[0];
      const node = nodes.find(n => n.id === nodeId);
      if (!node) return;
      showGraphNodeInfo(node);
    });

    _kgNetwork.on('stabilizationIterationsDone', () => {
      _kgNetwork.setOptions({ physics: { enabled: false } });
      _kgNetwork.fit();
    });

  } catch (e) {
    toast('Failed to load graph: ' + e.message, 'error');
    document.getElementById('graph-empty').style.display = 'flex';
  }

  // Render bridges diagram below the graph (non-blocking)
  loadAndRenderGraphBridges(sourceId);
}

// ── KG Bridges diagram ─────────────────────────────────────────────────────────

async function loadAndRenderGraphBridges(sourceId) {
  const canvas  = document.getElementById('graph-bridges-canvas');
  const emptyEl = document.getElementById('graph-bridges-empty');
  const countEl = document.getElementById('graph-bridges-count');
  if (!canvas) return;

  let bridges = [];
  try { bridges = await apiFetch('/kg-bridges'); } catch (_) { bridges = []; }

  // Find the source name to match against KG names
  const srcEl   = document.getElementById('graph-source-select');
  const srcName = srcEl?.options[srcEl.selectedIndex]?.text || sourceId;
  const re      = new RegExp(_escRe(srcName) + '|' + _escRe(sourceId), 'i');
  const relevant = bridges.filter(b =>
    re.test(b.from_kg) || re.test(b.to_kg) ||
    b.from_source_id === sourceId || b.to_source_id === sourceId
  );
  const drawBridges = relevant.length ? relevant : bridges;

  if (!drawBridges.length) {
    canvas.style.display  = 'none';
    emptyEl.style.display = 'flex';
    if (countEl) countEl.textContent = '';
    return;
  }

  emptyEl.style.display = 'none';
  canvas.style.display  = 'block';
  if (countEl) countEl.textContent = drawBridges.length;

  _drawBridgesCanvas(canvas, drawBridges);
}

function _escRe(s) {
  return String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function _drawBridgesCanvas(canvas, bridges) {
  const kgNames = [...new Set(bridges.flatMap(b => [b.from_kg, b.to_kg]))];
  const N       = kgNames.length;

  const H          = 180;
  const NODE_W     = 150;
  const NODE_H     = 40;
  const MIN_COL_W  = NODE_W + 60;
  const totalW     = Math.max(760, N * MIN_COL_W + 60);

  canvas.width  = totalW;
  canvas.height = H;
  canvas.style.height = H + 'px';

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, totalW, H);

  // Background
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg-1').trim() || '#0f1117';
  ctx.fillRect(0, 0, totalW, H);

  const GREEN   = '#3fb950';
  const BLUE    = '#58a6ff';
  const GREY    = '#4b5563';
  const LABEL_C = '#8b949e';
  const NODE_BG = getComputedStyle(document.documentElement).getPropertyValue('--bg-3').trim() || '#1c2230';
  const NODE_BD = '#f59e0b';
  const TEXT_C  = getComputedStyle(document.documentElement).getPropertyValue('--text-0').trim() || '#e6edf3';

  const nodeY = 48;
  const pos = {};
  kgNames.forEach((name, i) => {
    pos[name] = { x: Math.round((totalW / (N + 1)) * (i + 1)), y: nodeY };
  });

  // Draw edges
  bridges.forEach(b => {
    const from = pos[b.from_kg];
    const to   = pos[b.to_kg];
    if (!from || !to) return;

    const disabled   = b.enabled === false;
    const isInferred = b.source === 'inferred';
    const color = disabled ? GREY : isInferred ? BLUE : GREEN;

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth   = disabled ? 1 : 2;
    ctx.globalAlpha = disabled ? 0.4 : 1;
    if (disabled) ctx.setLineDash([5, 4]);

    const x1 = from.x, y1 = from.y + NODE_H;
    const x2 = to.x,   y2 = to.y + NODE_H;
    const mx = (x1 + x2) / 2;
    const cy = y1 + 56 + Math.abs(x2 - x1) * 0.07;

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.quadraticCurveTo(mx, cy, x2, y2);
    ctx.stroke();

    // Arrowhead
    const ang = Math.atan2(y2 - cy, x2 - mx);
    ctx.save();
    ctx.translate(x2, y2);
    ctx.rotate(ang);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(-9, -4);
    ctx.lineTo(-9,  4);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    ctx.restore();

    if (!disabled) {
      const confPct   = Math.round((b.confidence || 0) * 100);
      const edgeLbl   = `${b.from_entity ? b.from_entity + '.' : ''}${b.from_column} → ${b.to_entity ? b.to_entity + '.' : ''}${b.to_column}`;
      const subLbl    = `${b.join_type || 'FK'} · ${confPct}%`;

      ctx.globalAlpha = 1;
      ctx.textAlign   = 'center';
      ctx.textBaseline = 'middle';
      ctx.font        = 'bold 11px system-ui,sans-serif';
      ctx.fillStyle   = color;
      ctx.fillText(edgeLbl, mx, cy + 14);
      ctx.font        = '10px system-ui,sans-serif';
      ctx.fillStyle   = LABEL_C;
      ctx.fillText(subLbl, mx, cy + 27);
    }

    ctx.restore();
  });

  // Draw nodes (on top of edges)
  kgNames.forEach(name => {
    const { x, y } = pos[name];
    const nx = x - NODE_W / 2;
    const ny = y - NODE_H / 2;
    const r  = 7;

    ctx.beginPath();
    ctx.moveTo(nx + r, ny);
    ctx.lineTo(nx + NODE_W - r, ny);
    ctx.arcTo(nx + NODE_W, ny, nx + NODE_W, ny + r, r);
    ctx.lineTo(nx + NODE_W, ny + NODE_H - r);
    ctx.arcTo(nx + NODE_W, ny + NODE_H, nx + NODE_W - r, ny + NODE_H, r);
    ctx.lineTo(nx + r, ny + NODE_H);
    ctx.arcTo(nx, ny + NODE_H, nx, ny + NODE_H - r, r);
    ctx.lineTo(nx, ny + r);
    ctx.arcTo(nx, ny, nx + r, ny, r);
    ctx.closePath();
    ctx.fillStyle   = NODE_BG;
    ctx.fill();
    ctx.strokeStyle = NODE_BD;
    ctx.lineWidth   = 1.5;
    ctx.stroke();

    ctx.font         = 'bold 12px system-ui,sans-serif';
    ctx.fillStyle    = TEXT_C;
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'middle';
    let label = name;
    while (label.length > 4 && ctx.measureText(label + '…').width > NODE_W - 16) {
      label = label.slice(0, -1);
    }
    if (label !== name) label += '…';
    ctx.fillText(label, x, y);
  });
}

function showGraphNodeInfo(node) {
  const panel = document.getElementById('graph-info-panel');
  document.getElementById('gip-title').textContent = node.label;

  // Parse title lines into key-value pairs
  const lines = (node.title || '').split('\n').filter(l => l.trim());
  let html = '';
  for (const line of lines) {
    const colonIdx = line.indexOf(':');
    if (colonIdx > 0 && !line.startsWith('Class') && !line.startsWith('Properties')) {
      const key = line.slice(0, colonIdx).trim();
      const val = line.slice(colonIdx + 1).trim();
      html += `<div class="info-row"><span class="info-label">${_esc(key)}</span><span class="info-value text-mono">${_esc(val)}</span></div>`;
    } else {
      html += `<div style="color:var(--text-2);font-size:10px;margin-bottom:6px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em">${_esc(line)}</div>`;
    }
  }

  document.getElementById('gip-body').innerHTML = html || '<span style="color:var(--text-2);font-size:12px">No details available</span>';
  panel.classList.add('visible');
}

function resetGraphLayout() {
  if (_kgNetwork) {
    _kgNetwork.setOptions({ physics: { enabled: true } });
    setTimeout(() => _kgNetwork.setOptions({ physics: { enabled: false } }), 2000);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Schema Catalog
// ─────────────────────────────────────────────────────────────────────────────
function onViewCatalog() {
  const sourceId = document.getElementById('global-source-select').value;
  if (sourceId) loadCatalog(sourceId);
}

async function loadCatalog(sourceId) {
  if (!sourceId) return;
  try {
    _catalogEntities = await apiFetch(`/metadata/entities?source_id=${encodeURIComponent(sourceId)}`);
    renderCatalogTableList(_catalogEntities);
  } catch (e) { toast('Failed to load catalog: ' + e.message, 'error'); }
}

function renderCatalogTableList(entities) {
  const el = document.getElementById('catalog-table-list');
  if (!entities.length) {
    el.innerHTML = `<div class="empty-state" style="height:200px;font-size:12px;">No tables indexed yet</div>`;
    return;
  }
  el.innerHTML = entities.map(e => {
    const del = e.deleted_from_source ? 'opacity:0.5;' : '';
    return `<div class="table-list-item" style="${del}" onclick="loadCatalogTable('${_esc(e.metadata_id)}')">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:13px;height:13px;color:var(--text-2);flex-shrink:0"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/></svg>
      <div style="min-width:0">
        <div class="tbl-name">${_esc(e.schema_name ? e.schema_name+'.'+e.table_name : e.table_name)}</div>
        <div class="tbl-meta">${e.row_count != null ? _fmtNum(e.row_count)+' rows' : ''} ${e.deleted_from_source ? '· deleted' : ''}</div>
      </div>
      ${e.redundancy_count > 0 ? `<span class="badge badge-amber" style="font-size:9px">${e.redundancy_count}</span>` : ''}
    </div>`;
  }).join('');
}

function filterCatalogTables(q) {
  const lower = q.toLowerCase();
  const filtered = _catalogEntities.filter(e =>
    (e.table_name + (e.schema_name || '')).toLowerCase().includes(lower)
  );
  renderCatalogTableList(filtered);
}

async function loadCatalogTable(metadataId) {
  const entity = _catalogEntities.find(e => e.metadata_id === metadataId);
  if (!entity) return;

  // Highlight selected
  document.querySelectorAll('.table-list-item').forEach(el => {
    el.classList.toggle('selected', el.getAttribute('onclick')?.includes(metadataId));
  });

  try {
    const full = await apiFetch(`/metadata/entities/${metadataId}`);
    _catalogAttrs = full.attributes || [];
    // Reset PII filter when switching tables
    _piiFilterActive = false;
    const piiBtn = document.getElementById('btn-pii-filter');
    if (piiBtn) { piiBtn.style.background = 'transparent'; piiBtn.style.color = '#ff4d4d'; }

    const tblName = entity.schema_name ? `${entity.schema_name}.${entity.table_name}` : entity.table_name;
    document.getElementById('catalog-tbl-name').textContent = tblName;

    const stats = [];
    if (entity.row_count != null) stats.push(`${_fmtNum(entity.row_count)} rows`);
    stats.push(`${_catalogAttrs.length} columns`);
    if (entity.description) stats.push(entity.description);
    document.getElementById('catalog-tbl-stats').textContent = stats.join('  ·  ');
    document.getElementById('catalog-header').style.display = 'block';

    renderCatalogColumns(_catalogAttrs);
  } catch (e) { toast('Failed to load table: ' + e.message, 'error'); }
}

function renderCatalogColumns(attrs) {
  const el = document.getElementById('catalog-columns');
  if (!attrs.length) { el.innerHTML = '<div class="empty-state">No columns found</div>'; return; }
  el.innerHTML = `<table class="data-table">
    <thead>
      <tr>
        <th>Column</th>
        <th>Type</th>
        <th>Statistical Type</th>
        <th>Semantic Role</th>
        <th>Statistics</th>
        <th>Sample Values</th>
        <th>Flags</th>
      </tr>
    </thead>
    <tbody>
      ${attrs.map(a => _colRow(a)).join('')}
    </tbody>
  </table>`;
}

function _colRow(a) {
  const statBadge = a.statistical_type ? `<span class="badge ${_statColor(a.statistical_type)}">${_esc(a.statistical_type)}</span>` : '<span class="text-dim">—</span>';
  const roleBadge = a.semantic_role ? `<span class="badge ${_roleColor(a.semantic_role)}" style="margin-top:3px">${_esc(a.semantic_role.replace(/_/g,' '))}</span>` : '<span class="text-dim">—</span>';
  const flags = [];
  if (a.is_primary_key) flags.push(`<span class="badge badge-amber">PK</span>`);
  if (a.is_foreign_key) flags.push(`<span class="badge badge-purple">FK</span>`);
  if (a.nullable === false) flags.push(`<span class="badge badge-gray">NOT NULL</span>`);
  if (a.is_golden_record) flags.push(`<span class="badge badge-green">Golden</span>`);
  if (a.deleted_from_source) flags.push(`<span class="badge badge-red">Deleted</span>`);
  if (a.pii_flag === 'PII') {
    const piiLabel = a.pii_type ? `PII · ${a.pii_type.replace(/_/g,' ')}` : 'PII';
    flags.push(`<span class="badge badge-pii" title="${_esc(piiLabel)}" style="background:#ff4d4d;color:#fff;font-weight:700;letter-spacing:.3px">🔒 ${_esc(piiLabel)}</span>`);
  } else if (a.pii_flag === 'Non-PII') {
    flags.push(`<span class="badge badge-gray" title="Not PII" style="opacity:.55">Non-PII</span>`);
  }

  const topVals = (a.top_values || []).slice(0, 5);
  const valChips = topVals.map(v => `<span class="top-val-chip">${_esc(String(v))}</span>`).join('');

  const stats = [];
  if (a.unique_count != null) stats.push(`<span title="Distinct values">◈ ${_fmtNum(a.unique_count)}</span>`);
  if (a.null_count != null) stats.push(`<span title="Null count" style="color:${a.null_count>0?'var(--accent)':'inherit'}">∅ ${_fmtNum(a.null_count)}</span>`);
  if (a.avg_value != null) stats.push(`<span title="Average">μ ${parseFloat(a.avg_value).toFixed(2)}</span>`);

  return `<tr>
    <td>
      <div class="mono" style="font-weight:600">${_esc(a.column_name)}</div>
      ${a.description ? `<div class="dim" style="font-size:10px;margin-top:2px">${_esc(a.description)}</div>` : ''}
    </td>
    <td><span class="mono text-muted">${_esc(a.data_type || '—')}</span></td>
    <td>${statBadge}</td>
    <td>${roleBadge}</td>
    <td><div class="col-stats" style="flex-direction:column;gap:2px">${stats.join('')}</div></td>
    <td><div class="col-tags">${valChips || '<span class="text-dim" style="font-size:10px">—</span>'}</div></td>
    <td><div style="display:flex;gap:3px;flex-wrap:wrap">${flags.join('') || '<span class="text-dim">—</span>'}</div></td>
  </tr>`;
}

function filterCatalogColumns(q) {
  const lower = q.toLowerCase();
  renderCatalogColumns(_catalogAttrs.filter(a =>
    (a.column_name + (a.data_type || '') + (a.semantic_role || '') + (a.statistical_type || '')).toLowerCase().includes(lower)
  ));
}

function _statColor(s) {
  const m = { continuous:'badge-blue', categorical:'badge-amber', ordinal:'badge-orange', boolean:'badge-purple', identifier:'badge-gray', date:'badge-cyan', free_text:'badge-gray', nominal:'badge-green' };
  return m[s] || 'badge-gray';
}
function _roleColor(r) {
  const m = { measure:'badge-blue', time_period:'badge-cyan', time_dimension_key:'badge-cyan', product_category:'badge-amber', product_sub_category:'badge-orange', product_dimension_key:'badge-orange', geography:'badge-green', geography_dimension_key:'badge-green', org_unit:'badge-purple', org_dimension_key:'badge-purple', customer_dimension_key:'badge-red', demographic:'badge-orange', identifier:'badge-gray', boolean_flag:'badge-purple', free_text:'badge-gray', other:'badge-gray' };
  return m[r] || 'badge-gray';
}
function _fmtNum(n) { return n == null ? '—' : Number(n).toLocaleString(); }

// ─────────────────────────────────────────────────────────────────────────────
// Ontology Editor
// ─────────────────────────────────────────────────────────────────────────────
let _ontoNetwork = null;
let _ontoSourceId = null;
let _ontoDirty = false;

function onViewOntology() {
  const sel = document.getElementById('ontology-source-select');
  const global = document.getElementById('global-source-select').value;
  if (!global) return;
  sel.value = global;
  loadOntology(global);
}

async function loadOntology(sourceId) {
  if (!sourceId) return;
  _ontoSourceId = sourceId;
  _ontoDirty = false;
  _updateSaveBtn();

  const editor = document.getElementById('ontology-editor');
  editor.value = 'Loading…';
  _renderOntoTree(null);

  try {
    const data = await apiFetch(`/sources/${sourceId}/ontology`);
    const content = data.content || data.ontology_content || '';
    if (!content) {
      editor.value = '';
      editor.placeholder = 'No ontology available — run indexing to generate one.';
      _renderOntoTree({ classes: [], objProps: [], dataProps: [] });
      _renderOntoViz({ classes: [], objProps: [], dataProps: [], subClassOf: [], propDomainRange: [] });
      return;
    }
    editor.value = content;
    const parsed = _parseOntologyText(content);
    _renderOntoTree(parsed);
    _renderOntoViz(parsed);
  } catch (e) {
    editor.value = '';
    toast('Failed to load ontology: ' + e.message, 'error');
  }
}

function onOntologyEdit() {
  _ontoDirty = true;
  _updateSaveBtn();
}

function _updateSaveBtn() {
  const btn = document.getElementById('onto-save-btn');
  if (!btn) return;
  btn.textContent = _ontoDirty ? '● Save & Rebuild KG' : 'Save & Rebuild KG';
  btn.style.opacity = _ontoSourceId ? '1' : '0.4';
}

async function saveOntology() {
  if (!_ontoSourceId) { toast('Select a source first', 'warn'); return; }
  const content = document.getElementById('ontology-editor').value.trim();
  if (!content) { toast('Ontology is empty', 'warn'); return; }
  const btn = document.getElementById('onto-save-btn');
  btn.disabled = true; btn.textContent = 'Saving…';
  try {
    await apiFetch(`/sources/${_ontoSourceId}/ontology`, {
      method: 'POST',
      body: JSON.stringify({ content, rebuild_kg: true }),
    });
    _ontoDirty = false;
    const parsed = _parseOntologyText(content);
    _renderOntoTree(parsed);
    _renderOntoViz(parsed);
    toast('Ontology saved — KG rebuild started', 'success');
  } catch (e) {
    toast('Save failed: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    _updateSaveBtn();
  }
}

function copyOntology() {
  const content = document.getElementById('ontology-editor').value;
  if (!content) { toast('Nothing to copy', 'warn'); return; }
  navigator.clipboard.writeText(content).then(() => toast('Ontology copied to clipboard', 'success'));
}

// ── SHACL Ontology Validation ────────────────────────────────────────────────

// Catalogue of every check the service runs — used to render the full checklist
// including checks that PASSED (not just failures).
const _SHACL_CHECKS = [
  // SHACL structural shapes
  {
    id:          'ClassCompleteness',
    shapeKey:    'ClassCompletenessShape',
    category:    'Structure',
    title:       'Class Completeness',
    description: 'Every table class (owl:Class) must have a human-readable label (rdfs:label) and a description (rdfs:comment). These are used by the NL query planner to identify tables.',
    affectsWhat: 'owl:Class nodes',
  },
  {
    id:          'DatatypePropertyCompleteness',
    shapeKey:    'DatatypePropertyShape',
    category:    'Structure',
    title:       'Column Property Completeness',
    description: 'Every column property (owl:DatatypeProperty) must declare its owning table (rdfs:domain) and XSD data type (rdfs:range). Without these the KG translator cannot map columns to tables.',
    affectsWhat: 'owl:DatatypeProperty nodes',
  },
  {
    id:          'ObjectPropertyCompleteness',
    shapeKey:    'ObjectPropertyShape',
    category:    'Structure',
    title:       'Relationship Completeness',
    description: 'Every FK/IND relationship (owl:ObjectProperty) must declare the source table (rdfs:domain), target table (rdfs:range), and include join column details in rdfs:comment.',
    affectsWhat: 'owl:ObjectProperty nodes',
  },
  {
    id:          'OntologyHeader',
    shapeKey:    'OntologyHeaderShape',
    category:    'Structure',
    title:       'Ontology Header',
    description: 'The owl:Ontology declaration must carry a human-readable name (rdfs:label) so the ontology is self-identifying.',
    affectsWhat: 'owl:Ontology declaration',
  },
  {
    id:          'FunctionalPropertyClassification',
    shapeKey:    'FunctionalPropertyClassificationShape',
    category:    'Structure',
    title:       'Functional Property Typing',
    description: 'owl:FunctionalProperty markers (used for 1:1 and 1:N cardinality) must also be typed as either owl:DatatypeProperty or owl:ObjectProperty so OWL reasoners can classify them correctly.',
    affectsWhat: 'owl:FunctionalProperty nodes',
  },
  // Semantic / Python checks
  {
    id:          'OrphanClass',
    shapeKey:    null,
    category:    'Semantics',
    title:       'Orphan Classes',
    description: 'Classes that are never referenced as domain or range of any property. These suggest missing FK or inclusion-dependency (IND) relationships. Orphan classes produce isolated KG nodes with no edges.',
    affectsWhat: 'owl:Class nodes with no edges',
  },
  {
    id:          'LowCoverage',
    shapeKey:    null,
    category:    'Semantics',
    title:       'Low-Coverage Relationships',
    description: 'ObjectProperty edges where the IND coverage (% of source values found in target) is below the threshold. Low-coverage edges are candidate links that should be confirmed before using as JOIN keys.',
    affectsWhat: 'owl:ObjectProperty edges',
  },
  {
    id:          'NamespaceDrift',
    shapeKey:    null,
    category:    'Semantics',
    title:       'Namespace Consistency',
    description: 'All class and property URIs should share the same base namespace as the owl:Ontology declaration. Mismatched namespaces indicate copy-paste errors and break URI-based lookups.',
    affectsWhat: 'All owl:Class / owl:*Property URIs',
  },
  {
    id:          'DuplicateClassLabel',
    shapeKey:    null,
    category:    'Semantics',
    title:       'Duplicate Class Labels',
    description: 'Two or more classes sharing the same rdfs:label (case-insensitive). The NL query planner uses labels to match table names — duplicates cause ambiguous table resolution.',
    affectsWhat: 'owl:Class rdfs:label values',
  },
];

async function validateOntology() {
  if (!_ontoSourceId) { toast('Select a source first', 'warn'); return; }
  const content = document.getElementById('ontology-editor').value.trim();
  if (!content)       { toast('Ontology is empty', 'warn'); return; }

  const btn = document.getElementById('onto-validate-btn');
  btn.disabled = true;
  btn.textContent = 'Validating…';

  // Open modal immediately in loading state
  _openShaclModal();

  try {
    const report = await apiFetch(`/sources/${_ontoSourceId}/validate-ontology`, {
      method: 'POST',
      body: JSON.stringify({ ontology_text: content }),
    });
    _renderShaclModal(report);
  } catch (e) {
    _renderShaclModalError(e.message || String(e));
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:12px;height:12px"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg> Validate`;
  }
}

function _openShaclModal() {
  const overlay = document.getElementById('shacl-modal-overlay');
  overlay.style.display = 'flex';

  document.getElementById('shacl-modal-badge').textContent = '…';
  document.getElementById('shacl-modal-badge').style.cssText =
    'font-size:11px;font-weight:700;padding:2px 10px;border-radius:3px;background:var(--surface-3);color:var(--text-2);margin-left:4px';
  document.getElementById('shacl-modal-stats').innerHTML = '';
  document.getElementById('shacl-modal-body').innerHTML =
    '<div style="text-align:center;padding:40px;color:var(--text-2)">Running validation checks…</div>';
}

function closeShaclModal(e) {
  if (e && e.target !== document.getElementById('shacl-modal-overlay')) return;
  document.getElementById('shacl-modal-overlay').style.display = 'none';
}

function _renderShaclModalError(msg) {
  const badge = document.getElementById('shacl-modal-badge');
  badge.textContent = 'ERROR';
  badge.style.cssText = 'font-size:11px;font-weight:700;padding:2px 10px;border-radius:3px;background:#7f1d1d;color:#fca5a5;margin-left:4px';
  document.getElementById('shacl-modal-stats').innerHTML = '';
  document.getElementById('shacl-modal-body').innerHTML =
    `<div style="color:#fca5a5;padding:16px 0">${_esc(msg)}</div>`;
}

function _renderShaclModal(r) {
  const quality  = r.quality || 'ERROR';
  const stats    = r.ontology_stats || {};
  const allViol  = r.violations      || [];
  const allWarn  = r.warnings        || [];
  const allSem   = r.semantic_issues || [];
  const allSugg  = r.suggestions     || [];

  // ── Badge ──────────────────────────────────────────────────────────────────
  const badge = document.getElementById('shacl-modal-badge');
  const badgeCss = {
    PASS: 'background:#14532d;color:#86efac',
    WARN: 'background:#713f12;color:#fde68a',
    FAIL: 'background:#7f1d1d;color:#fca5a5',
  }[quality] || 'background:var(--surface-3);color:var(--text-2)';
  badge.textContent = quality;
  badge.style.cssText = `font-size:11px;font-weight:700;padding:2px 10px;border-radius:3px;margin-left:4px;${badgeCss}`;

  // ── Stats bar ──────────────────────────────────────────────────────────────
  const statsBar = document.getElementById('shacl-modal-stats');
  const statItems = [
    { label: 'Classes',       value: stats.classes        ?? '—' },
    { label: 'Relationships', value: stats.object_props   ?? '—' },
    { label: 'Columns',       value: stats.datatype_props ?? '—' },
    { label: 'Triples',       value: stats.triples        ?? '—' },
    { label: 'Violations',    value: allViol.length,  color: allViol.length  ? '#fca5a5' : '#86efac' },
    { label: 'Warnings',      value: allWarn.length + allSem.filter(s => s.severity === 'Warning').length,
                               color: (allWarn.length + allSem.filter(s => s.severity === 'Warning').length) ? '#fde68a' : '#86efac' },
  ];
  statsBar.innerHTML = statItems.map(s => `
    <div style="flex:1;padding:10px 14px;border-right:1px solid var(--border);text-align:center;min-width:0">
      <div style="font-size:18px;font-weight:700;color:${s.color || 'var(--text-0)'}">${s.value}</div>
      <div style="font-size:10px;color:var(--text-2);text-transform:uppercase;letter-spacing:.05em;margin-top:2px">${s.label}</div>
    </div>`).join('');

  // ── Body ───────────────────────────────────────────────────────────────────
  let html = '';

  // ── 1. Suggestions banner (if any) ────────────────────────────────────────
  if (allSugg.length) {
    html += `<div style="background:#1c1a0e;border:1px solid #713f12;border-radius:6px;padding:12px 16px;margin-bottom:18px">
      <div style="font-size:11px;font-weight:700;color:#fde68a;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">
        ⚡ Recommendations
      </div>
      ${allSugg.map(s => `<div style="display:flex;gap:8px;margin-bottom:5px;font-size:12px">
        <span style="color:#f59e0b;flex-shrink:0">→</span>
        <span style="color:var(--text-1)">${_esc(s)}</span>
      </div>`).join('')}
    </div>`;
  }

  // ── 2. Check-by-check results ──────────────────────────────────────────────
  // Build lookup: which issues belong to each check?
  const byShape = {};   // shapeKey → issues from violations+warnings
  for (const item of [...allViol, ...allWarn]) {
    const key = (item.shape || '').split('/').pop().split('#').pop();
    (byShape[key] = byShape[key] || []).push(item);
  }
  const bySemCheck = {};  // check name → semantic issues
  for (const item of allSem) {
    (bySemCheck[item.check] = bySemCheck[item.check] || []).push(item);
  }

  // Group checks by category
  const categories = [...new Set(_SHACL_CHECKS.map(c => c.category))];
  for (const cat of categories) {
    const checks = _SHACL_CHECKS.filter(c => c.category === cat);

    html += `<div style="margin-bottom:22px">
      <div style="font-size:10px;font-weight:700;color:var(--text-2);text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;padding-bottom:5px;border-bottom:1px solid var(--border)">${_esc(cat)} Checks</div>`;

    for (const chk of checks) {
      // Gather issues for this check
      const issues = chk.shapeKey
        ? (byShape[chk.shapeKey] || [])
        : (bySemCheck[chk.id]    || []);

      const hasViolation = issues.some(i => (i.severity || '').toLowerCase() === 'violation' || (i.check && !i.severity));
      const passed       = issues.length === 0;

      const statusIcon  = passed      ? '✓' : hasViolation ? '✕' : '⚠';
      const statusColor = passed      ? '#86efac'
                        : hasViolation ? '#fca5a5'
                        :               '#fde68a';
      const statusLabel = passed      ? 'PASS'
                        : hasViolation ? 'FAIL'
                        :               'WARN';
      const borderColor = passed      ? '#166534'
                        : hasViolation ? '#7f1d1d'
                        :               '#713f12';

      const detailId = `shacl-detail-${chk.id}`;

      html += `<div style="margin-bottom:8px;border:1px solid ${borderColor};border-radius:6px;overflow:hidden">
        <!-- Check header — always visible -->
        <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;background:var(--surface-2);cursor:${issues.length ? 'pointer' : 'default'}"
             onclick="${issues.length ? `_toggleShaclDetail('${detailId}')` : ''}">
          <span style="font-size:13px;font-weight:700;color:${statusColor};width:16px;text-align:center;flex-shrink:0">${statusIcon}</span>
          <div style="flex:1;min-width:0">
            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
              <span style="font-size:12px;font-weight:600;color:var(--text-0)">${_esc(chk.title)}</span>
              <span style="font-size:10px;padding:1px 6px;border-radius:3px;font-weight:700;background:${borderColor}22;color:${statusColor}">${statusLabel}</span>
              ${issues.length ? `<span style="font-size:10px;color:var(--text-2)">${issues.length} issue${issues.length !== 1 ? 's' : ''} · ${_esc(chk.affectsWhat)}</span>` : `<span style="font-size:10px;color:var(--text-2)">${_esc(chk.affectsWhat)}</span>`}
            </div>
            <div style="font-size:11px;color:var(--text-2);margin-top:3px">${_esc(chk.description)}</div>
          </div>
          ${issues.length ? `<span style="color:var(--text-2);font-size:12px;flex-shrink:0" id="${detailId}-arrow">▸</span>` : ''}
        </div>
        <!-- Issues detail — collapsed by default -->
        ${issues.length ? `<div id="${detailId}" style="display:none;padding:10px 14px;border-top:1px solid ${borderColor};background:var(--bg-3)">
          ${issues.map(item => {
            const node = item.node
              ? item.node.split('/').pop().split('#').pop()
              : (item.nodes || []).map(n => n.split('/').pop().split('#').pop()).join(', ');
            const cov  = item.coverage != null ? ` — coverage ${(item.coverage * 100).toFixed(1)}%` : '';
            return `<div style="display:flex;gap:8px;margin-bottom:6px;padding:6px 8px;background:var(--surface-2);border-radius:4px;border-left:3px solid ${statusColor}">
              <div style="flex:1;min-width:0">
                ${node ? `<div style="font-size:10px;font-weight:600;color:${statusColor};margin-bottom:2px;font-family:var(--font-mono)">${_esc(node)}${cov}</div>` : ''}
                <div style="color:var(--text-1);font-size:11px">${_esc(item.message || '')}</div>
              </div>
            </div>`;
          }).join('')}
        </div>` : ''}
      </div>`;
    }

    html += '</div>';
  }

  document.getElementById('shacl-modal-body').innerHTML = html;
}

function _toggleShaclDetail(id) {
  const el    = document.getElementById(id);
  const arrow = document.getElementById(id + '-arrow');
  if (!el) return;
  const open = el.style.display === 'none';
  el.style.display    = open ? 'block' : 'none';
  if (arrow) arrow.textContent = open ? '▾' : '▸';
}

function switchOntoTab(tab) {
  document.getElementById('tab-ttl').classList.toggle('active', tab === 'ttl');
  document.getElementById('tab-sparql').classList.toggle('active', tab === 'sparql');
  document.getElementById('onto-ttl-wrap').classList.toggle('onto-tab-hidden', tab !== 'ttl');
  document.getElementById('onto-sparql-wrap').classList.toggle('onto-tab-hidden', tab !== 'sparql');
}

function toggleOntoViz(btn) {
  const panel = document.getElementById('onto-viz-panel');
  const collapsed = panel.classList.toggle('collapsed');
  if (btn) btn.textContent = collapsed ? '▸' : '▾';
}

// ── Turtle/OWL parser ────────────────────────────────────────────────────────
function _parseOntologyText(text) {
  const result = { classes: [], objProps: [], dataProps: [], subClassOf: [], propDomainRange: [] };
  const seenC = new Set(), seenP = new Set(), seenD = new Set();

  // Classes
  let m;
  for (const re of [/(\w+)\s+a\s+owl:Class/gm, /(\w+)\s+a\s+rdfs:Class/gm]) {
    while ((m = re.exec(text)) !== null) {
      const n = m[1]; if (!seenC.has(n) && n !== 'owl' && n !== 'rdfs') { result.classes.push(n); seenC.add(n); }
    }
  }
  // Also: ":ClassName" style
  const colonClass = /:(\w+)\s+a\s+owl:Class/gm;
  while ((m = colonClass.exec(text)) !== null) {
    const n = m[1]; if (!seenC.has(n)) { result.classes.push(n); seenC.add(n); }
  }

  // Object Properties
  for (const re of [/(\w+)\s+a\s+owl:ObjectProperty/gm, /:(\w+)\s+a\s+owl:ObjectProperty/gm]) {
    while ((m = re.exec(text)) !== null) {
      const n = m[1]; if (!seenP.has(n)) { result.objProps.push(n); seenP.add(n); }
    }
  }

  // Data Properties
  for (const re of [/(\w+)\s+a\s+owl:DatatypeProperty/gm, /:(\w+)\s+a\s+owl:DatatypeProperty/gm]) {
    while ((m = re.exec(text)) !== null) {
      const n = m[1]; if (!seenD.has(n)) { result.dataProps.push(n); seenD.add(n); }
    }
  }

  // SubClassOf
  const scRe = /:?(\w+)\s+rdfs:subClassOf\s+:?(\w+)/gm;
  while ((m = scRe.exec(text)) !== null) result.subClassOf.push({ child: m[1], parent: m[2] });

  // Domain / Range (collect per property)
  const domMap = {};
  const domRe = /:?(\w+)\s+rdfs:domain\s+:?(\w+)/gm;
  while ((m = domRe.exec(text)) !== null) { domMap[m[1]] = domMap[m[1]] || {}; domMap[m[1]].domain = m[2]; }
  const rangeRe = /:?(\w+)\s+rdfs:range\s+:?(\w+)/gm;
  while ((m = rangeRe.exec(text)) !== null) { domMap[m[1]] = domMap[m[1]] || {}; domMap[m[1]].range = m[2]; }
  for (const [prop, dr] of Object.entries(domMap)) result.propDomainRange.push({ prop, ...dr });

  return result;
}

// ── Ontology Tree ────────────────────────────────────────────────────────────
function _renderOntoTree(parsed) {
  const tree = document.getElementById('onto-tree');
  if (!parsed) { tree.innerHTML = '<div class="empty-state" style="height:120px;font-size:11px;">Loading…</div>'; return; }
  if (!parsed.classes.length && !parsed.objProps.length && !parsed.dataProps.length) {
    tree.innerHTML = '<div class="empty-state" style="height:120px;font-size:11px;">No classes or properties found</div>';
    return;
  }

  const section = (label, icon, items, dotClass, clickFn) => {
    if (!items.length) return '';
    const rows = items.map(n =>
      `<div class="onto-item" onclick="${clickFn ? `${clickFn}('${_esc(n)}', this)` : ''}">
        <span class="onto-dot ${dotClass}"></span><span class="onto-item-label">${_esc(n)}</span>
      </div>`
    ).join('');
    return `<div class="onto-section">
      <div class="onto-section-hdr" onclick="this.nextElementSibling.classList.toggle('collapsed')">
        <span class="onto-section-toggle">▾</span>${icon}<span>${label}</span>
        <span class="badge" style="margin-left:auto">${items.length}</span>
      </div>
      <div class="onto-section-body">${rows}</div>
    </div>`;
  };

  // Virtual Graphs = sources with status "ready"
  const readySources = _sources.filter(s => s.status === 'ready');
  const vgRows = readySources.map(s =>
    `<div class="onto-item" onclick="document.getElementById('ontology-source-select').value='${_esc(s.id)}'; loadOntology('${_esc(s.id)}')">
      <span class="onto-dot vg-dot"></span><span class="onto-item-label">${_esc(s.name)}</span>
    </div>`
  ).join('');
  const vgSection = readySources.length ? `<div class="onto-section">
    <div class="onto-section-hdr" onclick="this.nextElementSibling.classList.toggle('collapsed')">
      <span class="onto-section-toggle">▾</span>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:13px;height:13px"><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>
      <span>Virtual Graphs</span>
      <span class="badge" style="margin-left:auto">${readySources.length}</span>
    </div>
    <div class="onto-section-body">${vgRows}</div>
  </div>` : '';

  const classIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:13px;height:13px"><circle cx="12" cy="12" r="8"/></svg>`;
  const propIcon  = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:13px;height:13px"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>`;
  const dataIcon  = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:13px;height:13px"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>`;

  tree.innerHTML =
    section('Classes', classIcon, parsed.classes, 'class-dot', 'ontoItemClick') +
    section('Object Properties', propIcon, parsed.objProps, 'prop-dot', null) +
    section('Data Properties', dataIcon, parsed.dataProps, 'data-dot', null) +
    vgSection;
}

function ontoItemClick(className, el) {
  // Scroll editor to first occurrence of the class name
  const editor = document.getElementById('ontology-editor');
  const text = editor.value;
  const idx = text.indexOf(className);
  if (idx >= 0) {
    editor.focus();
    editor.setSelectionRange(idx, idx + className.length);
    const lines = text.substring(0, idx).split('\n').length;
    editor.scrollTop = Math.max(0, (lines - 5) * 22);
  }
  // Highlight in tree
  document.querySelectorAll('.onto-item').forEach(e => e.classList.remove('active'));
  if (el) el.classList.add('active');
}

// ── Ontology class graph ─────────────────────────────────────────────────────
function _renderOntoViz(parsed) {
  const container = document.getElementById('onto-vis');
  if (_ontoNetwork) { _ontoNetwork.destroy(); _ontoNetwork = null; }

  const classSet = new Set(parsed.classes);
  if (!classSet.size) return;

  const visNodes = parsed.classes.map(c => ({
    id: c, label: c,
    color: { background: '#111827', border: '#58a6ff',
             highlight: { background: '#1c2230', border: '#79c0ff' },
             hover: { background: '#1c2230', border: '#93c5fd' } },
    font: { color: '#e6edf3', size: 11, face: 'JetBrains Mono, Consolas, monospace' },
    shape: 'ellipse', size: 24, borderWidth: 2,
    shadow: { enabled: true, color: 'rgba(88,166,255,0.15)', size: 8 },
  }));

  const visEdges = [];
  // SubClassOf (dashed)
  parsed.subClassOf.forEach(sc => {
    if (classSet.has(sc.child) && classSet.has(sc.parent))
      visEdges.push({ from: sc.child, to: sc.parent, label: 'subClassOf',
        color: { color: '#3a4a6a', highlight: '#8b949e' },
        font: { color: '#8b949e', size: 9 }, dashes: true,
        arrows: { to: { enabled: true, scaleFactor: 0.6 } }, smooth: { type: 'curvedCW', roundness: 0.2 } });
  });
  // Object properties (solid amber)
  parsed.propDomainRange.forEach(p => {
    if (p.domain && p.range && classSet.has(p.domain) && classSet.has(p.range))
      visEdges.push({ from: p.domain, to: p.range, label: p.prop,
        color: { color: '#f59e0b88', highlight: '#f59e0b' },
        font: { color: '#f59e0b', size: 9, strokeWidth: 2, strokeColor: '#0d1117' },
        arrows: { to: { enabled: true, scaleFactor: 0.7 } }, smooth: { type: 'curvedCW', roundness: 0.25 } });
  });

  _ontoNetwork = new vis.Network(container,
    { nodes: visNodes, edges: visEdges },
    {
      physics: { enabled: true, solver: 'forceAtlas2Based',
        forceAtlas2Based: { gravitationalConstant: -60, centralGravity: 0.005, springLength: 140, springConstant: 0.05 },
        stabilization: { iterations: 120 } },
      interaction: { hover: true, tooltipDelay: 150, navigationButtons: false, keyboard: false },
      layout: { randomSeed: 42 },
    }
  );
  _ontoNetwork.on('stabilizationIterationsDone', () => {
    _ontoNetwork.setOptions({ physics: { enabled: false } });
    _ontoNetwork.fit();
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Redundancies
// ─────────────────────────────────────────────────────────────────────────────
async function loadRedundancies() {
  try {
    const rows = await apiFetch('/metadata/redundancies');
    const badge = document.getElementById('badge-redundancies');
    if (rows.length) { badge.textContent = rows.length; badge.style.display = ''; }
    else badge.style.display = 'none';

    const tbody = document.getElementById('redundancy-body');
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;padding:40px;color:var(--text-2)">No redundancies detected (Jaccard &lt; 0.9 for all pairs)</td></tr>`;
      return;
    }
    tbody.innerHTML = rows.map(r => {
      const pct = (r.overlap_pct * 100).toFixed(1);
      const shared = (r.shared_columns || []).slice(0, 6).map(c => `<span class="top-val-chip">${_esc(c)}</span>`).join(' ');
      const more = r.shared_columns?.length > 6 ? `<span class="text-dim" style="font-size:10px">+${r.shared_columns.length - 6} more</span>` : '';
      return `<tr>
        <td class="mono">${_esc(r.a_schema ? r.a_schema+'.'+r.a_table : r.a_table)}<div class="dim" style="font-size:10px">${_esc(r.a_source_name || '')}</div></td>
        <td class="mono">${_esc(r.b_schema ? r.b_schema+'.'+r.b_table : r.b_table)}<div class="dim" style="font-size:10px">${_esc(r.b_source_name || '')}</div></td>
        <td><span class="badge badge-${pct >= 95 ? 'red' : 'amber'}">${pct}%</span></td>
        <td><div style="display:flex;flex-wrap:wrap;gap:3px">${shared}${more}</div></td>
        <td class="dim">${_fmtRelTime(r.detected_at)}</td>
      </tr>`;
    }).join('');
  } catch (e) { toast('Failed to load redundancies: ' + e.message, 'error'); }
}

// ─────────────────────────────────────────────────────────────────────────────
// SQL Console  —  uses /sources/{id}/execute-sql (direct connector, no LLM)
// ─────────────────────────────────────────────────────────────────────────────
function onViewSQL() {
  const sel = document.getElementById('sql-source-select');
  const global = document.getElementById('global-source-select').value;
  if (global && sel.value !== global) sel.value = global;
}

async function runSQL() {
  const sourceId = document.getElementById('sql-source-select').value;
  const sql = document.getElementById('sql-input').value.trim();
  if (!sourceId) { toast('Select a source first', 'warn'); return; }
  if (!sql) { toast('Enter a SQL query', 'warn'); return; }

  const status = document.getElementById('sql-status');
  const results = document.getElementById('sql-results');
  status.innerHTML = `<span class="spinner"></span>&nbsp;Running…`;
  results.innerHTML = '';

  try {
    const data = await apiFetch(`/sources/${sourceId}/execute-sql`, {
      method: 'POST',
      body: JSON.stringify({ sql, limit: 500 }),
    });
    const rows = data.rows || [];
    const cols = data.columns || (rows.length ? Object.keys(rows[0]) : []);
    const elapsed = data.elapsed_ms ?? '—';
    status.innerHTML = `<span class="text-green">${_fmtNum(rows.length)} rows · ${elapsed}ms</span>`;
    if (!rows.length) {
      results.innerHTML = '<div class="empty-state" style="height:120px">Query returned 0 rows</div>';
      return;
    }
    results.innerHTML = `<table class="data-table">
      <thead><tr>${cols.map(c => `<th>${_esc(c)}</th>`).join('')}</tr></thead>
      <tbody>${rows.slice(0, 500).map(row =>
        `<tr>${cols.map(c => `<td class="mono">${_esc(row[c] ?? '')}</td>`).join('')}</tr>`
      ).join('')}</tbody>
    </table>`;
  } catch (e) {
    status.innerHTML = `<span class="text-red">Error: ${_esc(e.message)}</span>`;
    results.innerHTML = `<div style="padding:20px; font-family:var(--font-mono); font-size:12px; color:var(--red); background:var(--bg-0)">${_esc(e.message)}</div>`;
  }
}

function clearSQL() {
  document.getElementById('sql-input').value = '';
  document.getElementById('sql-results').innerHTML = '<div class="empty-state"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="width:36px;height:36px;opacity:0.3"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>Run a query to see results</div>';
  document.getElementById('sql-status').innerHTML = '';
}

// Ctrl+Enter / Cmd+Enter to run SQL
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('sql-input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); runSQL(); }
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Change Log (CDC)
// ─────────────────────────────────────────────────────────────────────────────
async function loadCDC() {
  try {
    const rows = await apiFetch('/metadata/changes');
    const tbody = document.getElementById('cdc-body');
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;padding:40px;color:var(--text-2)">No schema changes recorded yet</td></tr>`;
      return;
    }
    const changeColor = { added:'badge-green', deleted:'badge-red', type_changed:'badge-amber', restored:'badge-cyan', updated:'badge-blue' };
    tbody.innerHTML = rows.map(r => {
      const cc = changeColor[r.change_type] || 'badge-gray';
      let detail = '';
      if (r.changed_fields?.data_type) {
        const { old: o, new: n } = r.changed_fields.data_type;
        detail = `<span class="mono" style="font-size:10px"><span style="color:var(--red)">${_esc(o)}</span> → <span style="color:var(--green)">${_esc(n)}</span></span>`;
      }
      const srcId = r.source_id || r.source || '';
      return `<tr>
        <td class="dim">${_fmtTime(r.detected_at || r.changed_at)}</td>
        <td class="mono dim">${_esc(srcId.slice(0, 8) || '—')}</td>
        <td class="mono" style="font-size:11px">${_esc(r.entity_label || r.table_name || r.entity_id?.slice(0,16) || '—')}</td>
        <td><span class="mono dim" style="font-size:10px">${_esc(r.entity_type || r.change_object || '')}</span></td>
        <td><span class="badge ${cc}">${_esc(r.change_type)}</span></td>
        <td>${detail}</td>
      </tr>`;
    }).join('');
  } catch (e) { toast('Failed to load changes: ' + e.message, 'error'); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Catalog Excel export
// ─────────────────────────────────────────────────────────────────────────────
async function exportCatalogExcel() {
  if (!_catalogAttrs.length) { toast('Load a table first', 'warn'); return; }

  const tblName = document.getElementById('catalog-tbl-name')?.textContent || 'Catalog';
  const columns = ['Column', 'Type', 'Statistical Type', 'Semantic Role',
                   'Distinct', 'Nulls', 'Sample Values', 'PII', 'Flags'];
  const rows = _catalogAttrs.map(a => {
    const flags = [];
    if (a.is_primary_key) flags.push('PK');
    if (a.is_foreign_key) flags.push('FK');
    if (a.nullable === false) flags.push('NOT NULL');
    if (a.is_golden_record) flags.push('Golden');
    if (a.deleted_from_source) flags.push('Deleted');
    const piiLabel = a.pii_flag === 'PII'
      ? (a.pii_type ? `PII · ${a.pii_type.replace(/_/g,' ')}` : 'PII')
      : (a.pii_flag || '');
    return [
      a.column_name || '',
      a.data_type || '',
      a.statistical_type || '',
      a.semantic_role ? a.semantic_role.replace(/_/g,' ') : '',
      a.unique_count != null ? a.unique_count : '',
      a.null_count != null ? a.null_count : '',
      (a.top_values || []).slice(0, 5).join(', '),
      piiLabel,
      flags.join(', '),
    ];
  });

  try {
    const resp = await fetch(`${API}/export-excel`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title:   `${tblName} — Data Catalog`,
        results: [{ description: tblName, columns, rows }],
      }),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const blob = await resp.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `catalog_${tblName.replace(/[^a-z0-9]/gi, '_')}_${new Date().toISOString().slice(0,10)}.xlsx`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (e) { toast('Excel export failed: ' + e.message, 'error'); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Catalog panel resizer
// ─────────────────────────────────────────────────────────────────────────────
(function() {
  let dragging = false, startX = 0, startW = 0;
  const resizer = document.getElementById('catalog-resizer');
  const leftPanel = document.getElementById('catalog-left');
  if (!resizer) return;
  resizer.addEventListener('mousedown', e => {
    dragging = true; startX = e.clientX; startW = leftPanel.offsetWidth;
    document.body.style.cursor = 'col-resize'; document.body.style.userSelect = 'none';
  });
  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    const w = Math.max(160, Math.min(500, startW + (e.clientX - startX)));
    leftPanel.style.width = w + 'px'; leftPanel.style.minWidth = w + 'px';
  });
  document.addEventListener('mouseup', () => { dragging = false; document.body.style.cursor = ''; document.body.style.userSelect = ''; });
})();

// ─────────────────────────────────────────────────────────────────────────────
// Initialise
// ─────────────────────────────────────────────────────────────────────────────
async function init() {
  await loadSources();
  // Auto-select first ready source
  const ready = _sources.find(s => s.status === 'ready');
  if (ready) {
    document.getElementById('global-source-select').value = ready.id;
    selectSource(ready.id);
  }
  // Poll for source status updates every 10s
  setInterval(loadSources, 10000);
}

document.addEventListener('DOMContentLoaded', init);

// =============================================================================
// DOCUMENTS TAB — Unstructured Data Intelligence
// =============================================================================

const _UNSTRUCTURED = '/unstructured';

let _docSources       = [];
let _docSelectedSrcId = null;
let _docSearchTimer   = null;

// ── Source list ──────────────────────────────────────────────────────────────

async function loadDocSources() {
  try {
    const sources = await apiFetch(`${_UNSTRUCTURED}/sources`);
    _docSources = sources || [];
    _renderDocSourceList();
  } catch {
    _renderDocSourceListError();
  }
}

function _renderDocSourceList() {
  const el = document.getElementById('doc-source-list');
  if (!el) return;
  if (!_docSources.length) {
    el.innerHTML = '<div style="padding:20px;text-align:center;color:var(--text-2);font-size:12px;">No sources yet.<br>Click <b>Add Source</b>.</div>';
    return;
  }
  el.innerHTML = _docSources.map(s => {
    const sel = s.source_id === _docSelectedSrcId ? 'selected' : '';
    const lastIdx = s.last_indexed_at
      ? new Date(s.last_indexed_at).toLocaleDateString()
      : 'Never indexed';
    return `<div class="table-list-item ${sel}" onclick="selectDocSource('${s.source_id}')">
      <div style="font-weight:600;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${_esc(s.name)}</div>
      <div style="font-size:10px;color:var(--text-2);margin-top:1px">${_esc(s.source_type)} · ${_esc(lastIdx)}</div>
    </div>`;
  }).join('');
}

function _renderDocSourceListError() {
  const el = document.getElementById('doc-source-list');
  if (el) el.innerHTML = '<div style="padding:16px;color:var(--accent);font-size:12px;">Could not reach unstructured service</div>';
}

async function selectDocSource(sourceId) {
  _docSelectedSrcId = sourceId;
  _renderDocSourceList();
  await loadDocAssets(sourceId);
}

// ── Asset list ───────────────────────────────────────────────────────────────

async function loadDocAssets(sourceId) {
  const tbody = document.getElementById('doc-asset-body');
  const badge = document.getElementById('doc-status-badge');
  if (tbody) tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;padding:30px;color:var(--text-2);">Loading…</td></tr>';
  closeDocDetail();

  try {
    const assets = await apiFetch(`${_UNSTRUCTURED}/assets?source_id=${sourceId}&limit=200`);
    _renderDocAssets(assets || []);
    if (badge) badge.textContent = `${(assets || []).length} documents`;
  } catch (e) {
    if (tbody) tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;padding:30px;color:var(--accent);">${_esc(e.message)}</td></tr>`;
  }
}

function _renderDocAssets(assets) {
  const tbody = document.getElementById('doc-asset-body');
  if (!tbody) return;
  if (!assets.length) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;padding:40px;color:var(--text-2);">No documents indexed yet. Click <b>Add Source</b> to register a folder and start indexing.</td></tr>';
    return;
  }
  tbody.innerHTML = assets.map(a => {
    const topics = (a.topics || []).slice(0, 3).map(t => `<span class="top-val-chip">${_esc(t)}</span>`).join('');
    const piiDot = a.pii_risk ? '<span class="badge badge-pii" style="background:#ff4d4d;color:#fff;font-size:9px;">PII</span>' : '';
    const sensColor = { public:'badge-green', internal:'badge-gray', confidential:'badge-amber', restricted:'badge-red' };
    const sensBadge = `<span class="badge ${sensColor[a.sensitivity] || 'badge-gray'}" style="font-size:9px;">${_esc(a.sensitivity || 'internal')}</span>`;
    const dt = a.indexed_at ? new Date(a.indexed_at).toLocaleDateString() : '—';
    return `<tr style="cursor:pointer" onclick="openDocDetail('${a.asset_id}')">
      <td>
        <div style="font-weight:600;font-size:12px">${_esc(a.title || a.file_name)}</div>
        <div style="font-size:10px;color:var(--text-2);margin-top:1px">${_esc(a.file_name)} · ${a.size_bytes ? _fmtBytes(a.size_bytes) : '—'}</div>
      </td>
      <td><span class="badge badge-gray" style="font-size:9px;text-transform:uppercase">${_esc(a.file_type || '—')}</span></td>
      <td style="font-size:11px">${_esc(a.domain || '—')}</td>
      <td><div style="display:flex;flex-wrap:wrap;gap:2px">${topics || '<span class="text-dim">—</span>'}</div></td>
      <td>${sensBadge}</td>
      <td>${piiDot || '<span class="text-dim" style="font-size:10px">—</span>'}</td>
      <td style="font-size:11px;color:var(--text-2)">${dt}</td>
    </tr>`;
  }).join('');
}

function _fmtBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  return (b / 1048576).toFixed(1) + ' MB';
}

// ── Asset detail drawer ───────────────────────────────────────────────────────

async function openDocDetail(assetId) {
  const drawer = document.getElementById('doc-detail-drawer');
  if (!drawer) return;
  drawer.style.display = 'block';
  document.getElementById('doc-detail-title').textContent = 'Loading…';
  document.getElementById('doc-detail-meta').textContent  = '';
  document.getElementById('doc-detail-summary').textContent = '';
  document.getElementById('doc-detail-topics').innerHTML  = '';
  document.getElementById('doc-detail-entities').innerHTML = '';
  document.getElementById('doc-detail-links').innerHTML   = 'Loading…';

  try {
    const [asset, linkData] = await Promise.all([
      apiFetch(`${_UNSTRUCTURED}/assets/${assetId}`),
      apiFetch(`${_UNSTRUCTURED}/assets/${assetId}/links`).catch(() => ({ links: [] })),
    ]);

    document.getElementById('doc-detail-title').textContent =
      asset.title || asset.file_name;
    document.getElementById('doc-detail-meta').textContent =
      [asset.doc_type, asset.domain, asset.language,
       asset.page_count ? `${asset.page_count} pages` : null,
       asset.indexed_at ? new Date(asset.indexed_at).toLocaleDateString() : null]
      .filter(Boolean).join(' · ');
    document.getElementById('doc-detail-summary').textContent =
      asset.summary || 'No summary available.';

    // Topics
    const topicsEl = document.getElementById('doc-detail-topics');
    topicsEl.innerHTML = (asset.topics || []).length
      ? asset.topics.map(t => `<span class="top-val-chip">${_esc(t)}</span>`).join('')
      : '<span class="text-dim">—</span>';

    // Named entities
    const ents = asset.named_entities || {};
    const entLines = [];
    if ((ents.kpis || []).length)           entLines.push(`<b>KPIs:</b> ${_esc(ents.kpis.join(', '))}`);
    if ((ents.organizations || []).length)  entLines.push(`<b>Orgs:</b> ${_esc(ents.organizations.join(', '))}`);
    if ((ents.products || []).length)       entLines.push(`<b>Products:</b> ${_esc(ents.products.join(', '))}`);
    if ((ents.geographies || []).length)    entLines.push(`<b>Geo:</b> ${_esc(ents.geographies.join(', '))}`);
    if ((ents.people || []).length)         entLines.push(`<b>People:</b> ${_esc(ents.people.join(', '))}`);
    document.getElementById('doc-detail-entities').innerHTML =
      entLines.join('<br>') || '<span class="text-dim">—</span>';

    // Cross-modal links
    const links = (linkData.links || []).filter(l => l.confidence >= 0.6);
    const linksEl = document.getElementById('doc-detail-links');
    if (!links.length) {
      linksEl.innerHTML = '<span class="text-dim">No links detected</span>';
    } else {
      linksEl.innerHTML = links.slice(0, 8).map(l => {
        const conf = Math.round(l.confidence * 100);
        const color = l.rel_type === 'DESCRIBES_KPI' ? 'badge-blue'
                    : l.rel_type === 'REFERENCES_TABLE' ? 'badge-amber'
                    : 'badge-gray';
        return `<div style="margin-bottom:4px">
          <span class="badge ${color}" style="font-size:9px;">${_esc(l.rel_type.replace(/_/g,' '))}</span>
          <span style="font-size:11px;color:var(--text-2);margin-left:4px">${_esc(l.basis.split('→')[1] || l.to_nanite_id || l.to_asset_id || '')} <span style="opacity:.6">${conf}%</span></span>
        </div>`;
      }).join('');
    }
  } catch (e) {
    document.getElementById('doc-detail-title').textContent = 'Error loading document';
    document.getElementById('doc-detail-summary').textContent = e.message;
  }
}

function closeDocDetail() {
  const drawer = document.getElementById('doc-detail-drawer');
  if (drawer) drawer.style.display = 'none';
}

// ── Search ────────────────────────────────────────────────────────────────────

function docSearchDebounced(value) {
  clearTimeout(_docSearchTimer);
  if (!value || value.length < 2) {
    if (_docSelectedSrcId) loadDocAssets(_docSelectedSrcId);
    return;
  }
  _docSearchTimer = setTimeout(() => docSearch(value), 400);
}

async function docSearch(q) {
  const badge = document.getElementById('doc-status-badge');
  try {
    const data = await apiFetch(`${_UNSTRUCTURED}/search?q=${encodeURIComponent(q)}`);
    _renderDocAssets(data.results || []);
    if (badge) badge.textContent = `${(data.results || []).length} results for "${q}"`;
  } catch (e) {
    toast('Document search failed: ' + e.message, 'error');
  }
}

// ── Add Document Source Modal ─────────────────────────────────────────────────

function openAddDocSourceModal() {
  // Populate nanite source select
  const sel = document.getElementById('doc-src-nanite-id');
  if (sel) {
    sel.innerHTML = '<option value="">— none —</option>' +
      _sources.map(s => `<option value="${s.id}">${_esc(s.name || s.id)}</option>`).join('');
  }
  const overlay = document.getElementById('doc-source-modal-overlay');
  if (overlay) { overlay.style.display = 'flex'; }
}

function closeDocSourceModal(e) {
  if (e && e.target !== document.getElementById('doc-source-modal-overlay')) return;
  const overlay = document.getElementById('doc-source-modal-overlay');
  if (overlay) overlay.style.display = 'none';
}

function onDocSrcTypeChange(/* type */) {
  // Future: toggle connection fields per source type
}

async function submitDocSource() {
  const name     = document.getElementById('doc-src-name')?.value.trim();
  const type     = document.getElementById('doc-src-type')?.value || 'local';
  const path     = document.getElementById('doc-src-path')?.value.trim();
  const domain   = document.getElementById('doc-src-domain')?.value.trim();
  const naniteId = document.getElementById('doc-src-nanite-id')?.value || null;

  if (!name)  { toast('Source name is required', 'warn'); return; }
  if (!path)  { toast('Folder path is required', 'warn');  return; }

  try {
    const source = await apiFetch(`${_UNSTRUCTURED}/sources`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name,
        source_type: type,
        connection:  { path },
        nanite_source_id: naniteId || null,
        domain: domain || '',
      }),
    });

    closeDocSourceModal();
    toast(`Source "${name}" registered. Starting index…`, 'info');
    await loadDocSources();

    // Kick off initial index
    await apiFetch(`${_UNSTRUCTURED}/sources/${source.source_id}/index`, { method: 'POST' });
    _docSelectedSrcId = source.source_id;
    _renderDocSourceList();
    _pollDocJob(source.source_id);
  } catch (e) {
    toast('Failed to register source: ' + e.message, 'error');
  }
}

async function _pollDocJob(sourceId) {
  const badge = document.getElementById('doc-status-badge');
  const start = Date.now();
  while (Date.now() - start < 300000) {  // max 5 min poll
    await new Promise(r => setTimeout(r, 3000));
    try {
      const jobs = await apiFetch(`${_UNSTRUCTURED}/sources/${sourceId}/jobs`);
      const latest = (jobs || [])[0];
      if (!latest) break;
      if (badge) badge.textContent = `Indexing… ${latest.processed}/${latest.total_files} files`;
      if (latest.status === 'done' || latest.status === 'done_with_errors') {
        const msg = `Indexed ${latest.enriched} documents`;
        if (badge) badge.textContent = msg;
        toast(msg + (latest.errors ? ` (${latest.errors} errors)` : ''), latest.errors ? 'warn' : 'info');
        await loadDocAssets(sourceId);
        break;
      }
    } catch { break; }
  }
}
