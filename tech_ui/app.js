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
        <div class="src-meta">${_esc(s.db_type)} · ${s.table_count ?? 0} tables</div>
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
  if (src.status) stats.push(`<span>Status: <b style="color:${src.status==='ready'?'var(--green)':src.status==='indexing'?'var(--accent)':'var(--red)'}">${src.status}</b></span>`);
  if (src.indexed_at) stats.push(`<span>Last indexed: <b>${_fmtTime(src.indexed_at)}</b></span>`);
  document.getElementById('detail-src-stats').innerHTML = stats.join('&nbsp;&nbsp;·&nbsp;&nbsp;');

  // Subscribe to SSE
  startSSE(sourceId);
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
      font: { color: '#8b949e', size: 10, face: 'JetBrains Mono, Consolas' },
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
        forceAtlas2Based: { gravitationalConstant: -80, centralGravity: 0.01, springLength: 120, springConstant: 0.06 },
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

    _kgNetwork.on('stabilizationIterationsDone', () => _kgNetwork.setOptions({ physics: { enabled: false } }));

  } catch (e) {
    toast('Failed to load graph: ' + e.message, 'error');
    document.getElementById('graph-empty').style.display = 'flex';
  }
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
  _ontoNetwork.on('stabilizationIterationsDone', () => _ontoNetwork.setOptions({ physics: { enabled: false } }));
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
