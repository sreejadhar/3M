'use strict';

// ── Config ────────────────────────────────────────────────────────────────────
const API = '';   // same origin

// ── Persona ───────────────────────────────────────────────────────────────────
const PERSONAS = {
  business_user: { label: 'Business User',    icon: '👤', showSQL: false, canConnect: false, isAdmin: false },
  analyst:       { label: 'Business Analyst', icon: '🔬', showSQL: true,  canConnect: true,  isAdmin: false },
  admin:         { label: 'Data Admin',       icon: '⚙️', showSQL: true,  canConnect: true,  isAdmin: true  },
};

let currentPersona = localStorage.getItem('datachat_persona') || 'business_user';

// ── State ─────────────────────────────────────────────────────────────────────
let activeSessionId   = null;
let activeEventSource = null;
let pendingFiles      = [];
let isWaitingForReply = false;
let progressMsgId     = null;
let sessions          = {};
let sources           = {};       // source_id → source dict
let activeSourceId    = null;     // selected source on landing
let wizardStep        = 1;
let wizardDbType      = null;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const sidebar          = document.getElementById('sidebar');
const sidebarToggle    = document.getElementById('sidebarToggle');
const menuBtn          = document.getElementById('menuBtn');
const newChatBtn       = document.getElementById('newChatBtn');
const sessionList      = document.getElementById('sessionList');
const sourceList       = document.getElementById('sourceList');
const personaBadge     = document.getElementById('personaBadge');
const personaIcon      = document.getElementById('personaIcon');
const personaLabel     = document.getElementById('personaLabel');
const personaSwitchBtn = document.getElementById('personaSwitchBtn');
const personaDropdown  = document.getElementById('personaDropdown');
const sidebarBottom    = document.getElementById('sidebarBottom');
const adminBtn         = document.getElementById('adminBtn');

const landing          = document.getElementById('landing');
const sourceCatalog    = document.getElementById('sourceCatalog');
const chatView         = document.getElementById('chatView');
const welcome          = document.getElementById('welcome');
const messages         = document.getElementById('messages');
const chatArea         = document.getElementById('chatArea');
const topbarSourceName = document.getElementById('topbarSourceName');

const fileInput        = document.getElementById('fileInput');
const fileChips        = document.getElementById('fileChips');
const msgInput         = document.getElementById('msgInput');
const sendBtn          = document.getElementById('sendBtn');
const pipelineStatus   = document.getElementById('pipelineStatus');
const pipelineBar      = document.getElementById('pipelineBar');
const pipelineBarInner = document.getElementById('pipelineBarInner');
const toastContainer   = document.getElementById('toastContainer');

const wizardOverlay    = document.getElementById('wizardOverlay');
const wizardClose      = document.getElementById('wizardClose');
const wizardBack       = document.getElementById('wizardBack');
const wizardNext       = document.getElementById('wizardNext');
const testConnBtn      = document.getElementById('testConnBtn');
const testConnResult   = document.getElementById('testConnResult');

const adminOverlay     = document.getElementById('adminOverlay');
const adminClose       = document.getElementById('adminClose');
const adminAddBtn      = document.getElementById('adminAddBtn');
const adminRefreshBtn  = document.getElementById('adminRefreshBtn');
const adminTableBody   = document.getElementById('adminTableBody');

// ── Utilities ─────────────────────────────────────────────────────────────────

function escHtml(str) {
  return String(str ?? '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function showToast(msg, type = 'info', duration = 4000) {
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  toastContainer.appendChild(t);
  setTimeout(() => t.remove(), duration);
}

function scrollToBottom(smooth = true) {
  chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: smooth ? 'smooth' : 'auto' });
}

function formatNumber(v) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') {
    return Number.isInteger(v) ? v.toLocaleString() : parseFloat(v.toFixed(4)).toLocaleString();
  }
  return String(v);
}

function isNumeric(v) {
  return typeof v === 'number' || (typeof v === 'string' && v !== '' && !isNaN(Number(v)));
}

function fileIcon(filename) {
  const ext = (filename || '').split('.').pop().toLowerCase();
  if (['xlsx','xls','xlsm','xlsb'].includes(ext)) return '📊';
  if (ext === 'csv') return '📄';
  return '📁';
}

function renderMarkdown(text) {
  if (!text) return '';
  if (typeof marked !== 'undefined') {
    marked.setOptions({ breaks: true, gfm: true });
    return marked.parse(text);
  }
  return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
}

function dbTypeLabel(t) {
  const MAP = { postgres:'PostgreSQL', postgresql:'PostgreSQL', sqlserver:'SQL Server',
    mssql:'SQL Server', oracle:'Oracle', mysql:'MySQL', sqlite:'SQLite', csv:'CSV/Excel', excel:'CSV/Excel' };
  return MAP[t] || t || '—';
}

// ── Persona ───────────────────────────────────────────────────────────────────

function applyPersona() {
  const p = PERSONAS[currentPersona] || PERSONAS.business_user;
  personaIcon.textContent  = p.icon;
  personaLabel.textContent = p.label;

  // Highlight active option in dropdown
  document.querySelectorAll('.persona-option').forEach(el => {
    el.classList.toggle('active', el.dataset.persona === currentPersona);
  });

  // Admin button visibility
  sidebarBottom.style.display = p.isAdmin ? '' : 'none';

  // SQL disclosure: show by default for analysts/admins
  document.documentElement.dataset.showSql = p.showSQL ? 'true' : 'false';

  // Reload source catalog with persona filter
  renderSourceCatalog();
  renderSourceSidebar();
}

function switchPersona(persona) {
  currentPersona = persona;
  localStorage.setItem('datachat_persona', persona);
  personaDropdown.style.display = 'none';
  applyPersona();
  showToast(`Switched to ${PERSONAS[persona].label}`, 'info', 2500);
}

// ── View management ───────────────────────────────────────────────────────────

function showLanding() {
  landing.style.display = 'flex';
  chatView.style.display = 'none';
  topbarSourceName.textContent = '';
  pipelineStatus.classList.remove('visible');
}

function showChatView(sourceNameOrTitle) {
  landing.style.display = 'none';
  chatView.style.display = '';   // use CSS flex
  topbarSourceName.textContent = sourceNameOrTitle || '';
}

// ── Source API calls ──────────────────────────────────────────────────────────

async function apiListSources() {
  const r = await fetch(`${API}/sources?persona=${encodeURIComponent(currentPersona)}`);
  if (!r.ok) return [];
  return r.json();
}

async function apiCreateSource(payload) {
  const r = await fetch(`${API}/sources`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function apiDeleteSource(id) {
  await fetch(`${API}/sources/${id}`, { method: 'DELETE' });
}

async function apiReindexSource(id) {
  const r = await fetch(`${API}/sources/${id}/reindex`, { method: 'POST' });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function apiTestConnection(payload) {
  const r = await fetch(`${API}/sources/test-connection`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!r.ok) return { ok: false, error: await r.text() };
  return r.json();
}

async function apiGetSource(id) {
  const r = await fetch(`${API}/sources/${id}`);
  if (!r.ok) return null;
  return r.json();
}

// ── Session API calls ─────────────────────────────────────────────────────────

async function apiCreateSession(title, sourceId) {
  const body = { title };
  if (sourceId) body.source_id = sourceId;
  const r = await fetch(`${API}/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function apiListSessions() {
  const r = await fetch(`${API}/sessions`);
  if (!r.ok) return [];
  return r.json();
}

async function apiDeleteSession(id) {
  await fetch(`${API}/sessions/${id}`, { method: 'DELETE' });
}

async function apiUploadFiles(sessionId, files) {
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  const r = await fetch(`${API}/sessions/${sessionId}/upload`, { method: 'POST', body: fd });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}

async function apiSendChat(sessionId, message) {
  const r = await fetch(`${API}/sessions/${sessionId}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}

// ── Load sources ──────────────────────────────────────────────────────────────

async function loadSources() {
  try {
    const list = await apiListSources();
    sources = {};
    list.forEach(s => sources[s.id] = s);
  } catch {
    sources = {};
  }
  renderSourceCatalog();
  renderSourceSidebar();
}

// ── Source Catalog (landing page) ─────────────────────────────────────────────

function renderSourceCatalog() {
  const p = PERSONAS[currentPersona] || PERSONAS.business_user;
  sourceCatalog.innerHTML = '';

  Object.values(sources).forEach(s => {
    const card = document.createElement('div');
    card.className = `source-card ${s.status}`;
    card.dataset.sourceId = s.id;

    const statusLabel = s.status === 'ready'    ? `✓ Ready — ${s.table_count} table${s.table_count !== 1 ? 's' : ''}`
                      : s.status === 'indexing'  ? '⟳ Indexing…'
                      : s.status === 'error'     ? '⚠ Error'
                      : '○ Not indexed';

    card.innerHTML = `
      <div class="source-card-top">
        <div class="source-card-icon">${escHtml(s.icon || '📊')}</div>
        <div class="source-card-info">
          <div class="source-card-name">${escHtml(s.name)}</div>
          <div class="source-card-type">${escHtml(dbTypeLabel(s.db_type))}</div>
        </div>
      </div>
      ${s.description ? `<div class="source-card-desc">${escHtml(s.description)}</div>` : ''}
      <div class="source-card-footer">
        <span class="domain-badge">${escHtml(s.domain)}</span>
        <span class="source-card-status ${s.status}">${escHtml(statusLabel)}</span>
      </div>`;

    if (s.status === 'ready') {
      card.addEventListener('click', () => openSourceSession(s.id));
    } else if (s.status === 'indexing') {
      card.title = 'Source is being indexed. Please wait.';
    } else if (s.status === 'error') {
      card.title = s.error_message || 'Indexing failed';
    }
    sourceCatalog.appendChild(card);
  });

  // "Upload your own data" card
  const uploadCard = document.createElement('div');
  uploadCard.className = 'source-card add-card';
  uploadCard.innerHTML = `
    <div class="add-card-icon">📁</div>
    <div class="add-card-label">Upload your own data</div>
    <div class="add-card-sub">CSV or Excel files</div>`;
  uploadCard.addEventListener('click', startAdHocUploadSession);
  sourceCatalog.appendChild(uploadCard);

  // "Connect a database" card (analysts + admins)
  if (p.canConnect) {
    const connectCard = document.createElement('div');
    connectCard.className = 'source-card add-card';
    connectCard.innerHTML = `
      <div class="add-card-icon">🔌</div>
      <div class="add-card-label">Connect a database</div>
      <div class="add-card-sub">PostgreSQL, SQL Server, Oracle, MySQL…</div>`;
    connectCard.addEventListener('click', openWizard);
    sourceCatalog.appendChild(connectCard);
  }
}

// ── Source Sidebar list ───────────────────────────────────────────────────────

function renderSourceSidebar() {
  sourceList.innerHTML = '';
  const list = Object.values(sources);
  if (list.length === 0) {
    sourceList.innerHTML = '<div class="source-list-empty">No sources configured</div>';
    return;
  }
  list.forEach(s => {
    const btn = document.createElement('button');
    btn.className = 'source-sidebar-item' + (s.id === activeSourceId ? ' active' : '');
    btn.innerHTML = `
      <span class="source-sidebar-icon">${escHtml(s.icon || '📊')}</span>
      <span class="source-sidebar-name">${escHtml(s.name)}</span>
      <span class="source-status-dot ${s.status}"></span>`;
    btn.addEventListener('click', () => {
      if (s.status === 'ready') openSourceSession(s.id);
      else showToast(`Source "${s.name}" is ${s.status}`, 'info', 3000);
    });
    sourceList.appendChild(btn);
  });
}

// ── Open a session from a registered source ───────────────────────────────────

async function openSourceSession(sourceId) {
  activeSourceId = sourceId;
  const src = sources[sourceId];
  try {
    const { session_id, title, stage } = await apiCreateSession(src.name, sourceId);
    sessions[session_id] = { session_id, title, stage, created_at: Date.now() / 1000, source_id: sourceId };
    activeSessionId = session_id;
    subscribeSSE(session_id);
    clearChatUI();
    renderSessionList();
    renderSourceSidebar();
    showChatView(src.name);

    if (stage === 'ready') {
      msgInput.disabled = false;
      msgInput.placeholder = 'Ask anything about your data…';
      updateSendState();
      welcome.style.display = '';
    }
  } catch (err) {
    showToast(err.message || 'Failed to open session', 'error');
  }
}

// ── Ad-hoc file upload session ────────────────────────────────────────────────

// Trigger file picker synchronously inside the click handler — no await before this
// or browsers block the programmatic open. Session is created lazily in handleFileUpload.
function startAdHocUploadSession() {
  activeSourceId  = null;
  activeSessionId = null;
  fileInput.click();
}

// ── SSE subscription ──────────────────────────────────────────────────────────

function subscribeSSE(sessionId) {
  if (activeEventSource) { activeEventSource.close(); activeEventSource = null; }
  const es = new EventSource(`${API}/sessions/${sessionId}/events`);
  activeEventSource = es;
  es.onmessage = (ev) => {
    let event;
    try { event = JSON.parse(ev.data); } catch { return; }
    handleSSEEvent(event);
  };
}

function handleSSEEvent(ev) {
  switch (ev.type) {
    case 'heartbeat': return;

    case 'progress':
      if (ev.background) updateTopbarStatus(ev.message, 'running');
      else updatePipelineProgress(ev.stage, ev.message, ev.pct || 0);
      break;

    case 'ready':
      finishPipeline(ev.message, ev.tables || []);
      break;

    case 'ontology_ready':
      showToast('Ontology built', 'success', 3000);
      updateTopbarStatus('Ontology ready', 'done');
      break;

    case 'kg_ready':
      showToast(ev.message || 'Knowledge graph ready', 'success', 3000);
      updateTopbarStatus('KG ready', 'done');
      break;

    case 'info':
      if (ev.background) updateTopbarStatus(ev.message, 'done');
      break;

    case 'error':
      handlePipelineError(ev.message);
      break;

    case 'thinking':
      showTypingIndicator(ev.msg_id);
      break;

    case 'chat_response':
      hideTypingIndicator(ev.msg_id);
      renderAIMessage(ev);
      isWaitingForReply = false;
      updateSendState();
      break;

    case 'chat_error':
      hideTypingIndicator(ev.msg_id);
      renderAIError(ev.msg_id, ev.message);
      isWaitingForReply = false;
      updateSendState();
      break;
  }
}

// ── Pipeline progress UI ──────────────────────────────────────────────────────

const STAGE_LABELS = {
  uploading:  'Uploading files',
  extracting: 'Extracting metadata',
  ontology:   'Building ontology',
  kg:         'Building knowledge graph',
};

function updatePipelineProgress(stage, message, pct) {
  if (!progressMsgId) {
    progressMsgId = 'progress-' + Date.now();
    const div = document.createElement('div');
    div.className = 'system-msg';
    div.id = progressMsgId;
    div.innerHTML = buildProgressCard(stage, message, pct);
    messages.appendChild(div);
    welcome.style.display = 'none';
  } else {
    const el = document.getElementById(progressMsgId);
    if (el) el.innerHTML = buildProgressCard(stage, message, pct);
  }
  pipelineBar.style.display = 'block';
  pipelineBarInner.style.width = pct + '%';
  updateTopbarStatus(message, 'running');
  scrollToBottom();
}

function buildProgressCard(stage, message, pct) {
  const stageLabel = STAGE_LABELS[stage] || stage;
  return `
    <div class="progress-card">
      <div class="progress-card-header">
        <div class="progress-card-icon running">⚡</div>
        <div>
          <div class="progress-card-title">${escHtml(stageLabel)}</div>
          <div class="progress-card-sub">${escHtml(message)}</div>
        </div>
      </div>
      <div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:${pct}%"></div></div>
    </div>`;
}

function finishPipeline(message, tables) {
  if (progressMsgId) {
    const el = document.getElementById(progressMsgId);
    if (el) el.innerHTML = buildReadyCard(message, tables);
    progressMsgId = null;
  } else {
    const div = document.createElement('div');
    div.className = 'system-msg';
    div.innerHTML = buildReadyCard(message, tables);
    messages.appendChild(div);
    welcome.style.display = 'none';
  }
  pipelineBar.style.display = 'none';
  pipelineBarInner.style.width = '0%';
  updateTopbarStatus('Ready', 'done');
  msgInput.disabled = false;
  msgInput.placeholder = 'Ask anything about your data…';
  updateSendState();
  document.querySelectorAll('.suggestion-chip').forEach(c => c.disabled = false);
  scrollToBottom();
}

function buildReadyCard(message, tables) {
  const tableList = tables.length
    ? `<div class="ready-card-sub">Tables: ${tables.map(t => `<strong>${escHtml(t)}</strong>`).join(', ')}</div>`
    : '';
  return `
    <div class="ready-card">
      <div class="ready-card-icon">✅</div>
      <div class="ready-card-text">
        <div class="ready-card-title">${escHtml(message)}</div>
        ${tableList}
      </div>
    </div>`;
}

function handlePipelineError(message) {
  if (progressMsgId) {
    const el = document.getElementById(progressMsgId);
    if (el) el.innerHTML = `<div class="error-card"><div class="error-card-icon">⚠️</div><div class="error-card-title">${escHtml(message)}</div></div>`;
    progressMsgId = null;
  }
  pipelineBar.style.display = 'none';
  updateTopbarStatus('Error', 'error');
  showToast(message, 'error', 6000);
}

function updateTopbarStatus(message, state) {
  pipelineStatus.classList.add('visible');
  const dotClass = state === 'done' ? 'done' : state === 'error' ? 'error' : '';
  pipelineStatus.innerHTML = `<span class="dot ${dotClass}"></span><span>${escHtml(message)}</span>`;
  if (state === 'done') {
    setTimeout(() => {
      if (pipelineStatus.querySelector('.dot.done')) pipelineStatus.classList.remove('visible');
    }, 4000);
  }
}

// ── Chat rendering ────────────────────────────────────────────────────────────

function renderUserMessage(message, msgId) {
  const row = document.createElement('div');
  row.className = 'msg-row user';
  row.id = 'msg-' + msgId;
  row.innerHTML = `<div class="msg-avatar">U</div><div class="msg-bubble">${escHtml(message)}</div>`;
  messages.appendChild(row);
  welcome.style.display = 'none';
  scrollToBottom();
}

function showTypingIndicator(msgId) {
  if (document.getElementById('typing-' + msgId)) return;
  const row = document.createElement('div');
  row.className = 'msg-row assistant';
  row.id = 'typing-' + msgId;
  row.innerHTML = `
    <div class="msg-avatar">⬡</div>
    <div class="msg-bubble"><div class="typing-indicator"><span></span><span></span><span></span></div></div>`;
  messages.appendChild(row);
  scrollToBottom();
}

function hideTypingIndicator(msgId) {
  const el = document.getElementById('typing-' + msgId);
  if (el) el.remove();
}

function renderAIMessage(ev) {
  const p          = PERSONAS[currentPersona] || PERSONAS.business_user;
  const row        = document.createElement('div');
  row.className    = 'msg-row assistant';
  row.id           = 'ai-msg-' + ev.msg_id;

  const mdHtml     = renderMarkdown(ev.content || '');
  const resultsHtml = buildResultBlocks(ev.results || []);
  const sqlHtml    = buildSQLDisclosure(ev.sql || [], ev.msg_id, p.showSQL);
  const errHtml    = buildErrorNotes(ev.errors || []);
  const cacheNote  = ev.cache_hit
    ? '<div style="font-size:11px;color:var(--clr-text-mute);margin-top:8px">⚡ From cache</div>' : '';

  row.innerHTML = `
    <div class="msg-avatar">⬡</div>
    <div class="msg-bubble">
      <div class="md-content">${mdHtml}</div>
      ${resultsHtml}${errHtml}${sqlHtml}${cacheNote}
    </div>`;
  messages.appendChild(row);
  scrollToBottom();
}

function renderAIError(msgId, message) {
  const row = document.createElement('div');
  row.className = 'msg-row assistant';
  row.id = 'ai-err-' + msgId;
  row.innerHTML = `<div class="msg-avatar">⬡</div><div class="msg-bubble"><div class="msg-error-note">⚠️ ${escHtml(message)}</div></div>`;
  messages.appendChild(row);
  scrollToBottom();
}

function buildResultBlocks(results) {
  if (!results || !results.length) return '';
  return results.map((r, i) => {
    const rows = r.rows || [];
    if (!rows.length) return '';
    const cols   = Object.keys(rows[0]);
    const header = cols.map(c => `<th>${escHtml(c)}</th>`).join('');
    const body   = rows.map(row =>
      `<tr>${cols.map(c => {
        const v = row[c], num = isNumeric(v);
        return `<td class="${num ? 'num-cell' : ''}" title="${escHtml(String(v ?? ''))}">${escHtml(formatNumber(v))}</td>`;
      }).join('')}</tr>`
    ).join('');
    return `
      <div class="result-block">
        <div class="result-block-header">📋 ${escHtml(r.query_label || `Query ${i+1}`)}
          <span style="margin-left:auto;font-weight:normal;color:var(--clr-text-mute)">${rows.length} row${rows.length !== 1 ? 's' : ''}</span>
        </div>
        <div class="result-table-wrap">
          <table class="result-table"><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>
        </div>
      </div>`;
  }).join('');
}

function buildSQLDisclosure(sqlQueries, msgId, showByDefault) {
  if (!sqlQueries || !sqlQueries.length) return '';
  const uid    = 'sql-' + msgId;
  const open   = showByDefault ? ' visible' : '';
  const iconCh = showByDefault ? '▼' : '▶';
  const blocks = sqlQueries.map((q, i) => {
    const label = q.query_label || `Query ${i+1}`;
    return `<div style="margin-bottom:8px"><div style="font-size:11px;color:var(--clr-text-mute);padding:6px 0 4px">${escHtml(label)}</div><pre>${escHtml(q.sql || '')}</pre></div>`;
  }).join('');
  return `
    <div class="sql-disclosure">
      <button class="sql-toggle" onclick="toggleSQL('${uid}',this)">
        <span class="sql-toggle-icon${showByDefault ? ' open' : ''}" id="icon-${uid}">${iconCh}</span>
        ${sqlQueries.length} SQL quer${sqlQueries.length === 1 ? 'y' : 'ies'}
      </button>
      <div class="sql-block${open}" id="${uid}">${blocks}</div>
    </div>`;
}

function buildErrorNotes(errors) {
  if (!errors || !errors.length) return '';
  return errors.map(e => `<div class="msg-error-note">⚠️ ${escHtml(String(e))}</div>`).join('');
}

window.toggleSQL = function(id) {
  const block = document.getElementById(id);
  const icon  = document.getElementById('icon-' + id);
  if (!block) return;
  const open = block.classList.toggle('visible');
  if (icon) { icon.textContent = open ? '▼' : '▶'; icon.classList.toggle('open', open); }
};

// ── Send logic ────────────────────────────────────────────────────────────────

function updateSendState() {
  const hasText = msgInput.value.trim().length > 0;
  sendBtn.disabled = !(hasText && !isWaitingForReply && !msgInput.disabled);
}

async function sendMessage() {
  const text = msgInput.value.trim();
  if (!text || isWaitingForReply || !activeSessionId) return;
  isWaitingForReply = true;
  const tmp = msgInput.value;
  msgInput.value = '';
  updateSendState();
  msgInput.style.height = 'auto';
  renderUserMessage(text, 'u-' + Date.now());
  try {
    await apiSendChat(activeSessionId, text);
  } catch (err) {
    isWaitingForReply = false;
    updateSendState();
    msgInput.value = tmp;
    showToast(err.message || 'Failed to send', 'error');
  }
}

// ── Session management ────────────────────────────────────────────────────────

async function loadSessions() {
  try {
    const list = await apiListSessions();
    sessions = {};
    list.forEach(s => sessions[s.session_id] = s);
    renderSessionList();
  } catch { /* ignore */ }
}

function renderSessionList() {
  sessionList.innerHTML = '';
  const sorted = Object.values(sessions).sort((a, b) => b.created_at - a.created_at);
  if (sorted.length === 0) {
    sessionList.innerHTML = '<div style="font-size:13px;color:var(--clr-text-mute);padding:8px 14px">No conversations yet</div>';
    return;
  }
  sorted.forEach(s => {
    const btn = document.createElement('button');
    btn.className = 'session-item' + (s.session_id === activeSessionId ? ' active' : '');
    btn.innerHTML = `
      <span class="session-item-icon">${s.stage === 'ready' ? '💬' : s.stage === 'error' ? '⚠️' : '📁'}</span>
      <span class="session-item-title">${escHtml(s.title || 'Untitled')}</span>
      <button class="session-item-del" data-id="${s.session_id}" title="Delete">✕</button>`;
    btn.addEventListener('click', (e) => {
      if (e.target.closest('.session-item-del')) return;
      resumeSession(s.session_id);
    });
    btn.querySelector('.session-item-del').addEventListener('click', async (e) => {
      e.stopPropagation();
      if (!confirm('Delete this conversation?')) return;
      await apiDeleteSession(s.session_id);
      delete sessions[s.session_id];
      if (activeSessionId === s.session_id) {
        activeSessionId = null;
        activeEventSource && activeEventSource.close();
        activeEventSource = null;
        showLanding();
      }
      renderSessionList();
    });
    sessionList.appendChild(btn);
  });
}

async function resumeSession(sessionId) {
  activeSessionId = sessionId;
  subscribeSSE(sessionId);
  clearChatUI();
  renderSessionList();
  const s = sessions[sessionId];
  const srcName = s?.source_id ? (sources[s.source_id]?.name || s.title) : s?.title;
  showChatView(srcName || '');

  try {
    const r    = await fetch(`${API}/sessions/${sessionId}/messages`);
    const data = await r.json();

    data.messages.forEach(msg => {
      if (msg.role === 'user') {
        renderUserMessage(msg.content, msg.id);
      } else if (msg.role === 'assistant' && !msg.error) {
        renderAIMessage({ msg_id: msg.id, content: msg.content, results: msg.results || [], sql: msg.sql || [], errors: msg.errors || [] });
      } else if (msg.error) {
        renderAIError(msg.id, msg.content);
      }
    });

    if (data.stage === 'ready') {
      msgInput.disabled = false;
      msgInput.placeholder = 'Ask anything about your data…';
      updateTopbarStatus('Ready', 'done');
      updateSendState();
    }
    if (data.messages.length > 0) welcome.style.display = 'none';
    scrollToBottom(false);
  } catch { /* ignore */ }
}

function clearChatUI() {
  messages.innerHTML   = '';
  progressMsgId        = null;
  isWaitingForReply    = false;
  msgInput.disabled    = true;
  msgInput.value       = '';
  msgInput.placeholder = 'Upload files to get started…';
  fileChips.innerHTML  = '';
  pendingFiles         = [];
  pipelineBar.style.display = 'none';
  pipelineStatus.classList.remove('visible');
  welcome.style.display = 'block';
  updateSendState();
}

async function createNewSession() {
  activeSourceId  = null;
  activeSessionId = null;
  pendingFiles    = [];
  fileChips.innerHTML = '';
  fileInput.value = '';
  await loadSources();
  if (Object.keys(sources).length > 0) {
    showLanding();
  } else {
    clearChatUI();
    showChatView('');
  }
}

// ── File handling ─────────────────────────────────────────────────────────────

function renderFileChips(files) {
  fileChips.innerHTML = '';
  Array.from(files).forEach((f, i) => {
    const chip = document.createElement('div');
    chip.className = 'file-chip';
    chip.innerHTML = `
      <span class="file-chip-icon">${fileIcon(f.name)}</span>
      <span class="file-chip-name" title="${escHtml(f.name)}">${escHtml(f.name)}</span>
      <button class="file-chip-remove" data-idx="${i}" title="Remove">✕</button>`;
    chip.querySelector('.file-chip-remove').addEventListener('click', () => {
      pendingFiles.splice(i, 1);
      renderFileChips(pendingFiles);
    });
    fileChips.appendChild(chip);
  });
}

async function handleFileUpload(files) {
  if (!files || files.length === 0) return;
  pendingFiles = Array.from(files);
  renderFileChips(pendingFiles);

  if (!activeSessionId) {
    const { session_id, title } = await apiCreateSession(files[0].name, null);
    sessions[session_id] = { session_id, title, stage: 'idle', created_at: Date.now() / 1000 };
    activeSessionId = session_id;
    subscribeSSE(session_id);
    renderSessionList();
    showChatView(files[0].name);
  }

  welcome.style.display = 'none';
  updatePipelineProgress('uploading', `Uploading ${files.length} file${files.length > 1 ? 's' : ''}…`, 2);

  try {
    await apiUploadFiles(activeSessionId, pendingFiles);
    pendingFiles = [];
    fileChips.innerHTML = '';
    sessions[activeSessionId].title = files[0].name;
    renderSessionList();
  } catch (err) {
    handlePipelineError(err.message || 'Upload failed');
    pendingFiles = [];
    fileChips.innerHTML = '';
  }
}

// ── Suggestion chips ──────────────────────────────────────────────────────────

document.querySelectorAll('.suggestion-chip').forEach(chip => {
  chip.disabled = true;
  chip.addEventListener('click', () => {
    const q = chip.dataset.q;
    if (!q || chip.disabled) return;
    msgInput.value = q;
    updateSendState();
    sendMessage();
  });
});

// ── Connection Wizard ─────────────────────────────────────────────────────────

function openWizard() {
  wizardStep   = 1;
  wizardDbType = null;
  renderWizardStep();
  wizardOverlay.style.display = 'flex';
}

function closeWizard() {
  wizardOverlay.style.display = 'none';
}

function renderWizardStep() {
  // Step indicators
  document.querySelectorAll('.wizard-step[data-step]').forEach(el => {
    const n = parseInt(el.dataset.step, 10);
    el.classList.toggle('active', n === wizardStep);
    el.classList.toggle('done',   n < wizardStep);
  });

  // Panes
  for (let i = 1; i <= 4; i++) {
    document.getElementById(`wizardStep${i}`).style.display = i === wizardStep ? '' : 'none';
  }

  // Back / Next labels
  wizardBack.style.display = wizardStep > 1 ? '' : 'none';
  wizardNext.textContent   = wizardStep < 4 ? 'Next' : 'Save & Connect';

  // Step 2: show correct fields
  if (wizardStep === 2) {
    const isFile = ['sqlite','csv','excel'].includes(wizardDbType);
    document.getElementById('wizardDbFields').style.display  = isFile ? 'none' : '';
    document.getElementById('wizardFileFields').style.display = isFile ? ''     : 'none';
    // Set default port
    const portMap = { postgres:5432, postgresql:5432, sqlserver:1433, mssql:1433, oracle:1521, mysql:3306 };
    if (!isFile) document.getElementById('wPort').placeholder = portMap[wizardDbType] || '5432';
    testConnResult.textContent = '';
    testConnResult.className   = 'test-conn-result';
  }

  // Step 4: confirm summary
  if (wizardStep === 4) renderConfirmSummary();
}

function renderConfirmSummary() {
  const name   = document.getElementById('wName').value.trim();
  const desc   = document.getElementById('wDesc').value.trim();
  const domain = document.getElementById('wDomain').value;
  const access = document.getElementById('wAccess').value;

  const accessLabel = { all: 'All users', analyst: 'Analysts & Admins', admin: 'Admins only' }[access] || access;
  const isFile = ['sqlite','csv','excel'].includes(wizardDbType);

  let connRow = '';
  if (isFile) {
    connRow = `<div class="confirm-row"><span class="confirm-key">File path</span><span class="confirm-val">${escHtml(document.getElementById('wFilePath').value || '—')}</span></div>`;
  } else {
    const host = document.getElementById('wHost').value;
    const port = document.getElementById('wPort').value;
    const db   = document.getElementById('wDatabase').value;
    connRow = `<div class="confirm-row"><span class="confirm-key">Connection</span><span class="confirm-val">${escHtml(host)}:${escHtml(port)}/${escHtml(db)}</span></div>`;
  }

  document.getElementById('confirmSummary').innerHTML = `
    <div class="confirm-row"><span class="confirm-key">Name</span><span class="confirm-val">${escHtml(name)}</span></div>
    <div class="confirm-row"><span class="confirm-key">Type</span><span class="confirm-val">${escHtml(dbTypeLabel(wizardDbType))}</span></div>
    ${connRow}
    <div class="confirm-row"><span class="confirm-key">Domain</span><span class="confirm-val">${escHtml(domain)}</span></div>
    <div class="confirm-row"><span class="confirm-key">Access</span><span class="confirm-val">${escHtml(accessLabel)}</span></div>
    ${desc ? `<div class="confirm-row"><span class="confirm-key">Description</span><span class="confirm-val">${escHtml(desc)}</span></div>` : ''}`;
}

async function advanceWizard() {
  if (wizardStep === 1) {
    if (!wizardDbType) { showToast('Please select a database type', 'error'); return; }
    wizardStep = 2;
  } else if (wizardStep === 2) {
    // Validate required fields
    const isFile = ['sqlite','csv','excel'].includes(wizardDbType);
    if (isFile) {
      if (!document.getElementById('wFilePath').value.trim()) { showToast('File path is required', 'error'); return; }
    } else {
      if (!document.getElementById('wHost').value.trim())     { showToast('Host is required', 'error'); return; }
      if (!document.getElementById('wDatabase').value.trim()) { showToast('Database name is required', 'error'); return; }
    }
    wizardStep = 3;
  } else if (wizardStep === 3) {
    if (!document.getElementById('wName').value.trim()) { showToast('Display name is required', 'error'); return; }
    wizardStep = 4;
  } else if (wizardStep === 4) {
    await saveSource();
    return;
  }
  renderWizardStep();
}

function buildWizardPayload() {
  const isFile = ['sqlite','csv','excel'].includes(wizardDbType);
  const access = document.getElementById('wAccess').value;
  const personaMap = {
    all:     ['business_user','analyst','admin'],
    analyst: ['analyst','admin'],
    admin:   ['admin'],
  };
  const conn = isFile
    ? { file_path: document.getElementById('wFilePath').value.trim() }
    : {
        host:              document.getElementById('wHost').value.trim(),
        port:              parseInt(document.getElementById('wPort').value, 10) || 5432,
        database:          document.getElementById('wDatabase').value.trim(),
        schema_:           document.getElementById('wSchema').value.trim() || 'public',
        username:          document.getElementById('wUsername').value.trim(),
        password:          document.getElementById('wPassword').value,
        connection_string: document.getElementById('wConnStr').value.trim(),
      };
  return {
    name:           document.getElementById('wName').value.trim(),
    description:    document.getElementById('wDesc').value.trim(),
    domain:         document.getElementById('wDomain').value,
    db_type:        wizardDbType,
    connection:     conn,
    persona_access: personaMap[access] || personaMap.all,
    auto_index:     document.getElementById('wAutoIndex').checked,
  };
}

async function saveSource() {
  wizardNext.disabled = true;
  wizardNext.textContent = 'Saving…';
  try {
    const payload = buildWizardPayload();
    const src     = await apiCreateSource(payload);
    sources[src.id] = src;
    closeWizard();
    renderSourceCatalog();
    renderSourceSidebar();
    showToast(`"${src.name}" registered${payload.auto_index ? ' — indexing started' : ''}`, 'success');
    // Poll if indexing
    if (payload.auto_index) pollSourceStatus(src.id);
  } catch (err) {
    showToast(err.message || 'Failed to save source', 'error');
  } finally {
    wizardNext.disabled = false;
    wizardNext.textContent = 'Save & Connect';
  }
}

async function pollSourceStatus(sourceId, intervalMs = 5000, maxMs = 600000) {
  const start = Date.now();
  const timer = setInterval(async () => {
    if (Date.now() - start > maxMs) { clearInterval(timer); return; }
    const s = await apiGetSource(sourceId);
    if (!s) { clearInterval(timer); return; }
    sources[sourceId] = s;
    renderSourceCatalog();
    renderSourceSidebar();
    if (s.status === 'ready') {
      clearInterval(timer);
      showToast(`"${s.name}" is ready — ${s.table_count} table${s.table_count !== 1 ? 's' : ''} indexed`, 'success');
    } else if (s.status === 'error') {
      clearInterval(timer);
      showToast(`Indexing failed for "${s.name}": ${s.error_message}`, 'error', 8000);
    }
  }, intervalMs);
}

// ── Test connection (wizard step 2) ───────────────────────────────────────────

async function testConnection() {
  testConnBtn.disabled     = true;
  testConnResult.textContent = 'Testing…';
  testConnResult.className   = 'test-conn-result';
  try {
    const isFile = ['sqlite','csv','excel'].includes(wizardDbType);
    const conn   = isFile
      ? { file_path: document.getElementById('wFilePath').value.trim() }
      : {
          host:              document.getElementById('wHost').value.trim(),
          port:              parseInt(document.getElementById('wPort').value, 10) || 5432,
          database:          document.getElementById('wDatabase').value.trim(),
          schema_:           document.getElementById('wSchema').value.trim() || 'public',
          username:          document.getElementById('wUsername').value.trim(),
          password:          document.getElementById('wPassword').value,
          connection_string: document.getElementById('wConnStr').value.trim(),
        };
    const result = await apiTestConnection({ db_type: wizardDbType, connection: conn });
    if (result.ok) {
      testConnResult.textContent = '✓ Connection successful';
      testConnResult.className   = 'test-conn-result ok';
    } else {
      testConnResult.textContent = `✗ ${result.error || 'Connection failed'}`;
      testConnResult.className   = 'test-conn-result fail';
    }
  } catch (err) {
    testConnResult.textContent = `✗ ${err.message}`;
    testConnResult.className   = 'test-conn-result fail';
  } finally {
    testConnBtn.disabled = false;
  }
}

// ── Admin Panel ───────────────────────────────────────────────────────────────

function openAdminPanel() {
  renderAdminTable();
  adminOverlay.style.display = 'flex';
}

function closeAdminPanel() {
  adminOverlay.style.display = 'none';
}

function renderAdminTable() {
  const list = Object.values(sources);
  if (list.length === 0) {
    adminTableBody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--clr-text-mute);padding:24px">No data sources registered yet.</td></tr>';
    return;
  }
  adminTableBody.innerHTML = list.map(s => `
    <tr>
      <td>
        <div style="font-weight:600">${escHtml(s.icon)} ${escHtml(s.name)}</div>
        ${s.description ? `<div style="font-size:12px;color:var(--clr-text-mute)">${escHtml(s.description)}</div>` : ''}
      </td>
      <td>${escHtml(dbTypeLabel(s.db_type))}</td>
      <td>${escHtml(s.domain)}</td>
      <td><span class="status-pill ${s.status}">${escHtml(s.status)}</span></td>
      <td>${s.table_count}</td>
      <td>
        <div class="admin-actions">
          <button class="admin-action-btn" data-action="reindex" data-id="${s.id}">Reindex</button>
          <button class="admin-action-btn danger" data-action="delete" data-id="${s.id}">Delete</button>
        </div>
      </td>
    </tr>`).join('');

  adminTableBody.querySelectorAll('[data-action]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id     = btn.dataset.id;
      const action = btn.dataset.action;
      const src    = sources[id];
      if (action === 'reindex') {
        try {
          await apiReindexSource(id);
          sources[id].status = 'indexing';
          showToast(`Reindexing "${src.name}"…`, 'info');
          renderAdminTable();
          renderSourceCatalog();
          renderSourceSidebar();
          pollSourceStatus(id);
        } catch (err) { showToast(err.message, 'error'); }
      } else if (action === 'delete') {
        if (!confirm(`Delete source "${src.name}"? Existing conversations will still work.`)) return;
        try {
          await apiDeleteSource(id);
          delete sources[id];
          showToast(`"${src.name}" deleted`, 'info');
          renderAdminTable();
          renderSourceCatalog();
          renderSourceSidebar();
        } catch (err) { showToast(err.message, 'error'); }
      }
    });
  });
}

// ── Event listeners ───────────────────────────────────────────────────────────

// Sidebar toggle
function toggleSidebar() { sidebar.classList.toggle('collapsed'); }
sidebarToggle.addEventListener('click', toggleSidebar);
menuBtn.addEventListener('click', toggleSidebar);

// New chat
newChatBtn.addEventListener('click', createNewSession);

// Browse sources button
const browseSourcesBtn = document.getElementById('browseSourcesBtn');
browseSourcesBtn.addEventListener('click', async () => {
  await loadSources();
  showLanding();
});

// Persona switcher
personaSwitchBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  const shown = personaDropdown.style.display !== 'none';
  personaDropdown.style.display = shown ? 'none' : 'flex';
  if (!shown) {
    // Position relative to sidebar
    const rect = sidebar.getBoundingClientRect();
    personaDropdown.style.left = `${rect.left + 8}px`;
    personaDropdown.style.top  = `${rect.top + 110}px`;
    personaDropdown.style.flexDirection = 'column';
    personaDropdown.style.width = `${rect.width - 16}px`;
  }
});
document.querySelectorAll('.persona-option').forEach(el => {
  el.addEventListener('click', () => switchPersona(el.dataset.persona));
});
document.addEventListener('click', (e) => {
  if (!personaDropdown.contains(e.target) && !personaSwitchBtn.contains(e.target)) {
    personaDropdown.style.display = 'none';
  }
});

// Admin panel
adminBtn.addEventListener('click', openAdminPanel);
adminClose.addEventListener('click', closeAdminPanel);
adminRefreshBtn.addEventListener('click', async () => { await loadSources(); renderAdminTable(); });
adminAddBtn.addEventListener('click', () => { closeAdminPanel(); openWizard(); });
adminOverlay.addEventListener('click', (e) => { if (e.target === adminOverlay) closeAdminPanel(); });

// Wizard
wizardClose.addEventListener('click', closeWizard);
wizardBack.addEventListener('click', () => { wizardStep--; renderWizardStep(); });
wizardNext.addEventListener('click', advanceWizard);
testConnBtn.addEventListener('click', testConnection);
wizardOverlay.addEventListener('click', (e) => { if (e.target === wizardOverlay) closeWizard(); });

// DB type cards (step 1)
document.querySelectorAll('.db-type-card').forEach(card => {
  card.addEventListener('click', () => {
    document.querySelectorAll('.db-type-card').forEach(c => c.classList.remove('active'));
    card.classList.add('active');
    wizardDbType = card.dataset.dbtype;
  });
});

// File input
fileInput.addEventListener('change', (e) => {
  handleFileUpload(e.target.files);
  fileInput.value = '';
});

// Drag and drop
chatArea.addEventListener('dragover',  (e) => { e.preventDefault(); chatArea.style.outline = '2px dashed var(--clr-primary)'; });
chatArea.addEventListener('dragleave', ()  => { chatArea.style.outline = ''; });
chatArea.addEventListener('drop', (e) => {
  e.preventDefault();
  chatArea.style.outline = '';
  handleFileUpload(e.dataTransfer.files);
});

// Textarea auto-resize + send
msgInput.addEventListener('input', () => {
  msgInput.style.height = 'auto';
  msgInput.style.height = Math.min(msgInput.scrollHeight, 160) + 'px';
  updateSendState();
});
msgInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
sendBtn.addEventListener('click', sendMessage);

// Purge file cache on leave
window.addEventListener('beforeunload', () => {
  if (activeSessionId) navigator.sendBeacon(`${API}/sessions/${activeSessionId}/file-cache`);
});

// ── Init ──────────────────────────────────────────────────────────────────────

(async function init() {
  applyPersona();
  await Promise.all([loadSources(), loadSessions()]);

  const sorted   = Object.values(sessions).sort((a, b) => b.created_at - a.created_at);
  const hasSources = Object.keys(sources).length > 0;

  if (sorted.length > 0 && sorted[0].stage === 'ready') {
    // Resume the most recent ready session
    await resumeSession(sorted[0].session_id);
  } else if (hasSources) {
    // Sources registered — show the catalog so users can pick one
    showLanding();
  } else {
    // No sessions, no sources — go straight to chat (like the original UI)
    showChatView('');
  }

  // Poll indexing sources on load
  Object.values(sources).forEach(s => {
    if (s.status === 'indexing') pollSourceStatus(s.id);
  });
})();
