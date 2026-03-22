/**
 * DataChat — frontend application
 *
 * Responsibilities:
 *  - Session lifecycle (create, switch, delete)
 *  - File upload → shows chips, triggers pipeline
 *  - SSE subscription → renders pipeline progress + chat responses in real-time
 *  - Chat message send / receive
 *  - Markdown rendering (marked.js)
 *  - Query result table rendering
 *  - SQL disclosure toggle
 *  - Toast notifications
 */

'use strict';

// ── Config ────────────────────────────────────────────────────────────────────
const API = '';   // same origin — orchestrator serves the UI

// ── State ─────────────────────────────────────────────────────────────────────
let activeSessionId   = null;
let activeEventSource = null;
let pendingFiles      = [];          // FileList items staged for upload
let isWaitingForReply = false;       // true while a chat response is in-flight
let progressMsgId     = null;        // DOM id of the current pipeline progress card
let sessions          = {};          // session_id → {id, title, stage, ...}

// ── DOM refs ──────────────────────────────────────────────────────────────────
const sidebar       = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
const menuBtn       = document.getElementById('menuBtn');
const newChatBtn    = document.getElementById('newChatBtn');
const sessionList   = document.getElementById('sessionList');
const welcome       = document.getElementById('welcome');
const messages      = document.getElementById('messages');
const chatArea      = document.getElementById('chatArea');
const fileInput     = document.getElementById('fileInput');
const fileChips     = document.getElementById('fileChips');
const msgInput      = document.getElementById('msgInput');
const sendBtn       = document.getElementById('sendBtn');
const pipelineStatus = document.getElementById('pipelineStatus');
const pipelineBar    = document.getElementById('pipelineBar');
const pipelineBarInner = document.getElementById('pipelineBarInner');
const toastContainer = document.getElementById('toastContainer');
const suggestionChips = document.querySelectorAll('.suggestion-chip');

// ── Utility ───────────────────────────────────────────────────────────────────

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
    if (Number.isInteger(v)) return v.toLocaleString();
    return parseFloat(v.toFixed(4)).toLocaleString();
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
  // Fallback — plain text with newlines as <br>
  return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
             .replace(/\n/g,'<br>');
}

// ── Session API calls ─────────────────────────────────────────────────────────

async function apiCreateSession(title = 'New conversation') {
  const r = await fetch(`${API}/sessions`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ title }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();   // {session_id, title}
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
  const r = await fetch(`${API}/sessions/${sessionId}/upload`, {
    method: 'POST',
    body: fd,
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({detail: r.statusText}));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}

async function apiSendChat(sessionId, message) {
  const r = await fetch(`${API}/sessions/${sessionId}/chat`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ message }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({detail: r.statusText}));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();   // {msg_id}
}

// ── SSE subscription ──────────────────────────────────────────────────────────

function subscribeSSE(sessionId) {
  if (activeEventSource) {
    activeEventSource.close();
    activeEventSource = null;
  }
  const es = new EventSource(`${API}/sessions/${sessionId}/events`);
  activeEventSource = es;

  es.onmessage = (ev) => {
    let event;
    try { event = JSON.parse(ev.data); } catch { return; }
    handleSSEEvent(event);
  };
  es.onerror = () => {
    // SSE reconnects automatically; no action needed
  };
}

function handleSSEEvent(ev) {
  switch (ev.type) {
    case 'heartbeat': return;

    case 'progress':
      if (ev.background) {
        updateTopbarStatus(ev.message, 'running');
      } else {
        updatePipelineProgress(ev.stage, ev.message, ev.pct || 0);
      }
      break;

    case 'ready':
      finishPipeline(ev.message, ev.tables || []);
      break;

    case 'ontology_ready':
      showToast('Ontology built successfully', 'success');
      updateTopbarStatus('Ontology ready', 'done');
      break;

    case 'kg_ready':
      showToast(ev.message || 'Knowledge graph ready', 'success');
      updateTopbarStatus('KG ready', 'done');
      break;

    case 'info':
      if (ev.background) updateTopbarStatus(ev.message, 'done');
      break;

    case 'error':
      handlePipelineError(ev.message);
      break;

    case 'user_message':
      // Reflected back — only render if not already shown (optimistic render handles it)
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
  // Show/update a progress card in the chat
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

  // Also update inline progress bar above input
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
      <div class="progress-bar-wrap">
        <div class="progress-bar-fill" style="width:${pct}%"></div>
      </div>
    </div>`;
}

function finishPipeline(message, tables) {
  // Replace progress card with ready card
  if (progressMsgId) {
    const el = document.getElementById(progressMsgId);
    if (el) {
      el.innerHTML = buildReadyCard(message, tables);
    }
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

  // Enable input
  msgInput.disabled = false;
  msgInput.placeholder = 'Ask anything about your data…';
  updateSendState();

  // Enable suggestion chips
  suggestionChips.forEach(c => c.disabled = false);

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
    if (el) {
      el.innerHTML = `
        <div class="error-card">
          <div class="error-card-icon">⚠️</div>
          <div class="error-card-title">${escHtml(message)}</div>
        </div>`;
    }
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
      if (pipelineStatus.querySelector('.dot.done')) {
        pipelineStatus.classList.remove('visible');
      }
    }, 4000);
  }
}

// ── Chat message rendering ────────────────────────────────────────────────────

function renderUserMessage(message, msgId) {
  const row = document.createElement('div');
  row.className = 'msg-row user';
  row.id = 'msg-' + msgId;
  row.innerHTML = `
    <div class="msg-avatar">U</div>
    <div class="msg-bubble">${escHtml(message)}</div>`;
  messages.appendChild(row);
  welcome.style.display = 'none';
  scrollToBottom();
}

function showTypingIndicator(msgId) {
  const existing = document.getElementById('typing-' + msgId);
  if (existing) return;
  const row = document.createElement('div');
  row.className = 'msg-row assistant';
  row.id = 'typing-' + msgId;
  row.innerHTML = `
    <div class="msg-avatar">⬡</div>
    <div class="msg-bubble">
      <div class="typing-indicator">
        <span></span><span></span><span></span>
      </div>
    </div>`;
  messages.appendChild(row);
  scrollToBottom();
}

function hideTypingIndicator(msgId) {
  const el = document.getElementById('typing-' + msgId);
  if (el) el.remove();
}

function renderAIMessage(ev) {
  const row = document.createElement('div');
  row.className = 'msg-row assistant';
  row.id = 'ai-msg-' + ev.msg_id;

  const markdownHtml = renderMarkdown(ev.content || '');
  const resultsHtml  = buildResultBlocks(ev.results || []);
  const sqlHtml      = buildSQLDisclosure(ev.sql || [], ev.msg_id);
  const errHtml      = buildErrorNotes(ev.errors || []);
  const cacheNote    = ev.cache_hit
    ? '<div style="font-size:11px;color:var(--clr-text-mute);margin-top:8px">⚡ From cache</div>'
    : '';

  row.innerHTML = `
    <div class="msg-avatar">⬡</div>
    <div class="msg-bubble">
      <div class="md-content">${markdownHtml}</div>
      ${resultsHtml}
      ${errHtml}
      ${sqlHtml}
      ${cacheNote}
    </div>`;
  messages.appendChild(row);
  scrollToBottom();
}

function renderAIError(msgId, message) {
  const row = document.createElement('div');
  row.className = 'msg-row assistant';
  row.innerHTML = `
    <div class="msg-avatar">⬡</div>
    <div class="msg-bubble">
      <div class="msg-error-note">⚠️ ${escHtml(message)}</div>
    </div>`;
  messages.appendChild(row);
  scrollToBottom();
}

function buildResultBlocks(results) {
  if (!results || results.length === 0) return '';
  return results.map((r, i) => {
    const rows = r.rows || [];
    if (!rows.length) return '';
    const cols = Object.keys(rows[0]);
    const header = cols.map(c => `<th>${escHtml(c)}</th>`).join('');
    const body   = rows.map(row =>
      `<tr>${cols.map(c => {
        const v = row[c];
        const num = isNumeric(v);
        return `<td class="${num ? 'num-cell' : ''}" title="${escHtml(String(v ?? ''))}">${escHtml(formatNumber(v))}</td>`;
      }).join('')}</tr>`
    ).join('');

    return `
      <div class="result-block">
        <div class="result-block-header">
          📋 ${escHtml(r.query_label || `Query ${i+1}`)}
          <span style="margin-left:auto;font-weight:normal;color:var(--clr-text-mute)">${rows.length} row${rows.length !== 1 ? 's' : ''}</span>
        </div>
        <div class="result-table-wrap">
          <table class="result-table">
            <thead><tr>${header}</tr></thead>
            <tbody>${body}</tbody>
          </table>
        </div>
      </div>`;
  }).join('');
}

function buildSQLDisclosure(sqlQueries, msgId) {
  if (!sqlQueries || sqlQueries.length === 0) return '';
  const uniqueId = 'sql-' + msgId;
  const blocks = sqlQueries.map((q, i) => {
    const label = q.query_label || `Query ${i+1}`;
    const sql   = q.sql || '';
    return `<div style="margin-bottom:8px"><div style="font-size:11px;color:var(--clr-text-mute);padding:6px 0 4px">${escHtml(label)}</div><pre>${escHtml(sql)}</pre></div>`;
  }).join('');

  return `
    <div class="sql-disclosure">
      <button class="sql-toggle" onclick="toggleSQL('${uniqueId}', this)">
        <span class="sql-toggle-icon" id="icon-${uniqueId}">▶</span>
        ${sqlQueries.length} SQL quer${sqlQueries.length === 1 ? 'y' : 'ies'}
      </button>
      <div class="sql-block" id="${uniqueId}">
        ${blocks}
      </div>
    </div>`;
}

function buildErrorNotes(errors) {
  if (!errors || errors.length === 0) return '';
  return errors.map(e => `<div class="msg-error-note">⚠️ ${escHtml(String(e))}</div>`).join('');
}

// ── Global toggle for SQL disclosure (called from inline onclick) ─────────────
window.toggleSQL = function(id, btn) {
  const block = document.getElementById(id);
  const icon  = document.getElementById('icon-' + id);
  if (!block) return;
  const open = block.classList.toggle('visible');
  if (icon) {
    icon.textContent = open ? '▼' : '▶';
    icon.classList.toggle('open', open);
  }
};

// ── Send logic ────────────────────────────────────────────────────────────────

function updateSendState() {
  const hasText = msgInput.value.trim().length > 0;
  const ready   = !isWaitingForReply && !msgInput.disabled;
  sendBtn.disabled = !(hasText && ready);
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
    showToast(err.message || 'Failed to send message', 'error');
  }
}

// ── Session management ────────────────────────────────────────────────────────

async function loadSessions() {
  try {
    const list = await apiListSessions();
    sessions = {};
    list.forEach(s => sessions[s.session_id] = s);
    renderSessionList();
  } catch {
    // Silently ignore — sessions are non-critical on load
  }
}

function renderSessionList() {
  sessionList.innerHTML = '';
  const sorted = Object.values(sessions).sort((a, b) => b.created_at - a.created_at);
  if (sorted.length === 0) {
    sessionList.innerHTML = '<div style="font-size:13px;color:var(--clr-text-mute);padding:12px 14px">No conversations yet</div>';
    return;
  }
  sorted.forEach(s => {
    const btn = document.createElement('button');
    btn.className = 'session-item' + (s.session_id === activeSessionId ? ' active' : '');
    btn.innerHTML = `
      <div class="session-item-icon">${s.stage === 'ready' ? '💬' : s.stage === 'error' ? '⚠️' : '📁'}</div>
      <span class="session-item-title">${escHtml(s.title || 'Untitled')}</span>
      <button class="session-item-del" title="Delete" data-id="${s.session_id}">✕</button>`;
    btn.addEventListener('click', (e) => {
      if (e.target.closest('.session-item-del')) return;
      switchSession(s.session_id);
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
        clearChatUI();
      }
      renderSessionList();
    });
    sessionList.appendChild(btn);
  });
}

async function switchSession(sessionId) {
  if (sessionId === activeSessionId) return;
  activeSessionId = sessionId;

  // Re-subscribe SSE
  subscribeSSE(sessionId);

  // Reset UI
  clearChatUI();
  renderSessionList();

  // Reload messages from server
  try {
    const r = await fetch(`${API}/sessions/${sessionId}/messages`);
    const data = await r.json();

    data.messages.forEach(msg => {
      if (msg.role === 'user') {
        renderUserMessage(msg.content, msg.id);
      } else if (msg.role === 'assistant' && !msg.error) {
        renderAIMessage({
          msg_id: msg.id,
          content: msg.content,
          results: msg.results || [],
          sql: msg.sql || [],
          errors: msg.errors || [],
        });
      } else if (msg.error) {
        renderAIError(msg.id, msg.content);
      }
    });

    if (data.stage === 'ready') {
      msgInput.disabled = false;
      msgInput.placeholder = 'Ask anything about your data…';
      const tables = data.files || [];
      updateTopbarStatus('Ready', 'done');
    }
    if (data.messages.length > 0) welcome.style.display = 'none';

    scrollToBottom(false);
  } catch {
    // Ignore reload errors
  }
}

function clearChatUI() {
  messages.innerHTML = '';
  progressMsgId = null;
  isWaitingForReply = false;
  msgInput.disabled = true;
  msgInput.value = '';
  msgInput.placeholder = 'Upload files to get started…';
  fileChips.innerHTML = '';
  pendingFiles = [];
  pipelineBar.style.display = 'none';
  pipelineStatus.classList.remove('visible');
  welcome.style.display = '';
  updateSendState();
}

async function createNewSession() {
  pendingFiles = [];
  fileChips.innerHTML = '';
  fileInput.value = '';

  const { session_id, title } = await apiCreateSession('New conversation');
  sessions[session_id] = { session_id, title, stage: 'idle', created_at: Date.now() / 1000 };
  activeSessionId = session_id;
  subscribeSSE(session_id);
  clearChatUI();
  renderSessionList();
  msgInput.focus();
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

  // Auto-create or ensure a session exists
  if (!activeSessionId) {
    const { session_id, title } = await apiCreateSession(files[0].name);
    sessions[session_id] = { session_id, title, stage: 'idle', created_at: Date.now() / 1000 };
    activeSessionId = session_id;
    subscribeSSE(session_id);
    renderSessionList();
  }

  // Show initial progress message
  welcome.style.display = 'none';
  updatePipelineProgress('uploading', `Uploading ${files.length} file${files.length > 1 ? 's' : ''}…`, 2);

  try {
    await apiUploadFiles(activeSessionId, pendingFiles);
    // Pipeline events will arrive over SSE
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

suggestionChips.forEach(chip => {
  chip.disabled = true;  // enabled once pipeline is ready
  chip.addEventListener('click', () => {
    const q = chip.dataset.q;
    if (!q || chip.disabled) return;
    msgInput.value = q;
    updateSendState();
    sendMessage();
  });
});

// ── Event listeners ───────────────────────────────────────────────────────────

// Sidebar toggle
function toggleSidebar() {
  sidebar.classList.toggle('collapsed');
}
sidebarToggle.addEventListener('click', toggleSidebar);
menuBtn.addEventListener('click', toggleSidebar);

// New chat
newChatBtn.addEventListener('click', createNewSession);

// File input
fileInput.addEventListener('change', (e) => {
  handleFileUpload(e.target.files);
  fileInput.value = '';   // allow re-selecting same file
});

// Drag and drop on chat area
chatArea.addEventListener('dragover', (e) => { e.preventDefault(); chatArea.style.outline = '2px dashed var(--clr-primary)'; });
chatArea.addEventListener('dragleave', () => { chatArea.style.outline = ''; });
chatArea.addEventListener('drop', (e) => {
  e.preventDefault();
  chatArea.style.outline = '';
  handleFileUpload(e.dataTransfer.files);
});

// Text input auto-resize + send state
msgInput.addEventListener('input', () => {
  msgInput.style.height = 'auto';
  msgInput.style.height = Math.min(msgInput.scrollHeight, 160) + 'px';
  updateSendState();
});

msgInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);

// Purge file cache when user leaves the page
window.addEventListener('beforeunload', () => {
  if (activeSessionId) {
    navigator.sendBeacon(`${API}/sessions/${activeSessionId}/file-cache`);
  }
});

// ── Helpers ───────────────────────────────────────────────────────────────────

function escHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Init ──────────────────────────────────────────────────────────────────────

(async function init() {
  await loadSessions();

  // If sessions exist, switch to the most recent
  const sorted = Object.values(sessions).sort((a, b) => b.created_at - a.created_at);
  if (sorted.length > 0) {
    await switchSession(sorted[0].session_id);
  }
})();
