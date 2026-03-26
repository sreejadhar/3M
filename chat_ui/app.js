'use strict';

// ── Config ────────────────────────────────────────────────────────────────────
const API = '';   // same origin

// ── Persona ───────────────────────────────────────────────────────────────────
const PERSONAS = {
  business_user: { label: 'Business User',    icon: '👤', showSQL: false, canConnect: false, isAdmin: false, isDataManager: false },
  analyst:       { label: 'Business Analyst', icon: '🔬', showSQL: true,  canConnect: true,  isAdmin: false, isDataManager: false },
  admin:         { label: 'Data Admin',       icon: '⚙️', showSQL: true,  canConnect: true,  isAdmin: true,  isDataManager: false },
  data_manager:  { label: 'Data Manager',     icon: '🗂️', showSQL: false, canConnect: false, isAdmin: false, isDataManager: true  },
};

// ── Analyst roles (business_user sub-personas) ─────────────────────────────
const ANALYST_ROLES = [
  { key: 'supply_chain',  label: 'Supply Chain Analyst',       icon: '🔗' },
  { key: 'program_mgmt',  label: 'Program Management Analyst', icon: '📋' },
  { key: 'financial',     label: 'Financial Analyst',          icon: '💰' },
  { key: 'people',        label: 'People & Workforce Analyst', icon: '👥' },
  { key: 'logistics',     label: 'Logistics Analyst',          icon: '🚚' },
  { key: 'sales',         label: 'Sales Analyst',              icon: '📈' },
  { key: 'marketing',     label: 'Marketing Analyst',          icon: '📣' },
  { key: 'it',            label: 'IT Analyst',                 icon: '💻' },
  { key: 'other',         label: 'Other',                      icon: '✏️' },
];

let currentPersona     = localStorage.getItem('datachat_persona') || 'business_user';
// currentAnalystRole: '' | role key | 'other' (pending) | 'other:Custom text'
let currentAnalystRole = localStorage.getItem('datachat_analyst_role') || '';

// ── State ─────────────────────────────────────────────────────────────────────
let activeSessionId   = null;
let activeEventSource = null;
let pendingFiles      = [];
let isWaitingForReply = false;
let progressMsgId     = null;
let sessions          = {};
let sources           = {};       // source_id → source dict
let activeSourceId    = null;     // selected source on landing
let _newChatRequested = false;    // set by "New chat" btn; forces a fresh session on next source open
let wizardStep             = 1;
let wizardDbType           = null;
let wizardUploadedPath     = null;   // server path returned by /sources/upload-file
let wizardUploadedFilename = null;   // original filename
let wizardUploadedDbType   = null;   // actual db_type from upload response (may differ from wizardDbType)

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

const uploadBtn        = document.getElementById('uploadBtn');
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

const dmCatalogBtn         = document.getElementById('dmCatalogBtn');
const mdCatalog            = document.getElementById('mdCatalog');
const mdSourceFilter       = document.getElementById('mdSourceFilter');
const mdSearch             = document.getElementById('mdSearch');
const mdRefreshBtn         = document.getElementById('mdRefreshBtn');
const mdEntityBody         = document.getElementById('mdEntityBody');
const mdAttrPanel          = document.getElementById('mdAttrPanel');
const mdAttrPanelTitle     = document.getElementById('mdAttrPanelTitle');
const mdAttrClose          = document.getElementById('mdAttrClose');
const mdAttrBody           = document.getElementById('mdAttrBody');
const mdTabBar             = document.getElementById('mdTabBar');
const mdRedundancyBody     = document.getElementById('mdRedundancyBody');
const mdRedundancyEmpty    = document.getElementById('mdRedundancyEmpty');
const mdRedundancyTable    = document.getElementById('mdRedundancyTable');
const mdChangesBody        = document.getElementById('mdChangesBody');
const mdChangesEmpty       = document.getElementById('mdChangesEmpty');
const mdChangesTable       = document.getElementById('mdChangesTable');
const mdSourcesBody        = document.getElementById('mdSourcesBody');
const mdSourcesEmpty       = document.getElementById('mdSourcesEmpty');
const mdSourcesTable       = document.getElementById('mdSourcesTable');

const bridgeManagerOverlay = document.getElementById('bridgeManagerOverlay');
const bridgeManagerClose   = document.getElementById('bridgeManagerClose');
const bridgeAddBtn         = document.getElementById('bridgeAddBtn');
const bridgeRefreshBtn     = document.getElementById('bridgeRefreshBtn');
const bridgeFilterSource   = document.getElementById('bridgeFilterSource');
const bridgeTableBody      = document.getElementById('bridgeTableBody');
const bridgeFormPanel      = document.getElementById('bridgeFormPanel');
const bridgeFormTitle      = document.getElementById('bridgeFormTitle');
const bridgeFormSave       = document.getElementById('bridgeFormSave');
const bridgeFormCancel     = document.getElementById('bridgeFormCancel');
const bfFromKg             = document.getElementById('bfFromKg');
const bfFromEntity         = document.getElementById('bfFromEntity');
const bfFromCol            = document.getElementById('bfFromCol');
const bfToKg               = document.getElementById('bfToKg');
const bfToEntity           = document.getElementById('bfToEntity');
const bfToCol              = document.getElementById('bfToCol');
const bfJoinType           = document.getElementById('bfJoinType');
const bfNotes              = document.getElementById('bfNotes');

// ── Chart state ───────────────────────────────────────────────────────────────
const _charts = {};   // canvasId → Chart.js instance

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

// ── Chart helpers ─────────────────────────────────────────────────────────────

const CHART_PALETTE = [
  '#4285F4','#9B72CB','#EA4335','#34A853','#FBBC04','#FF6D00','#00ACC1','#3F51B5',
];

/** Convert a QueryResult (columns + row arrays) into an array of row objects. */
function normalizeRows(result) {
  const rawRows = result.rows || [];
  const cols    = result.columns || Object.keys(rawRows[0] || {});
  return {
    cols,
    rows: rawRows.map(row =>
      Array.isArray(row)
        ? Object.fromEntries(cols.map((c, j) => [c, row[j]]))
        : row
    ),
  };
}

function formatAxisValue(n) {
  const v = Number(n);
  if (isNaN(v)) return n;
  const abs = Math.abs(v);
  if (abs >= 1e9) return (v / 1e9).toFixed(1).replace(/\.0$/, '') + 'B';
  if (abs >= 1e6) return (v / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
  if (abs >= 1e3) return (v / 1e3).toFixed(1).replace(/\.0$/, '') + 'K';
  return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2);
}

function detectChartConfig(cols, rows) {
  if (!rows.length || cols.length < 2) return null;
  const numCols = cols.filter(c => rows.slice(0, 5).every(r => r[c] !== null && r[c] !== undefined && isNumeric(r[c])));
  const lblCols = cols.filter(c => !numCols.includes(c));
  if (!lblCols.length || !numCols.length) return null;
  const labelCol = lblCols[0];

  // Suppress charts when the label column is an identifier/key — not a meaningful category.
  // Identifiers: column name contains id/key/sku/code/uuid/hash/no/num/ref/pk suffixes.
  const isIdCol = /(\b|_)(id|key|sku|code|uuid|guid|hash|no|num|nr|ref|pk)(\b|_|$)/i.test(labelCol);
  if (isIdCol) return null;

  // Suppress charts when label values are not unique (duplicate rows = unaggregated detail data).
  const labelVals = rows.map(r => r[labelCol]);
  const hasDuplicates = labelVals.length !== new Set(labelVals).size;
  if (hasDuplicates) return null;

  const isTime = /date|month|year|week|day|quarter|period|time|fiscal/i.test(labelCol);
  if (isTime)                                             return { type: 'line',    labelCol, numCols: numCols.slice(0, 5) };
  // KPI tiles: few rows with multiple metrics — each row becomes a card
  if (rows.length <= 5 && numCols.length === 1)           return { type: 'doughnut', labelCol, numCols };
  if (rows.length <= 5 && numCols.length >= 2)            return { type: 'kpi',     labelCol, numCols: numCols.slice(0, 5) };
  // Stacked bar: multi-series breakdown
  if (numCols.length >= 2 && rows.length <= 25)           return { type: 'stacked', labelCol, numCols: numCols.slice(0, 6) };
  // Single-series fallbacks
  if (rows.length <= 8)                                   return { type: 'doughnut', labelCol, numCols: numCols.slice(0, 1) };
  if (rows.length <= 60)                                  return { type: 'hbar',    labelCol, numCols: numCols.slice(0, 1) };
  return null;
}

function renderChart(canvasId, cols, rows) {
  if (typeof Chart === 'undefined') return;
  const cfg = detectChartConfig(cols, rows);
  if (!cfg || cfg.type === 'kpi') return;  // kpi is HTML-only

  const wrap = document.getElementById(canvasId)?.parentElement;
  if (!wrap) return;

  // Dynamic height for horizontal bars
  if (cfg.type === 'hbar') {
    const rowH = cfg.numCols.length > 1 ? 28 : 34;
    wrap.style.height = Math.max(220, rows.length * rowH + 90) + 'px';
  }
  // Taller canvas for stacked bars with many groups
  if (cfg.type === 'stacked') {
    wrap.style.height = Math.max(280, rows.length * 32 + 100) + 'px';
  }

  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  if (_charts[canvasId]) { _charts[canvasId].destroy(); delete _charts[canvasId]; }

  const isDark  = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const gridClr = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.05)';
  const tickClr = isDark ? '#9aa0ac' : '#6b7280';
  const lblClr  = isDark ? '#c9d1d9' : '#374151';
  const labels  = rows.map(r => String(r[cfg.labelCol] ?? ''));
  const ctx     = canvas.getContext('2d');

  const datasets = cfg.numCols.map((col, i) => {
    const color = CHART_PALETTE[i % CHART_PALETTE.length];

    if (cfg.type === 'doughnut') {
      return {
        label: col,
        data: rows.map(r => Number(r[col]) || 0),
        backgroundColor: CHART_PALETTE.slice(0, rows.length).map(c => c + 'dd'),
        borderWidth: 3,
        borderColor: isDark ? '#1e2432' : '#ffffff',
        hoverBorderWidth: 4,
        hoverOffset: 8,
      };
    }

    if (cfg.type === 'line') {
      const h = wrap.clientHeight || 300;
      const grad = ctx.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, color + '55');
      grad.addColorStop(1, color + '00');
      return {
        label: col,
        data: rows.map(r => Number(r[col]) || 0),
        backgroundColor: grad,
        borderColor: color,
        borderWidth: 2.5,
        fill: true,
        tension: 0.4,
        pointRadius: rows.length < 25 ? 4 : 2,
        pointHoverRadius: 6,
        pointBackgroundColor: color,
        pointBorderColor: isDark ? '#1e2432' : '#fff',
        pointBorderWidth: 2,
      };
    }

    // Stacked vertical bar
    if (cfg.type === 'stacked') {
      return {
        label: col,
        data: rows.map(r => Number(r[col]) || 0),
        backgroundColor: color + 'dd',
        hoverBackgroundColor: color,
        borderRadius: i === cfg.numCols.length - 1 ? [4, 4, 0, 0] : 0,
        borderSkipped: false,
      };
    }

    // Horizontal bar
    return {
      label: col,
      data: rows.map(r => Number(r[col]) || 0),
      backgroundColor: color + 'cc',
      hoverBackgroundColor: color,
      borderRadius: 5,
      borderSkipped: false,
    };
  });

  // Datalabels: values on bars + percentages on doughnut slices
  const datalabelsPlugin = (typeof ChartDataLabels !== 'undefined');
  const datalabelsCfg = datalabelsPlugin ? {
    anchor:    cfg.type === 'doughnut' ? 'center' : 'end',
    align:     cfg.type === 'doughnut' ? 'center' : 'end',
    offset:    cfg.type === 'doughnut' ? 0 : 5,
    color:     cfg.type === 'doughnut' ? '#fff' : lblClr,
    font:      { size: 11, weight: '600' },
    formatter: (v, context) => {
      if (cfg.type === 'doughnut') {
        const total = context.dataset.data.reduce((a, b) => a + b, 0);
        return total > 0 ? (v / total * 100).toFixed(1) + '%' : '';
      }
      return formatAxisValue(v);
    },
    display: (context) => {
      if (cfg.type === 'doughnut') return context.dataset.data[context.dataIndex] > 0;
      if (cfg.type === 'line') return false;
      const max = Math.max(...context.dataset.data);
      return max > 0 ? context.dataset.data[context.dataIndex] / max > 0.06 : false;
    },
  } : false;

  const isHbar    = cfg.type === 'hbar';
  const isStacked = cfg.type === 'stacked';
  const chartType = (isHbar || isStacked) ? 'bar' : cfg.type;

  const scaleOpts = cfg.type === 'doughnut' ? {} : {
    [isHbar ? 'x' : 'y']: {
      stacked: isStacked,
      beginAtZero: true,
      grid: { color: gridClr },
      ticks: { color: tickClr, callback: formatAxisValue, maxTicksLimit: 6 },
    },
    [isHbar ? 'y' : 'x']: {
      stacked: isStacked,
      grid: { display: false },
      ticks: {
        color: tickClr,
        maxRotation: isHbar ? 0 : 35,
        callback: (_v, i) => {
          const lbl = labels[i] || '';
          return lbl.length > 26 ? lbl.slice(0, 24) + '…' : lbl;
        },
      },
    },
  };

  const plugins = {
    title: { display: false },
    legend: {
      display: cfg.numCols.length > 1 || cfg.type === 'doughnut' || isStacked,
      position: cfg.type === 'doughnut' ? 'right' : 'bottom',
      labels: {
        color: tickClr,
        boxWidth: 12,
        padding: 16,
        font: { size: 12 },
        usePointStyle: cfg.type === 'line',
      },
    },
    tooltip: {
      mode: cfg.type === 'doughnut' ? 'nearest' : 'index',
      intersect: false,
      backgroundColor: isDark ? '#2d3347' : '#ffffff',
      titleColor: lblClr,
      bodyColor: tickClr,
      borderColor: isDark ? '#3d4661' : '#e3e8f0',
      borderWidth: 1,
      padding: 10,
      callbacks: {
        label: (context) => {
          const v = isHbar ? context.parsed.x : (context.parsed.y ?? context.parsed);
          return ` ${context.dataset.label}: ${formatAxisValue(v)}`;
        },
      },
    },
    ...(datalabelsPlugin ? { datalabels: datalabelsCfg } : {}),
  };

  if (datalabelsPlugin) Chart.register(ChartDataLabels);

  _charts[canvasId] = new Chart(canvas, {
    type: chartType,
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: isHbar ? 'y' : 'x',
      // stacked bars always use vertical orientation (indexAxis stays 'x')
      plugins,
      scales: scaleOpts,
      animation: { duration: 450, easing: 'easeInOutQuart' },
    },
  });
}

// ── Insight callout post-processing ──────────────────────────────────────────

function processInsightCallouts(html) {
  const KNOWN_ICONS = ['💡','✅','⚠️','🔍','📌','🎯','⚡','🚨','📊','💰','🔑','🏆','📈','📉','🔎'];
  return html.replace(/<blockquote>\s*<p>([\s\S]*?)<\/p>\s*<\/blockquote>/g, (_, inner) => {
    let icon = '';
    let bodyText = inner.trim();
    for (const ic of KNOWN_ICONS) {
      if (bodyText.startsWith(ic)) {
        icon = ic;
        bodyText = bodyText.slice(ic.length).trim();
        break;
      }
    }
    const isRec    = /recommendation|action item|next step|should|must|consider/i.test(inner);
    const cls      = isRec ? 'callout-recommendation' : 'callout-insight';
    const iconHtml = icon ? `<div class="callout-icon">${icon}</div>` : '';
    return `<div class="insight-callout ${cls}">${iconHtml}<div class="callout-body"><p>${bodyText}</p></div></div>`;
  });
}

function fileIcon(filename) {
  const ext = (filename || '').split('.').pop().toLowerCase();
  if (['xlsx','xls','xlsm','xlsb'].includes(ext)) return '📊';
  if (ext === 'csv') return '📄';
  return '📁';
}

function renderMarkdown(text) {
  if (!text) return '';
  let html;
  if (typeof marked !== 'undefined') {
    marked.setOptions({ breaks: true, gfm: true });
    html = marked.parse(text);
  } else {
    html = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
  }
  return processInsightCallouts(html);
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

  // Show/hide role picker section in dropdown
  renderAnalystRoleSection();
  updateRoleBadge();

  // Admin / Data Manager button visibility
  sidebarBottom.style.display = (p.isAdmin || p.isDataManager) ? '' : 'none';
  adminBtn.style.display      = p.isAdmin       ? '' : 'none';
  dmCatalogBtn.style.display  = p.isDataManager ? '' : 'none';

  // Upload button: admins only
  uploadBtn.style.display = p.isAdmin ? '' : 'none';

  // SQL disclosure: show by default for analysts/admins
  document.documentElement.dataset.showSql = p.showSQL ? 'true' : 'false';

  // Data manager sees the metadata catalog instead of chat/landing
  if (p.isDataManager) {
    landing.style.display  = 'none';
    chatView.style.display = 'none';
    mdCatalog.style.display = 'flex';
    loadMdCatalog();
  } else {
    mdCatalog.style.display = 'none';
    // Reload source catalog and session list with persona filter
    renderSourceCatalog();
    renderSourceSidebar();
    loadSessions();
    return;
  }

  // Reload source catalog and session list with persona filter
  renderSourceCatalog();
  renderSourceSidebar();
  loadSessions();
}

function switchPersona(persona) {
  currentPersona = persona;
  localStorage.setItem('datachat_persona', persona);
  if (persona !== 'business_user') {
    personaDropdown.style.display = 'none';
  }
  applyPersona();
  showToast(`Switched to ${PERSONAS[persona].label}`, 'info', 2500);
}

// ── Analyst role helpers ───────────────────────────────────────────────────

function getAnalystRoleLabel() {
  if (!currentAnalystRole || currentPersona !== 'business_user') return '';
  if (currentAnalystRole.startsWith('other:')) return currentAnalystRole.slice(6).trim();
  const r = ANALYST_ROLES.find(x => x.key === currentAnalystRole);
  return r && r.key !== 'other' ? r.label : '';
}

function updateRoleBadge() {
  const roleText = document.getElementById('personaRoleText');
  if (!roleText) return;
  const label = getAnalystRoleLabel();
  roleText.textContent = label;
  roleText.style.display = label ? '' : 'none';
}

function renderAnalystRoleSection() {
  const section  = document.getElementById('personaRoleSection');
  const chipsEl  = document.getElementById('personaRoleChips');
  const customEl = document.getElementById('personaRoleCustom');
  if (!section || !chipsEl) return;

  if (currentPersona !== 'business_user') {
    section.style.display = 'none';
    return;
  }
  section.style.display = '';

  const isOtherPending  = currentAnalystRole === 'other';
  const isOtherCustom   = currentAnalystRole.startsWith('other:');
  const activeKey       = isOtherCustom ? 'other' : currentAnalystRole;

  chipsEl.innerHTML = ANALYST_ROLES.map(r =>
    `<button class="role-chip${activeKey === r.key ? ' active' : ''}"
             onclick="selectAnalystRole('${r.key}')">${r.icon} ${r.label}</button>`
  ).join('');

  if (customEl) {
    customEl.style.display = (isOtherPending || isOtherCustom) ? '' : 'none';
    if (isOtherCustom) {
      const inp = document.getElementById('personaRoleInput');
      if (inp && !inp.value) inp.value = currentAnalystRole.slice(6).trim();
    }
  }
}

window.selectAnalystRole = function(key) {
  if (key === 'other') {
    // Mark as "other pending" — show input but don't confirm yet
    currentAnalystRole = 'other';
    localStorage.setItem('datachat_analyst_role', 'other');
    renderAnalystRoleSection();
    setTimeout(() => document.getElementById('personaRoleInput')?.focus(), 50);
    return;
  }
  currentAnalystRole = key;
  localStorage.setItem('datachat_analyst_role', key);
  renderAnalystRoleSection();
  updateRoleBadge();
  // Close dropdown after picking a named role
  personaDropdown.style.display = 'none';
  const r = ANALYST_ROLES.find(x => x.key === key);
  if (r) showToast(`Role set: ${r.label}`, 'info', 2000);
};

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

async function apiGetSourceGraph(id) {
  const r = await fetch(`${API}/sources/${id}/graph`);
  if (!r.ok) return { nodes: [], edges: [] };
  return r.json();
}

async function apiGetSourceOntology(id) {
  const r = await fetch(`${API}/sources/${id}/ontology`);
  if (!r.ok) return { content: '' };
  return r.json();
}

async function apiSaveOntology(id, content, rebuildKg = true) {
  const r = await fetch(`${API}/sources/${id}/ontology`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content, rebuild_kg: rebuildKg }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function apiPreviewKG(id, ontologyText) {
  const r = await fetch(`${API}/sources/${id}/kg-preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ontology_text: ontologyText }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ── Session API calls ─────────────────────────────────────────────────────────

async function apiCreateSession(title, sourceId) {
  const body = { title, persona: currentPersona };
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
  const r = await fetch(`${API}/sessions?persona=${encodeURIComponent(currentPersona)}`);
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
    body: JSON.stringify({ message, analyst_role: getAnalystRoleLabel() }),
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
        ${s.status === 'ready' ? `<button class="source-card-kg-btn" data-id="${s.id}" title="View Knowledge Graph">KG</button>` : ''}
      </div>`;

    if (s.status === 'ready') {
      card.addEventListener('click', (e) => {
        if (e.target.closest('.source-card-kg-btn')) return;
        openSourceSession(s.id);
      });
      card.querySelector('.source-card-kg-btn')?.addEventListener('click', (e) => {
        e.stopPropagation();
        openKGExplorer(s.id);
      });
    } else if (s.status === 'indexing') {
      card.title = 'Source is being indexed. Please wait.';
    } else if (s.status === 'error') {
      card.title = s.error_message || 'Indexing failed';
    }
    sourceCatalog.appendChild(card);
  });

  // Data configuration cards — admins only
  if (p.isAdmin) {
    const uploadCard = document.createElement('div');
    uploadCard.className = 'source-card add-card';
    uploadCard.innerHTML = `
      <div class="add-card-icon">📁</div>
      <div class="add-card-label">Upload your own data</div>
      <div class="add-card-sub">CSV or Excel files</div>`;
    uploadCard.addEventListener('click', startAdHocUploadSession);
    sourceCatalog.appendChild(uploadCard);

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

  // Resume the most recent ready session for this source (within the current persona)
  // unless the user explicitly asked for a new chat.
  if (!_newChatRequested) {
    const existing = Object.values(sessions)
      .filter(s => s.source_id === sourceId && s.stage === 'ready')
      .sort((a, b) => b.created_at - a.created_at)[0];
    if (existing) {
      await resumeSession(existing.session_id);
      renderSourceSidebar();
      return;
    }
  }
  _newChatRequested = false;

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

  const mdHtml      = renderMarkdown(ev.content || '');
  const resultsHtml = buildResultBlocks(ev.results || [], ev.msg_id);
  const sqlHtml     = buildSQLDisclosure(ev.sql || [], ev.msg_id, p.showSQL);
  const errHtml     = buildErrorNotes(ev.errors || []);
  const cacheNote   = ev.cache_hit
    ? '<div style="font-size:11px;color:var(--clr-text-mute);margin-top:8px">⚡ From cache</div>' : '';

  row.innerHTML = `
    <div class="msg-avatar">⬡</div>
    <div class="msg-bubble">
      <div class="md-content">${mdHtml}</div>
      ${resultsHtml}${errHtml}${sqlHtml}${cacheNote}
    </div>`;
  messages.appendChild(row);

  // Render Chart.js charts after the canvases are in the DOM (skip HTML-only types)
  (ev.results || []).forEach((r, i) => {
    const { cols, rows } = normalizeRows(r);
    if (!rows.length) return;
    const cfg = detectChartConfig(cols, rows);
    if (!cfg || cfg.type === 'kpi') return;
    renderChart(`chart-${ev.msg_id}-${i}`, cols, rows);
  });

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

function buildResultBlocks(results, msgId) {
  if (!results || !results.length) return '';
  return results.map((r, i) => {
    const { cols, rows } = normalizeRows(r);
    if (!rows.length) return '';

    const canvasId  = `chart-${msgId}-${i}`;
    const tableId   = `tbl-${msgId}-${i}`;
    const chartable = detectChartConfig(cols, rows);
    const desc      = r.description || '';
    const rowCount  = r.row_count ?? rows.length;

    // Data table (up to 50 rows in the block; full data is in query results)
    const displayRows = rows.slice(0, 50);
    const thead = '<tr>' + cols.map(c => `<th>${escHtml(c)}</th>`).join('') + '</tr>';
    const tbody = displayRows.map(row =>
      '<tr>' + cols.map(c => {
        const v = row[c];
        return `<td class="${isNumeric(v) ? 'num-cell' : ''}">${escHtml(formatNumber(v))}</td>`;
      }).join('') + '</tr>'
    ).join('');
    const moreNote = rows.length > 50
      ? `<div class="table-more-note">Showing 50 of ${rows.length.toLocaleString()} rows</div>` : '';

    if (chartable) {
      const header = `
          <div class="result-block-header">
            <span class="result-block-icon">📊</span>
            <span class="result-block-title">${escHtml(desc)}</span>
            <span class="result-row-badge">${rowCount.toLocaleString()} rows</span>
          </div>`;
      const dataFooter = `
          <div class="result-data-footer">
            <button class="data-toggle-btn" onclick="toggleDataTable('${tableId}', this)">
              <span class="data-toggle-icon">▶</span> Show data
            </button>
          </div>
          <div class="result-table-wrap table-hidden" id="${tableId}">
            <table class="result-table"><thead>${thead}</thead><tbody>${tbody}</tbody></table>
            ${moreNote}
          </div>`;

      // KPI tiles — HTML, no canvas
      if (chartable.type === 'kpi') {
        return `<div class="result-block">${header}${buildKPICards(rows, chartable)}${dataFooter}</div>`;
      }

      // Canvas-based chart (line / doughnut / hbar / stacked)
      return `
        <div class="result-block">
          ${header}
          <div class="chart-wrap"><canvas id="${canvasId}"></canvas></div>
          ${dataFooter}
        </div>`;
    }

    // Not chartable — show table only
    return `
      <div class="result-block">
        <div class="result-block-header">
          <span class="result-block-icon">📋</span>
          <span class="result-block-title">${escHtml(desc)}</span>
          <span class="result-row-badge">${rowCount.toLocaleString()} rows</span>
        </div>
        <div class="result-table-wrap">
          <table class="result-table"><thead>${thead}</thead><tbody>${tbody}</tbody></table>
          ${moreNote}
        </div>
      </div>`;
  }).join('');
}

// ── KPI card renderer ─────────────────────────────────────────────────────────

function buildKPICards(rows, viz) {
  const { labelCol, numCols } = viz;
  // Pick accent colors cycling through palette
  const cards = rows.map(row => {
    const label = escHtml(String(row[labelCol] ?? ''));
    const metrics = numCols.map((col, i) => {
      const v = Number(row[col]);
      const fmt = isNaN(v) ? escHtml(String(row[col] ?? '—')) : escHtml(formatAxisValue(v));
      return `<div class="kpi-metric${i === 0 ? ' kpi-primary' : ''}">
        <div class="kpi-val">${fmt}</div>
        <div class="kpi-col-label">${escHtml(col)}</div>
      </div>`;
    }).join('');
    return `<div class="kpi-card">${metrics}<div class="kpi-card-title">${label}</div></div>`;
  }).join('');
  return `<div class="kpi-grid">${cards}</div>`;
}

// ── Heatmap renderer ──────────────────────────────────────────────────────────

function heatColor(t) {
  // t: 0 (low, cool green) → 1 (high, warm red-orange)
  const stops = [
    [52, 168, 83],   // #34a853 green
    [251, 188, 4],   // #fbbc04 yellow
    [234, 67, 53],   // #ea4335 red
  ];
  const seg  = t * (stops.length - 1);
  const idx  = Math.min(Math.floor(seg), stops.length - 2);
  const frac = seg - idx;
  const [r1, g1, b1] = stops[idx];
  const [r2, g2, b2] = stops[idx + 1];
  const r = Math.round(r1 + (r2 - r1) * frac);
  const g = Math.round(g1 + (g2 - g1) * frac);
  const b = Math.round(b1 + (b2 - b1) * frac);
  return `rgb(${r},${g},${b})`;
}

function buildHeatmapTable(rows, viz) {
  const { labelCol, numCols } = viz;
  const allVals = rows.flatMap(r => numCols.map(c => Number(r[c]) || 0));
  const vMin = Math.min(...allVals);
  const vMax = Math.max(...allVals);
  const range = vMax - vMin || 1;

  const thead = '<tr><th class="heatmap-th-label">' + escHtml(labelCol) + '</th>' +
    numCols.map(c => `<th class="heatmap-th">${escHtml(c)}</th>`).join('') + '</tr>';
  const tbody = rows.map(row => {
    const cells = numCols.map(col => {
      const v   = Number(row[col]) || 0;
      const t   = (v - vMin) / range;
      const bg  = heatColor(t);
      const fg  = t > 0.55 ? '#fff' : '#1f1f1f';
      return `<td class="heatmap-cell" style="background:${bg};color:${fg}">${escHtml(formatAxisValue(v))}</td>`;
    }).join('');
    return `<tr><td class="heatmap-label-cell">${escHtml(String(row[labelCol] ?? ''))}</td>${cells}</tr>`;
  }).join('');
  return `<div class="heatmap-wrap"><table class="heatmap-table"><thead>${thead}</thead><tbody>${tbody}</tbody></table></div>`;
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

window.toggleDataTable = function(id, btn) {
  const tbl = document.getElementById(id);
  if (!tbl) return;
  const hidden = tbl.classList.toggle('table-hidden');
  if (btn) btn.innerHTML = `<span class="data-toggle-icon">${hidden ? '▶' : '▼'}</span> ${hidden ? 'Show data' : 'Hide data'}`;
};

// ── Send logic ────────────────────────────────────────────────────────────────

function updateSendState() {
  const hasText = msgInput.value.trim().length > 0;
  sendBtn.disabled = !(hasText && !isWaitingForReply && !msgInput.disabled);
}

async function sendMessage() {
  const text = msgInput.value.trim();
  if (!text || isWaitingForReply || !activeSessionId) return;

  // If business_user selected "Other" but hasn't entered a custom role, prompt them first
  if (currentPersona === 'business_user' && currentAnalystRole === 'other') {
    const rect = sidebar.getBoundingClientRect();
    personaDropdown.style.display = 'flex';
    personaDropdown.style.left = `${rect.left + 8}px`;
    personaDropdown.style.top  = `${rect.top + 110}px`;
    personaDropdown.style.flexDirection = 'column';
    personaDropdown.style.width = `${rect.width - 16}px`;
    renderAnalystRoleSection();
    setTimeout(() => document.getElementById('personaRoleInput')?.focus(), 50);
    showToast('Please describe your role so I can tailor my answers to you.', 'info', 5000);
    return;
  }

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
  msgInput.placeholder = (PERSONAS[currentPersona] || PERSONAS.business_user).isAdmin
    ? 'Upload files to get started…'
    : 'Select a data source to get started…';
  fileChips.innerHTML  = '';
  pendingFiles         = [];
  pipelineBar.style.display = 'none';
  pipelineStatus.classList.remove('visible');
  welcome.style.display = 'block';
  updateSendState();
}

async function createNewSession() {
  _newChatRequested = true;
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
    clearChatUI();
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
  wizardStep             = 1;
  wizardDbType           = null;
  wizardUploadedPath     = null;
  wizardUploadedFilename = null;
  wizardUploadedDbType   = null;
  renderWizardStep();
  wizardOverlay.style.display = 'flex';
}

function closeWizard() {
  wizardOverlay.style.display = 'none';
}

async function apiUploadSourceFile(file) {
  const fd = new FormData();
  fd.append('file', file);
  const r = await fetch(`${API}/sources/upload-file`, { method: 'POST', body: fd });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}

function setWizardFileChosen(filename) {
  wizardUploadedFilename = filename;
  const ext = (filename || '').split('.').pop().toLowerCase();
  const icon = ['xlsx','xls','xlsm'].includes(ext) ? '📊' : ext === 'csv' ? '📄' : '🗄️';
  document.getElementById('sourceFileChosenIcon').textContent = icon;
  document.getElementById('sourceFileChosenName').textContent = filename;
  document.getElementById('sourceFilePrompt').style.display = 'none';
  document.getElementById('sourceFileChosen').style.display = '';
}

function clearWizardFile() {
  wizardUploadedPath     = null;
  wizardUploadedFilename = null;
  wizardUploadedDbType   = null;
  document.getElementById('wFileInput').value = '';
  document.getElementById('sourceFilePrompt').style.display = '';
  document.getElementById('sourceFileChosen').style.display = 'none';
}

function initSourceFileDrop() {
  const drop    = document.getElementById('sourceFileDrop');
  const input   = document.getElementById('wFileInput');
  const browse  = document.getElementById('sourceFileBrowse');
  const clear   = document.getElementById('sourceFileClear');

  // Restore state if a file was already chosen
  if (wizardUploadedFilename) {
    setWizardFileChosen(wizardUploadedFilename);
  } else {
    document.getElementById('sourceFilePrompt').style.display = '';
    document.getElementById('sourceFileChosen').style.display = 'none';
  }

  async function handleSourceFile(file) {
    if (!file) return;
    setWizardFileChosen(file.name);
    wizardNext.disabled = true;
    wizardNext.textContent = 'Uploading…';
    try {
      const info = await apiUploadSourceFile(file);
      wizardUploadedPath   = info.path;
      wizardUploadedDbType = info.db_type;
      showToast(`"${file.name}" uploaded`, 'success', 3000);
    } catch (err) {
      clearWizardFile();
      showToast(err.message || 'Upload failed', 'error');
    } finally {
      wizardNext.disabled = false;
      wizardNext.textContent = 'Next';
    }
  }

  drop.addEventListener('click', (e) => {
    if (e.target === clear || e.target.closest('.source-file-clear')) return;
    input.click();
  });
  browse.addEventListener('click', (e) => { e.stopPropagation(); input.click(); });
  clear.addEventListener('click',  (e) => { e.stopPropagation(); clearWizardFile(); });
  input.addEventListener('change', (e) => { handleSourceFile(e.target.files[0]); input.value = ''; });
  drop.addEventListener('dragover',  (e) => { e.preventDefault(); drop.classList.add('drag-over'); });
  drop.addEventListener('dragleave', ()  => { drop.classList.remove('drag-over'); });
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('drag-over');
    handleSourceFile(e.dataTransfer.files[0]);
  });
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
    document.querySelector('.test-connection-row').style.display = isFile ? 'none' : '';
    if (isFile) initSourceFileDrop();
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
    connRow = `<div class="confirm-row"><span class="confirm-key">File</span><span class="confirm-val">${escHtml(wizardUploadedFilename || '—')}</span></div>`;
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
      if (!wizardUploadedPath) { showToast('Please upload a file first', 'error'); return; }
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
    ? { file_path: wizardUploadedPath, uploaded: true }
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
    db_type:        isFile ? (wizardUploadedDbType || wizardDbType) : wizardDbType,
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
          <button class="admin-action-btn" data-action="index-log" data-id="${s.id}">Index Log</button>
          <button class="admin-action-btn" data-action="view-kg" data-id="${s.id}">View KG</button>
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
      if (action === 'index-log') {
        openIndexLog(id);
      } else if (action === 'view-kg') {
        openKGExplorer(id);
      } else if (action === 'reindex') {
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

// ── Index Log ─────────────────────────────────────────────────────────────────

let _indexLogSourceId = null;
let _indexLogSSE      = null;

const INDEX_STEP_LABELS = {
  extract:  'Metadata Extraction',
  ontology: 'Ontology Generation',
  kg:       'Knowledge Graph Build',
  complete: 'Complete',
};

function openIndexLog(sourceId) {
  _indexLogSourceId = sourceId;
  const src = sources[sourceId];
  document.getElementById('indexLogTitle').textContent =
    `Indexing Progress — ${src ? src.name : sourceId}`;
  document.getElementById('indexLogDetail').textContent = '';
  renderIndexLogSteps([]);
  document.getElementById('indexLogOverlay').style.display = 'flex';
  connectIndexSSE(sourceId);
}

function closeIndexLog() {
  document.getElementById('indexLogOverlay').style.display = 'none';
  if (_indexLogSSE) { _indexLogSSE.close(); _indexLogSSE = null; }
  _indexLogSourceId = null;
}

function connectIndexSSE(sourceId) {
  if (_indexLogSSE) { _indexLogSSE.close(); }
  const es = new EventSource(`${API}/sources/${sourceId}/index-events`);
  _indexLogSSE = es;

  const stepsMap = {};   // step → latest event

  es.onmessage = (e) => {
    let ev;
    try { ev = JSON.parse(e.data); } catch { return; }
    if (ev.type === 'heartbeat') return;
    if (ev.type === 'index_step') {
      stepsMap[ev.step] = ev;
      renderIndexLogSteps(Object.values(stepsMap));
    }
    if (ev.step === 'complete') {
      es.close();
      _indexLogSSE = null;
    }
  };
  es.onerror = () => {
    es.close();
    _indexLogSSE = null;
  };
}

function renderIndexLogSteps(events) {
  const container = document.getElementById('indexLogSteps');
  if (!events.length) {
    container.innerHTML = '<div style="padding:24px;text-align:center;color:var(--clr-text-mute);font-size:14px">No events yet — trigger reindex to start.</div>';
    return;
  }
  container.innerHTML = events.map(ev => {
    const label = INDEX_STEP_LABELS[ev.step] || ev.step;
    let iconHtml;
    if (ev.status === 'running') {
      iconHtml = `<div class="step-icon-running"></div>`;
    } else if (ev.status === 'done') {
      iconHtml = `<span style="color:var(--clr-success,#34A853);font-size:16px">✓</span>`;
    } else if (ev.status === 'error') {
      iconHtml = `<span style="color:var(--clr-error,#EA4335);font-size:16px">✗</span>`;
    } else {
      iconHtml = `<span style="color:var(--clr-warn,#FBBC04);font-size:14px">⚠</span>`;
    }
    return `
      <div class="index-log-step status-${escHtml(ev.status)}" data-detail="${escHtml(ev.detail || '')}">
        <div class="index-log-step-icon">${iconHtml}</div>
        <div class="index-log-step-body">
          <div class="index-log-step-name">${escHtml(label)}</div>
          <div class="index-log-step-msg">${escHtml(ev.message)}</div>
        </div>
      </div>`;
  }).join('');

  container.querySelectorAll('.index-log-step').forEach(row => {
    row.addEventListener('click', () => {
      document.getElementById('indexLogDetail').textContent = row.dataset.detail || '';
    });
  });
}

// ── KG Explorer ───────────────────────────────────────────────────────────────

let _kgExplorerSourceId  = null;
let _kgNetwork           = null;
let _kgVisNodes          = null;   // vis.DataSet — kept for highlight updates
let _kgVisEdges          = null;
let _kgOriginalColors    = {};     // node id → original color object
let _kgOntologyOriginal  = '';
let _kgPreviewDebounce   = null;

async function openKGExplorer(sourceId) {
  _kgExplorerSourceId = sourceId;
  const src = sources[sourceId];
  document.getElementById('kgExplorerTitle').textContent =
    `Knowledge Graph & Ontology — ${src ? src.name : sourceId}`;

  // Show save button for admins only
  const actionsEl = document.getElementById('kgExplorerHeaderActions');
  const isAdmin = PERSONAS[currentPersona]?.isAdmin;
  actionsEl.innerHTML = isAdmin
    ? `<button class="btn-primary" id="kgSaveBtn">Save Ontology &amp; Rebuild KG</button>`
    : '';
  if (isAdmin) {
    document.getElementById('kgSaveBtn').addEventListener('click', () => saveOntologyAndKG(sourceId));
  }

  document.getElementById('kgModifiedBadge').style.display = 'none';
  document.getElementById('kgOntologyHint').textContent = isAdmin
    ? 'Edit the ontology and click "Save Ontology & Rebuild KG" to persist changes.'
    : 'Editing the ontology updates the graph preview. Only admins can save changes.';

  document.getElementById('kgExplorerOverlay').style.display = 'flex';
  document.getElementById('kgGraphPlaceholder').style.display = 'flex';
  document.getElementById('kgGraphPlaceholder').textContent = 'Loading…';

  // Destroy any existing vis network
  if (_kgNetwork) { _kgNetwork.destroy(); _kgNetwork = null; }

  // Load graph + ontology in parallel
  const [graphData, ontologyData] = await Promise.all([
    apiGetSourceGraph(sourceId),
    apiGetSourceOntology(sourceId),
  ]);

  // Render graph
  renderKGGraph(graphData.nodes || [], graphData.edges || []);

  // Load ontology into editor
  _kgOntologyOriginal = ontologyData.content || '';
  const editor = document.getElementById('kgOntologyEditor');
  editor.value  = _kgOntologyOriginal;
  editor.disabled = false;
}

function closeKGExplorer() {
  document.getElementById('kgExplorerOverlay').style.display = 'none';
  if (_kgNetwork) { _kgNetwork.destroy(); _kgNetwork = null; }
  // Eagerly clean the container so vis.js residual DOM/styles don't affect the next open
  const _closeCtr = document.getElementById('kgGraphContainer');
  if (_closeCtr) { _closeCtr.innerHTML = ''; _closeCtr.style.cssText = ''; }
  if (_kgPreviewDebounce) { clearTimeout(_kgPreviewDebounce); _kgPreviewDebounce = null; }
  _kgVisNodes = null;
  _kgVisEdges = null;
  _kgOriginalColors = {};
  _kgExplorerSourceId = null;
  document.getElementById('kgGraphragLegend').style.display = 'none';
  document.getElementById('kgGraphragReset').style.display  = 'none';
  document.getElementById('kgGraphragInput').value          = '';
}

function renderKGGraph(nodes, edges) {
  const container = document.getElementById('kgGraphContainer');
  const placeholder = document.getElementById('kgGraphPlaceholder');

  if (!nodes.length) {
    placeholder.style.display = 'flex';
    placeholder.textContent = 'No graph data available. Index this source first.';
    return;
  }
  placeholder.style.display = 'none';

  // Destroy previous instance and eagerly clear the container
  if (_kgNetwork) { _kgNetwork.destroy(); _kgNetwork = null; }
  container.innerHTML = '';
  container.style.cssText = '';

  const isDark   = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const fontClr  = isDark ? '#e2e8f0' : '#1a202c';
  const edgeClr  = isDark ? '#718096' : '#a0aec0';
  const edgeFont = isDark ? '#a0aec0' : '#4a5568';

  const visNodes = new vis.DataSet(nodes.map(n => ({
    id:    n.id,
    label: n.label,
    title: n.title,
    color: { background: n.color || '#63b3ed', border: '#3182ce',
             highlight: { background: '#90cdf4', border: '#2b6cb0' } },
    size:  n.size || 20,
    font:  { color: fontClr, size: 13 },
  })));

  const visEdges = new vis.DataSet(edges.map((e, i) => ({
    id:     i,
    from:   e.from,
    to:     e.to,
    label:  e.label,
    title:  e.title,
    arrows: { to: { enabled: true, scaleFactor: 0.6 } },
    color:  { color: edgeClr, highlight: '#63b3ed' },
    font:   { color: edgeFont, size: 11, align: 'middle' },
    smooth: { type: 'dynamic' },
  })));

  const options = {
    physics: {
      stabilization: { iterations: 200, fit: true },
      barnesHut: { gravitationalConstant: -4000, springLength: 120 },
    },
    interaction: { tooltipDelay: 100, hover: true },
    nodes: { shape: 'ellipse', borderWidth: 1.5, widthConstraint: { maximum: 140 } },
    layout: { improvedLayout: true },
  };

  // Store DataSets for highlight updates without rebuilding the network
  _kgVisNodes = visNodes;
  _kgVisEdges = visEdges;

  // Capture original colors for reset
  _kgOriginalColors = {};
  nodes.forEach(n => { _kgOriginalColors[n.id] = null; });  // null = keep default

  // Capture source id so stale callbacks from a previous open are ignored
  const renderSourceId = _kgExplorerSourceId;

  // Try to initialise vis.js. If the flex container hasn't been sized yet,
  // retry every 50 ms (up to ~2 s) until it has real pixel dimensions.
  function _initNetwork(attempt) {
    if (attempt > 40) return;  // give up after ~2 s
    if (_kgExplorerSourceId !== renderSourceId) return;  // stale: modal closed or re-opened
    const overlay = document.getElementById('kgExplorerOverlay');
    if (!overlay || overlay.style.display === 'none') return;

    const w = container.offsetWidth;
    const h = container.offsetHeight;
    if (w < 10 || h < 10) {
      // Container not sized yet — wait another frame
      setTimeout(() => _initNetwork(attempt + 1), 50);
      return;
    }

    _kgNetwork = new vis.Network(container, { nodes: visNodes, edges: visEdges }, options);
    _kgNetwork.once('stabilizationIterationsDone', () => {
      _kgNetwork.fit({ animation: { duration: 300, easingFunction: 'easeInOutQuad' } });
    });
  }

  // Defer by two frames so the overlay's flex layout is computed before the first check
  requestAnimationFrame(() => requestAnimationFrame(() => _initNetwork(0)));
}

async function apiGraphRAGQuery(sourceId, query, topK = 8, hopDepth = 2) {
  const r = await fetch(`${API}/sources/${sourceId}/graphrag-query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK, hop_depth: hopDepth }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function testGraphRAG(sourceId) {
  const input = document.getElementById('kgGraphragInput');
  const btn   = document.getElementById('kgGraphragBtn');
  const query = input.value.trim();
  if (!query) { input.focus(); return; }

  btn.disabled = true;
  btn.textContent = '…';
  try {
    const result = await apiGraphRAGQuery(sourceId, query);
    applyGraphRAGHighlight(result);
  } catch (err) {
    showToast(`GraphRAG query failed: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Search';
  }
}

function applyGraphRAGHighlight(result) {
  if (!_kgVisNodes) return;

  const seedIds     = new Set(result.seed_nodes.map(n => n.id));
  const expandedIds = new Set(result.expanded_ids || []);
  const scoreMap    = Object.fromEntries(result.seed_nodes.map(n => [n.id, n.score]));

  const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const dimmed = isDark
    ? { background: '#2d3748', border: '#4a5568' }
    : { background: '#e2e8f0', border: '#cbd5e0' };

  const updates = _kgVisNodes.getIds().map(id => {
    if (seedIds.has(id)) {
      const score  = scoreMap[id] || 0;
      const pct    = Math.round(score * 100);
      const alpha  = 0.5 + score * 0.5;   // 0.5–1.0 opacity for background
      return {
        id,
        color: {
          background: `rgba(246,173,85,${alpha.toFixed(2)})`,  // gold
          border:     '#d97706',
          highlight:  { background: '#fbd38d', border: '#b45309' },
        },
        label:  _kgVisNodes.get(id).label + `\n(${pct}%)`,
        font:   { color: isDark ? '#1a202c' : '#1a202c', size: 13, bold: true },
      };
    }
    if (expandedIds.has(id)) {
      return {
        id,
        color: {
          background: '#68d391',    // green
          border:     '#276749',
          highlight:  { background: '#9ae6b4', border: '#276749' },
        },
        font: { color: '#1a202c', size: 13 },
        label: _kgVisNodes.get(id).label,
      };
    }
    // Dim everything else
    return {
      id,
      color: { background: dimmed.background, border: dimmed.border },
      font:  { color: isDark ? '#718096' : '#a0aec0', size: 12 },
      label: _kgVisNodes.get(id).label,
    };
  });

  _kgVisNodes.update(updates);

  // Show legend + backend info
  document.getElementById('kgGraphragLegend').style.display = 'flex';
  document.getElementById('kgGraphragReset').style.display  = 'inline-flex';
  document.getElementById('kgGraphragBackend').textContent  =
    `backend: ${result.backend} · ${result.retrieved_nodes}/${result.total_nodes} nodes retrieved`;

  // Focus the graph on seed nodes
  if (_kgNetwork && result.seed_nodes.length) {
    _kgNetwork.fit({
      nodes:     result.seed_nodes.map(n => n.id),
      animation: { duration: 400, easingFunction: 'easeInOutQuad' },
    });
  }
}

function resetGraphRAGHighlight() {
  if (!_kgVisNodes) return;

  const isDark  = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const fontClr = isDark ? '#e2e8f0' : '#1a202c';

  const updates = _kgVisNodes.getIds().map(id => ({
    id,
    color: { background: '#63b3ed', border: '#3182ce',
             highlight: { background: '#90cdf4', border: '#2b6cb0' } },
    font:  { color: fontClr, size: 13, bold: false },
    label: _kgVisNodes.get(id).label.replace(/\n\(\d+%\)$/, ''),
  }));
  _kgVisNodes.update(updates);

  document.getElementById('kgGraphragLegend').style.display = 'none';
  document.getElementById('kgGraphragReset').style.display  = 'none';
  document.getElementById('kgGraphragInput').value          = '';

  if (_kgNetwork) {
    _kgNetwork.fit({ animation: { duration: 300, easingFunction: 'easeInOutQuad' } });
  }
}

async function saveOntologyAndKG(sourceId) {
  const editor = document.getElementById('kgOntologyEditor');
  const content = editor.value;
  const btn = document.getElementById('kgSaveBtn');
  if (btn) { btn.disabled = true; btn.textContent = 'Saving…'; }
  try {
    await apiSaveOntology(sourceId, content, true);
    _kgOntologyOriginal = content;
    document.getElementById('kgModifiedBadge').style.display = 'none';
    showToast('Ontology saved. KG rebuild started in background.', 'info');
  } catch (err) {
    showToast(`Save failed: ${err.message}`, 'error');
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Save Ontology & Rebuild KG'; }
  }
}

function _scheduleKGPreview() {
  if (!_kgExplorerSourceId) return;
  if (_kgPreviewDebounce) clearTimeout(_kgPreviewDebounce);
  document.getElementById('kgModifiedBadge').style.display =
    document.getElementById('kgOntologyEditor').value !== _kgOntologyOriginal ? 'inline' : 'none';
  _kgPreviewDebounce = setTimeout(async () => {
    const sourceId = _kgExplorerSourceId;
    const text = document.getElementById('kgOntologyEditor').value;
    if (!text.trim() || !sourceId) return;
    document.getElementById('kgGraphPlaceholder').style.display = 'flex';
    document.getElementById('kgGraphPlaceholder').textContent = 'Generating preview…';
    try {
      const graph = await apiPreviewKG(sourceId, text);
      renderKGGraph(graph.nodes || [], graph.edges || []);
    } catch (err) {
      document.getElementById('kgGraphPlaceholder').style.display = 'flex';
      document.getElementById('kgGraphPlaceholder').textContent = `Preview failed: ${err.message}`;
    }
  }, 1800);
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

// Custom role "OK" button
document.getElementById('personaRoleOk')?.addEventListener('click', () => {
  const inp = document.getElementById('personaRoleInput');
  const val = inp?.value.trim();
  if (!val) { inp?.focus(); return; }
  currentAnalystRole = 'other:' + val;
  localStorage.setItem('datachat_analyst_role', currentAnalystRole);
  renderAnalystRoleSection();
  updateRoleBadge();
  personaDropdown.style.display = 'none';
  showToast(`Role set: ${val}`, 'info', 2000);
});
document.getElementById('personaRoleInput')?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') document.getElementById('personaRoleOk')?.click();
});

// Index Log modal
document.getElementById('indexLogClose').addEventListener('click', closeIndexLog);
document.getElementById('indexLogOverlay').addEventListener('click', (e) => {
  if (e.target === document.getElementById('indexLogOverlay')) closeIndexLog();
});

// KG Explorer modal
document.getElementById('kgExplorerClose').addEventListener('click', closeKGExplorer);
document.getElementById('kgExplorerOverlay').addEventListener('click', (e) => {
  if (e.target === document.getElementById('kgExplorerOverlay')) closeKGExplorer();
});
document.getElementById('kgOntologyEditor').addEventListener('input', _scheduleKGPreview);
document.getElementById('kgGraphragBtn').addEventListener('click', () => {
  if (_kgExplorerSourceId) testGraphRAG(_kgExplorerSourceId);
});
document.getElementById('kgGraphragInput').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && _kgExplorerSourceId) testGraphRAG(_kgExplorerSourceId);
});
document.getElementById('kgGraphragReset').addEventListener('click', resetGraphRAGHighlight);

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
uploadBtn.addEventListener('click', startAdHocUploadSession);

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
  activeSourceId  = null;
  activeSessionId = null;
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

// ── Bridge Manager ────────────────────────────────────────────────────────────

let _bridges        = [];   // cached bridge list
let _bridgeEditId   = null; // null = create mode, string = edit mode

async function apiBridgeList() {
  const r = await fetch(`${API}/kg-bridges`);
  if (!r.ok) return [];
  return r.json();
}

async function apiBridgeCreate(payload) {
  const r = await fetch(`${API}/kg-bridges`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function apiBridgeUpdate(id, payload) {
  const r = await fetch(`${API}/kg-bridges/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function apiBridgeDelete(id) {
  const r = await fetch(`${API}/kg-bridges/${id}`, { method: 'DELETE' });
  if (!r.ok) throw new Error(await r.text());
}

function openBridgeManager() {
  closeAdminPanel();
  bridgeManagerOverlay.style.display = 'flex';
  hideBridgeForm();
  loadAndRenderBridges();
}

function closeBridgeManager() {
  bridgeManagerOverlay.style.display = 'none';
}

async function loadAndRenderBridges() {
  bridgeTableBody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--clr-text-mute);padding:24px">Loading…</td></tr>';
  try {
    _bridges = await apiBridgeList();
  } catch (_) {
    _bridges = [];
  }
  renderBridgeTable();
}

function renderBridgeTable() {
  const filter = bridgeFilterSource.value;
  const list   = filter ? _bridges.filter(b => b.source === filter) : _bridges;

  if (list.length === 0) {
    bridgeTableBody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--clr-text-mute);padding:24px">No bridges found.</td></tr>';
    return;
  }

  bridgeTableBody.innerHTML = list.map(b => {
    const confPct = Math.round((b.confidence || 0) * 100);
    const fromLabel = `<strong>${escHtml(b.from_kg)}</strong> → ${escHtml(b.from_entity || '')}.${escHtml(b.from_column)}`;
    const toLabel   = `<strong>${escHtml(b.to_kg)}</strong> → ${escHtml(b.to_entity || '')}.${escHtml(b.to_column)}`;
    const canPromote = b.source === 'inferred';
    return `
    <tr data-bridge-id="${escHtml(b.id)}">
      <td>${fromLabel}</td>
      <td>${toLabel}</td>
      <td>${escHtml(b.join_type || 'FK')}</td>
      <td><span class="bridge-source-badge ${escHtml(b.source)}">${escHtml(b.source)}</span></td>
      <td>
        <div class="bridge-confidence-bar">
          <div class="bridge-confidence-track">
            <div class="bridge-confidence-fill" style="width:${confPct}%"></div>
          </div>
          <span class="bridge-confidence-label">${confPct}%</span>
        </div>
      </td>
      <td>
        <label class="bridge-toggle" title="${b.enabled ? 'Enabled — click to disable' : 'Disabled — click to enable'}">
          <input type="checkbox" class="bridge-enable-chk" data-id="${escHtml(b.id)}" ${b.enabled ? 'checked' : ''}>
          <span class="bridge-toggle-slider"></span>
        </label>
      </td>
      <td>
        <div class="admin-actions">
          <button class="admin-action-btn" data-action="edit" data-id="${escHtml(b.id)}">Edit</button>
          ${canPromote ? `<button class="admin-action-btn" data-action="promote" data-id="${escHtml(b.id)}">Declare</button>` : ''}
          <button class="admin-action-btn danger" data-action="delete" data-id="${escHtml(b.id)}">Delete</button>
        </div>
      </td>
    </tr>`;
  }).join('');

  // Toggle enable/disable
  bridgeTableBody.querySelectorAll('.bridge-enable-chk').forEach(chk => {
    chk.addEventListener('change', async () => {
      const id      = chk.dataset.id;
      const enabled = chk.checked;
      try {
        await apiBridgeUpdate(id, { enabled });
        const b = _bridges.find(x => x.id === id);
        if (b) b.enabled = enabled;
      } catch (err) {
        showToast(err.message, 'error');
        chk.checked = !enabled; // revert
      }
    });
  });

  // Row action buttons
  bridgeTableBody.querySelectorAll('[data-action]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id     = btn.dataset.id;
      const action = btn.dataset.action;
      const b      = _bridges.find(x => x.id === id);
      if (!b) return;

      if (action === 'edit') {
        showBridgeForm(b);
      } else if (action === 'promote') {
        try {
          await apiBridgeUpdate(id, { source: 'declared', confidence: 1.0 });
          showToast('Bridge promoted to declared', 'success');
          await loadAndRenderBridges();
        } catch (err) { showToast(err.message, 'error'); }
      } else if (action === 'delete') {
        if (!confirm('Delete this bridge? This cannot be undone.')) return;
        try {
          await apiBridgeDelete(id);
          showToast('Bridge deleted', 'info');
          await loadAndRenderBridges();
        } catch (err) { showToast(err.message, 'error'); }
      }
    });
  });
}

function showBridgeForm(bridge = null) {
  _bridgeEditId = bridge ? bridge.id : null;
  bridgeFormTitle.textContent = bridge ? 'Edit bridge' : 'Declare a new bridge';
  bfFromKg.value     = bridge ? (bridge.from_kg     || '') : '';
  bfFromEntity.value = bridge ? (bridge.from_entity || '') : '';
  bfFromCol.value    = bridge ? (bridge.from_column  || '') : '';
  bfToKg.value       = bridge ? (bridge.to_kg       || '') : '';
  bfToEntity.value   = bridge ? (bridge.to_entity   || '') : '';
  bfToCol.value      = bridge ? (bridge.to_column    || '') : '';
  bfJoinType.value   = bridge ? (bridge.join_type    || 'FK') : 'FK';
  bfNotes.value      = bridge ? (bridge.notes        || '') : '';
  bridgeFormPanel.style.display = 'block';
  bfFromKg.focus();
}

function hideBridgeForm() {
  bridgeFormPanel.style.display = 'none';
  _bridgeEditId = null;
}

async function saveBridgeForm() {
  const payload = {
    from_kg:     bfFromKg.value.trim(),
    from_entity: bfFromEntity.value.trim(),
    from_column: bfFromCol.value.trim(),
    to_kg:       bfToKg.value.trim(),
    to_entity:   bfToEntity.value.trim(),
    to_column:   bfToCol.value.trim(),
    join_type:   bfJoinType.value,
    notes:       bfNotes.value.trim(),
  };

  if (!payload.from_kg || !payload.from_column || !payload.to_kg || !payload.to_column) {
    showToast('From KG, From column, To KG, and To column are required.', 'error');
    return;
  }

  try {
    if (_bridgeEditId) {
      await apiBridgeUpdate(_bridgeEditId, payload);
      showToast('Bridge updated', 'success');
    } else {
      // Declared bridges always get confidence 1.0 and source=declared
      payload.source     = 'declared';
      payload.confidence = 1.0;
      payload.enabled    = true;
      await apiBridgeCreate(payload);
      showToast('Bridge declared', 'success');
    }
    hideBridgeForm();
    await loadAndRenderBridges();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// ── Metadata Catalog (Data Manager persona) ───────────────────────────────────

// Raw data cache
let _mdEntities  = [];   // full list from API
let _mdActiveId  = null; // currently expanded entity metadata_id
let _mdActiveTab = 'entities';

// API helpers
const apiMdListEntities  = (sourceId) =>
  fetch(`${API}/metadata/entities${sourceId ? `?source_id=${encodeURIComponent(sourceId)}` : ''}`).then(r => r.json());
const apiMdGetEntity     = (mid) =>
  fetch(`${API}/metadata/entities/${encodeURIComponent(mid)}`).then(r => r.json());
const apiMdPatchEntity   = (mid, body) =>
  fetch(`${API}/metadata/entities/${encodeURIComponent(mid)}`, {
    method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
  }).then(r => r.json());
const apiMdPatchAttr     = (attrId, body) =>
  fetch(`${API}/metadata/attributes/${encodeURIComponent(attrId)}`, {
    method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
  }).then(r => r.json());

async function apiMdListRedundancies(sourceId) {
  const qs = sourceId ? `?source_id=${encodeURIComponent(sourceId)}` : '';
  const r = await fetch(`${API}/metadata/redundancies${qs}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function apiMdListChanges(sourceId) {
  const qs = sourceId ? `?source_id=${encodeURIComponent(sourceId)}` : '';
  const r = await fetch(`${API}/metadata/changes${qs}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
async function apiMdListSources() {
  const r = await fetch(`${API}/metadata/sources`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
async function apiMdPatchSource(sourceId, body) {
  const r = await fetch(`${API}/metadata/sources/${encodeURIComponent(sourceId)}`, {
    method: 'PATCH', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function loadMdCatalog() {
  mdEntityBody.innerHTML = '<tr><td colspan="8" class="md-empty">Loading…</td></tr>';
  mdAttrPanel.style.display = 'none';
  _mdActiveId = null;
  try {
    const srcId = mdSourceFilter.value || '';
    const raw = await apiMdListEntities(srcId);
    if (!Array.isArray(raw)) {
      throw new Error(raw?.detail || raw?.message || JSON.stringify(raw));
    }
    _mdEntities = raw;
    // Populate source filter dropdown (first load)
    _populateMdSourceFilter();
    renderMdEntityTable();
  } catch (e) {
    mdEntityBody.innerHTML = `<tr><td colspan="8" class="md-empty">Error loading metadata: ${e.message}</td></tr>`;
  }
  // Refresh whichever tab is active
  if (_mdActiveTab === 'redundancies') loadMdRedundancies();
  if (_mdActiveTab === 'changes')      loadMdChanges();
}

function _populateMdSourceFilter() {
  const existing = new Set([...mdSourceFilter.options].map(o => o.value));
  _mdEntities.forEach(e => {
    if (!existing.has(e.source_id)) {
      const opt = document.createElement('option');
      opt.value = e.source_id;
      opt.textContent = e.source_name || e.source_id;
      mdSourceFilter.appendChild(opt);
      existing.add(e.source_id);
    }
  });
}

function renderMdEntityTable() {
  const query   = (mdSearch.value || '').toLowerCase().trim();
  const srcId   = mdSourceFilter.value || '';
  let rows = _mdEntities;
  if (srcId)   rows = rows.filter(e => e.source_id === srcId);
  if (query)   rows = rows.filter(e =>
    e.table_name.toLowerCase().includes(query) ||
    (e.schema_name || '').toLowerCase().includes(query) ||
    (e.description || '').toLowerCase().includes(query)
  );

  if (!rows.length) {
    mdEntityBody.innerHTML = '<tr><td colspan="8" class="md-empty">No entities found.</td></tr>';
    return;
  }

  mdEntityBody.innerHTML = rows.map(e => {
    const shortId    = e.metadata_id.slice(0, 8) + '…';
    const isActive   = e.metadata_id === _mdActiveId;
    const desc       = e.description || '';
    const golden     = e.is_golden_record;
    const rows_n     = e.row_count != null ? e.row_count.toLocaleString() : '—';
    const isDeleted  = e.deleted_from_source;
    const redundant  = e.redundancy_count > 0;
    const deletedBadge    = isDeleted  ? '<span class="md-deleted-badge">Deleted from source</span>' : '';
    const redundancyBadge = redundant  ? `<span class="md-redundancy-badge" title="${e.redundancy_count} overlapping entity pair(s)">⚠ Redundant</span>` : '';

    return `<tr class="md-entity-row${isActive ? ' md-row-selected' : ''}${isDeleted ? ' md-row-deleted' : ''}" data-mid="${e.metadata_id}">
      <td>
        <button class="md-expand-btn" data-mid="${e.metadata_id}" title="View attributes">
          ${isActive ? '▼' : '▶'}
        </button>
      </td>
      <td><span class="md-id-badge" title="${e.metadata_id}">${shortId}</span></td>
      <td>${_esc(e.source_name || e.source_id)}</td>
      <td>${_esc(e.schema_name || '')}</td>
      <td><strong>${_esc(e.table_name)}</strong>${deletedBadge}${redundancyBadge}</td>
      <td>
        <div class="md-desc-cell">
          <span class="md-desc-text${desc ? '' : ' empty'}" data-mid="${e.metadata_id}" title="${_esc(desc)}">${_esc(desc) || 'No description'}</span>
          <button class="md-edit-btn" data-edit-entity="${e.metadata_id}" title="Edit description">✏️</button>
        </div>
      </td>
      <td>${rows_n}</td>
      <td>
        <label class="md-golden-toggle" title="Mark as golden record">
          <input type="checkbox" class="md-golden-chk" data-mid="${e.metadata_id}" ${golden ? 'checked' : ''}>
          <span class="md-golden-pip">★</span>
          <span class="md-golden-label">${golden ? 'Golden' : ''}</span>
        </label>
      </td>
    </tr>`;
  }).join('');

  // Bind expand buttons
  mdEntityBody.querySelectorAll('.md-expand-btn').forEach(btn => {
    btn.addEventListener('click', () => toggleMdEntityRow(btn.dataset.mid));
  });

  // Bind entity description edit
  mdEntityBody.querySelectorAll('[data-edit-entity]').forEach(btn => {
    btn.addEventListener('click', (ev) => {
      ev.stopPropagation();
      startEntityDescEdit(btn.dataset.editEntity);
    });
  });

  // Bind golden record toggles for entities
  mdEntityBody.querySelectorAll('.md-golden-chk').forEach(chk => {
    chk.addEventListener('change', async () => {
      const mid = chk.dataset.mid;
      const val = chk.checked;
      // Optimistic update in cache
      const ent = _mdEntities.find(e => e.metadata_id === mid);
      if (ent) ent.is_golden_record = val;
      const label = chk.closest('.md-golden-toggle').querySelector('.md-golden-label');
      if (label) label.textContent = val ? 'Golden' : '';
      try {
        await apiMdPatchEntity(mid, { is_golden_record: val });
      } catch (_) {
        showToast('Failed to update golden record flag', 'error');
        if (ent) ent.is_golden_record = !val;
        chk.checked = !val;
      }
    });
  });
}

async function toggleMdEntityRow(mid) {
  if (_mdActiveId === mid) {
    _mdActiveId = null;
    mdAttrPanel.style.display = 'none';
    renderMdEntityTable();
    return;
  }
  _mdActiveId = mid;
  renderMdEntityTable();
  await loadMdAttributes(mid);
}

async function loadMdAttributes(mid) {
  const ent = _mdEntities.find(e => e.metadata_id === mid);
  mdAttrPanelTitle.textContent = ent
    ? `Attributes — ${ent.schema_name ? ent.schema_name + '.' : ''}${ent.table_name}`
    : 'Attributes';
  mdAttrPanel.style.display = '';
  mdAttrBody.innerHTML = '<tr><td colspan="9" class="md-empty">Loading…</td></tr>';

  try {
    const entity = await apiMdGetEntity(mid);
    renderMdAttributeTable(entity.attributes || []);
  } catch (e) {
    mdAttrBody.innerHTML = `<tr><td colspan="9" class="md-empty">Error: ${e.message}</td></tr>`;
  }
}

function renderMdAttributeTable(attrs) {
  if (!attrs.length) {
    mdAttrBody.innerHTML = '<tr><td colspan="9" class="md-empty">No attributes found.</td></tr>';
    return;
  }

  mdAttrBody.innerHTML = attrs.map(a => {
    const nullPct = (a.row_count && a.null_count != null)
      ? Math.round((a.null_count / a.row_count) * 100) : null;
    const nullBar = nullPct != null
      ? `<div class="md-null-bar">
          <div class="md-null-track"><div class="md-null-fill" style="width:${nullPct}%"></div></div>
          <span class="md-null-pct">${nullPct}%</span>
         </div>`
      : '—';
    const pkChip  = a.is_primary_key ? '<span class="md-chip md-chip-pk">PK</span>' : '<span class="md-chip md-chip-no">—</span>';
    const fkChip  = a.is_foreign_key ? '<span class="md-chip md-chip-fk">FK</span>' : '<span class="md-chip md-chip-no">—</span>';
    const desc    = a.description || '';
    const uniqueN = a.unique_count != null ? a.unique_count.toLocaleString() : '—';
    const golden  = a.is_golden_record;
    const isAttrDeleted  = a.deleted_from_source;
    const attrDeletedBadge = isAttrDeleted ? '<span class="md-deleted-badge">Deleted</span>' : '';

    return `<tr${isAttrDeleted ? ' class="md-row-deleted"' : ''}>
      <td><strong>${_esc(a.column_name)}</strong>${attrDeletedBadge}</td>
      <td><code style="font-size:12px">${_esc(a.data_type)}</code></td>
      <td>${_esc(a.domain || '—')}</td>
      <td>
        <div class="md-desc-cell">
          <span class="md-desc-text${desc ? '' : ' empty'}" title="${_esc(desc)}">${_esc(desc) || 'No description'}</span>
          <button class="md-edit-btn" data-edit-attr="${a.attr_id}" title="Edit description">✏️</button>
        </div>
      </td>
      <td>${pkChip}</td>
      <td>${fkChip}</td>
      <td>${uniqueN}</td>
      <td>${nullBar}</td>
      <td>
        <label class="md-golden-toggle" title="Mark as golden record">
          <input type="checkbox" class="md-attr-golden-chk" data-attr-id="${a.attr_id}" ${golden ? 'checked' : ''}>
          <span class="md-golden-pip">★</span>
          <span class="md-golden-label">${golden ? 'Golden' : ''}</span>
        </label>
      </td>
    </tr>`;
  }).join('');

  // Bind attribute description edit
  mdAttrBody.querySelectorAll('[data-edit-attr]').forEach(btn => {
    btn.addEventListener('click', (ev) => {
      ev.stopPropagation();
      startAttrDescEdit(btn.dataset.editAttr, btn);
    });
  });

  // Bind golden record toggles for attributes
  mdAttrBody.querySelectorAll('.md-attr-golden-chk').forEach(chk => {
    chk.addEventListener('change', async () => {
      const attrId = chk.dataset.attrId;
      const val    = chk.checked;
      const label  = chk.closest('.md-golden-toggle').querySelector('.md-golden-label');
      if (label) label.textContent = val ? 'Golden' : '';
      try {
        await apiMdPatchAttr(attrId, { is_golden_record: val });
      } catch (_) {
        showToast('Failed to update golden record flag', 'error');
        chk.checked = !val;
        if (label) label.textContent = !val ? 'Golden' : '';
      }
    });
  });
}

// ── Inline description editing ─────────────────────────────────────────────

function startEntityDescEdit(mid) {
  const ent  = _mdEntities.find(e => e.metadata_id === mid);
  const cell = mdEntityBody.querySelector(`.md-desc-text[data-mid="${mid}"]`);
  if (!cell) return;
  const descCell = cell.closest('.md-desc-cell');
  const current  = ent ? (ent.description || '') : '';
  descCell.innerHTML = `
    <input class="md-desc-input" value="${_esc(current)}" maxlength="500" />
    <button class="btn-primary" style="padding:3px 8px;font-size:12px" id="mdSaveDesc_${mid}">Save</button>
    <button class="btn-secondary" style="padding:3px 8px;font-size:12px" id="mdCancelDesc_${mid}">✕</button>`;
  const inp = descCell.querySelector('.md-desc-input');
  inp.focus();
  inp.select();

  descCell.querySelector(`#mdSaveDesc_${mid}`).addEventListener('click', async () => {
    const newDesc = inp.value.trim();
    if (ent) ent.description = newDesc;
    try {
      await apiMdPatchEntity(mid, { description: newDesc });
      showToast('Description saved', 'success', 1800);
    } catch (_) { showToast('Save failed', 'error'); }
    renderMdEntityTable();
  });

  descCell.querySelector(`#mdCancelDesc_${mid}`).addEventListener('click', () => {
    renderMdEntityTable();
  });

  inp.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') descCell.querySelector(`#mdSaveDesc_${mid}`).click();
    if (ev.key === 'Escape') descCell.querySelector(`#mdCancelDesc_${mid}`).click();
  });
}

function startAttrDescEdit(attrId, editBtn) {
  const td      = editBtn.closest('td');
  const descCell = td.querySelector('.md-desc-cell');
  const current  = td.querySelector('.md-desc-text')?.title || '';
  descCell.innerHTML = `
    <input class="md-desc-input" value="${_esc(current)}" maxlength="500" />
    <button class="btn-primary" style="padding:3px 8px;font-size:12px" id="mdSaveAttr_${attrId}">Save</button>
    <button class="btn-secondary" style="padding:3px 8px;font-size:12px" id="mdCancelAttr_${attrId}">✕</button>`;
  const inp = descCell.querySelector('.md-desc-input');
  inp.focus();
  inp.select();

  descCell.querySelector(`#mdSaveAttr_${attrId}`).addEventListener('click', async () => {
    const newDesc = inp.value.trim();
    try {
      await apiMdPatchAttr(attrId, { description: newDesc });
      showToast('Description saved', 'success', 1800);
    } catch (_) { showToast('Save failed', 'error'); }
    // Re-render attribute table from API to reflect saved value
    if (_mdActiveId) await loadMdAttributes(_mdActiveId);
  });

  descCell.querySelector(`#mdCancelAttr_${attrId}`).addEventListener('click', async () => {
    if (_mdActiveId) await loadMdAttributes(_mdActiveId);
  });

  inp.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') descCell.querySelector(`#mdSaveAttr_${attrId}`).click();
    if (ev.key === 'Escape') descCell.querySelector(`#mdCancelAttr_${attrId}`).click();
  });
}

// Tiny HTML-escape helper (reuse or define if not already present)
function _esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

async function loadMdRedundancies() {
  const srcId = mdSourceFilter ? mdSourceFilter.value : '';
  try {
    const rows = await apiMdListRedundancies(srcId || null);
    renderMdRedundancyTable(rows);
  } catch (e) {
    mdRedundancyEmpty.textContent = 'Error loading redundancies: ' + e.message;
    mdRedundancyEmpty.style.display = '';
    mdRedundancyTable.style.display = 'none';
  }
}

function renderMdRedundancyTable(rows) {
  if (!rows || !rows.length) {
    mdRedundancyEmpty.style.display = '';
    mdRedundancyTable.style.display = 'none';
    return;
  }
  mdRedundancyEmpty.style.display = 'none';
  mdRedundancyTable.style.display = '';
  mdRedundancyBody.innerHTML = rows.map(r => {
    const aLabel = (r.a_schema ? r.a_schema + '.' : '') + r.a_table + ` <small>(${_esc(r.a_source_name || r.a_source_id)})</small>`;
    const bLabel = (r.b_schema ? r.b_schema + '.' : '') + r.b_table + ` <small>(${_esc(r.b_source_name || r.b_source_id)})</small>`;
    const pct    = (r.overlap_pct * 100).toFixed(1) + '%';
    const shared = Array.isArray(r.shared_columns) ? r.shared_columns.slice(0, 8).join(', ') + (r.shared_columns.length > 8 ? '…' : '') : '';
    const when   = r.detected_at ? new Date(r.detected_at).toLocaleString() : '';
    return `<tr class="md-redundancy-row">
      <td>${aLabel}</td>
      <td>${bLabel}</td>
      <td><span class="md-overlap-pct high">${pct}</span></td>
      <td class="md-shared-cols">${_esc(shared)}</td>
      <td class="md-ts">${when}</td>
    </tr>`;
  }).join('');
}

async function loadMdChanges() {
  const srcId = mdSourceFilter ? mdSourceFilter.value : '';
  try {
    const rows = await apiMdListChanges(srcId || null);
    renderMdChangesTable(rows);
  } catch (e) {
    mdChangesEmpty.textContent = 'Error loading changes: ' + e.message;
    mdChangesEmpty.style.display = '';
    mdChangesTable.style.display = 'none';
  }
}

function renderMdChangesTable(rows) {
  if (!rows || !rows.length) {
    mdChangesEmpty.style.display = '';
    mdChangesTable.style.display = 'none';
    return;
  }
  mdChangesEmpty.style.display = 'none';
  mdChangesTable.style.display = '';
  mdChangesBody.innerHTML = rows.map(r => {
    const ICONS  = { added: '✚', deleted: '✕', restored: '↩', type_changed: '⟳' };
    const COLORS = { added: 'green', deleted: 'red', restored: 'blue', type_changed: 'orange' };
    const icon   = ICONS[r.change_type] || '•';
    const color  = COLORS[r.change_type] || '';
    const when   = r.detected_at ? new Date(r.detected_at).toLocaleString() : '';
    let detail   = '';
    if (r.change_type === 'type_changed' && r.changed_fields && r.changed_fields.data_type) {
      detail = `${r.changed_fields.data_type.old} → ${r.changed_fields.data_type.new}`;
    }
    return `<tr>
      <td class="md-ts">${when}</td>
      <td><span class="md-change-type ${color}">${icon} ${r.change_type}</span></td>
      <td class="md-change-label">${_esc(r.entity_label || r.entity_id)}</td>
      <td>${_esc(r.entity_type)}</td>
      <td class="md-change-detail">${_esc(detail)}</td>
    </tr>`;
  }).join('');
}

async function loadMdSources() {
  try {
    const rows = await apiMdListSources();
    renderMdSourcesTable(rows);
  } catch (e) {
    if (mdSourcesEmpty) { mdSourcesEmpty.textContent = 'Error: ' + e.message; mdSourcesEmpty.style.display = ''; }
    if (mdSourcesTable) mdSourcesTable.style.display = 'none';
  }
}

function renderMdSourcesTable(rows) {
  if (!rows || !rows.length) {
    if (mdSourcesEmpty) mdSourcesEmpty.style.display = '';
    if (mdSourcesTable) mdSourcesTable.style.display = 'none';
    return;
  }
  if (mdSourcesEmpty) mdSourcesEmpty.style.display = 'none';
  if (mdSourcesTable) mdSourcesTable.style.display = '';
  mdSourcesBody.innerHTML = rows.map(src => {
    const domainVal = _esc(src.domain || '');
    const domainDisplay = src.domain
      ? `<span class="md-domain-badge">${domainVal}</span>`
      : `<span class="md-domain-unset">— unset —</span>`;
    return `<tr data-source-id="${_esc(src.source_id)}">
      <td class="md-source-name">${_esc(src.source_name || src.source_id)}</td>
      <td class="md-domain-cell">
        ${domainDisplay}
        <button class="md-edit-btn md-domain-edit-btn" title="Edit domain">✎</button>
        <span class="md-domain-edit-form" style="display:none">
          <input class="md-domain-input" type="text" value="${domainVal}" placeholder="e.g. Sales &amp; CRM" />
          <button class="md-domain-save-btn">Save</button>
          <button class="md-domain-cancel-btn">✕</button>
        </span>
      </td>
      <td>${src.active_entity_count ?? 0}</td>
      <td>${src.redundancy_count ?? 0}</td>
      <td class="md-source-desc">${_esc(src.description || '')}</td>
    </tr>`;
  }).join('');

  // Attach inline-edit handlers
  mdSourcesBody.querySelectorAll('tr[data-source-id]').forEach(row => {
    const srcId      = row.dataset.sourceId;
    const editBtn    = row.querySelector('.md-domain-edit-btn');
    const editForm   = row.querySelector('.md-domain-edit-form');
    const domainCell = row.querySelector('.md-domain-cell');
    const input      = row.querySelector('.md-domain-input');
    const saveBtn    = row.querySelector('.md-domain-save-btn');
    const cancelBtn  = row.querySelector('.md-domain-cancel-btn');

    editBtn.addEventListener('click', () => {
      domainCell.querySelector('.md-domain-badge, .md-domain-unset').style.display = 'none';
      editBtn.style.display = 'none';
      editForm.style.display = 'inline-flex';
      input.focus();
    });
    cancelBtn.addEventListener('click', () => {
      editForm.style.display = 'none';
      domainCell.querySelector('.md-domain-badge, .md-domain-unset').style.display = '';
      editBtn.style.display = '';
    });
    saveBtn.addEventListener('click', async () => {
      saveBtn.disabled = true;
      try {
        await apiMdPatchSource(srcId, { domain: input.value.trim() });
        loadMdSources();
      } catch (e) {
        alert('Save failed: ' + e.message);
        saveBtn.disabled = false;
      }
    });
  });
}

// Event listeners for metadata catalog
dmCatalogBtn.addEventListener('click', loadMdCatalog);
mdRefreshBtn.addEventListener('click', loadMdCatalog);
mdSourceFilter.addEventListener('change', loadMdCatalog);
mdSearch.addEventListener('input', renderMdEntityTable);
mdAttrClose.addEventListener('click', () => {
  _mdActiveId = null;
  mdAttrPanel.style.display = 'none';
  renderMdEntityTable();
});

if (mdTabBar) {
  mdTabBar.addEventListener('click', e => {
    const btn = e.target.closest('.md-tab');
    if (!btn) return;
    _mdActiveTab = btn.dataset.tab;
    mdTabBar.querySelectorAll('.md-tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('mdTabEntities').style.display     = _mdActiveTab === 'entities'     ? '' : 'none';
    document.getElementById('mdTabRedundancies').style.display = _mdActiveTab === 'redundancies' ? '' : 'none';
    document.getElementById('mdTabChanges').style.display      = _mdActiveTab === 'changes'      ? '' : 'none';
    document.getElementById('mdTabSources').style.display      = _mdActiveTab === 'sources'      ? '' : 'none';
    if (_mdActiveTab === 'redundancies') loadMdRedundancies();
    if (_mdActiveTab === 'changes')      loadMdChanges();
    if (_mdActiveTab === 'sources')      loadMdSources();
  });
}

// Event listeners for bridge manager
document.getElementById('adminBridgesBtn').addEventListener('click', openBridgeManager);
bridgeManagerClose.addEventListener('click', closeBridgeManager);
bridgeManagerOverlay.addEventListener('click', (e) => { if (e.target === bridgeManagerOverlay) closeBridgeManager(); });
bridgeRefreshBtn.addEventListener('click', loadAndRenderBridges);
bridgeFilterSource.addEventListener('change', renderBridgeTable);
bridgeAddBtn.addEventListener('click', () => showBridgeForm(null));
bridgeFormSave.addEventListener('click', saveBridgeForm);
bridgeFormCancel.addEventListener('click', hideBridgeForm);

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
