// Formatting helpers ported from tech_ui/app.js.

// Convert a value that may be Unix seconds into milliseconds.
export function toMs(val) {
  if (val == null) return null;
  const n = typeof val === 'number' ? val : Date.parse(val);
  if (Number.isNaN(n)) return null;
  // seconds vs ms heuristic (year 3000 ≈ 32503680000 s)
  return n < 32503680000 ? n * 1000 : n;
}

export function fmtTime(val) {
  const ms = toMs(val);
  return ms ? new Date(ms).toLocaleString() : '—';
}

export function fmtRelTime(val) {
  const ms = toMs(val);
  if (!ms) return '—';
  const s = Math.max(0, Math.floor((Date.now() - ms) / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

export function fmtNum(n) {
  return typeof n === 'number' ? n.toLocaleString() : n == null ? '—' : n;
}

// Map source status → status-dot class.
export function statusDotClass(status) {
  if (status === 'ready' || status === 'indexed') return 'indexed';
  if (status === 'indexing') return 'indexing';
  if (status === 'error') return 'error';
  return 'pending';
}

// db_type → emoji icon (matches the workbench source cards).
const DB_ICONS = {
  postgres: '🐘',
  redshift: '🔺',
  mysql: '🐬',
  snowflake: '❄️',
  bigquery: '📈',
  sqlite: '🗄️',
  csv: '📄',
  excel: '📊',
  oracle: '🏛️',
  sqlserver: '🪟',
  teradata: '📊',
};
export const dbIcon = (t) => DB_ICONS[t] || '🗄️';
