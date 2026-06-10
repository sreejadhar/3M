// DataChat API base.
//  • Dev (Vite, :5174): VITE_API_BASE unset → '/api', proxied to the orchestrator.
//  • Prod (served by the orchestrator at :8005): VITE_API_BASE='' → root paths
//    (/sources, /sessions, …) hit the orchestrator routes directly.
// Auth Bearer is injected by the fetch patch in auth.jsx.
const API = import.meta.env.VITE_API_BASE ?? '/api';

async function req(path, opts = {}) {
  const res = await fetch(`${API}${path}`, {
    ...opts,
    headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail || j.error || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  if (res.status === 204) return null;
  const ct = res.headers.get('content-type') || '';
  return ct.includes('application/json') ? res.json() : res.text();
}

export const get = (p) => req(p);
export const post = (p, b) => req(p, { method: 'POST', body: b == null ? undefined : JSON.stringify(b) });
export const patch = (p, b) => req(p, { method: 'PATCH', body: JSON.stringify(b) });
export const del = (p) => req(p, { method: 'DELETE' });

export async function upload(path, formData) {
  const res = await fetch(`${API}${path}`, { method: 'POST', body: formData });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* */ }
    throw new Error(detail);
  }
  return res.json();
}

export const sessionEvents = (id) => new EventSource(`${API}/sessions/${id}/events`);

// ── Sources ───────────────────────────────────────────────────────────────────
export const listSources = (persona) => get(`/sources${persona ? `?persona=${encodeURIComponent(persona)}` : ''}`);
export const getSource = (id) => get(`/sources/${id}`);
export const createSource = (payload) => post('/sources', payload);
export const deleteSource = (id) => del(`/sources/${id}`);
export const reindexSource = (id) => post(`/sources/${id}/reindex`);
export const testConnection = (payload) => post('/sources/test-connection', payload);
export const uploadSourceFile = (fd) => upload('/sources/upload-file', fd);
export const getSourceGraph = (id) => get(`/sources/${id}/graph`);

// ── Sessions ──────────────────────────────────────────────────────────────────
export const createSession = (payload) => post('/sessions', payload);
export const listSessions = (persona) => get(`/sessions${persona ? `?persona=${encodeURIComponent(persona)}` : ''}`);
export const deleteSession = (id) => del(`/sessions/${id}`);
export const uploadToSession = (id, fd) => upload(`/sessions/${id}/upload`, fd);
export const sendChat = (id, body) => post(`/sessions/${id}/chat`, body);
export const getMessages = (id) => get(`/sessions/${id}/messages`);

// ── Misc ──────────────────────────────────────────────────────────────────────
export async function exportExcel(payload) {
  const res = await fetch(`${API}/export-excel`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error('Export failed');
  return res.blob();
}
