// Workbench API — all calls go to /api/* which Vite proxies to the orchestrator
// (chat-ui, 8005) with the /api layer stripped, exactly like tech_ui_server.py.
// The auth Bearer token is injected by the global fetch patch in auth.jsx.
const API = '/api';

async function apiFetch(path, opts = {}) {
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

export const apiGet = (path) => apiFetch(path);
export const apiPost = (path, body) =>
  apiFetch(path, { method: 'POST', body: body == null ? undefined : JSON.stringify(body) });
export const apiPut = (path, body) =>
  apiFetch(path, { method: 'PUT', body: JSON.stringify(body) });
export const apiPatch = (path, body) =>
  apiFetch(path, { method: 'PATCH', body: JSON.stringify(body) });
export const apiDelete = (path) => apiFetch(path, { method: 'DELETE' });

// Multipart upload (no JSON content-type; browser sets the boundary).
export async function apiUpload(path, formData) {
  const res = await fetch(`${API}${path}`, { method: 'POST', body: formData });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

// SSE — EventSource cannot send Authorization headers; the proxy/middleware
// exempts /index-events from auth, so a bare connection is fine.
export function sourceEvents(sourceId) {
  return new EventSource(`${API}/sources/${sourceId}/index-events`);
}

// ── Sources ───────────────────────────────────────────────────────────────────
export const listSources = () => apiGet('/sources');
export const createSource = (payload) => apiPost('/sources', payload);
export const patchSource = (id, payload) => apiPatch(`/sources/${id}`, payload);
export const reindexSource = (id) => apiPost(`/sources/${id}/reindex`);
export const testConnection = (payload) => apiPost('/sources/test-connection', payload);
export const getGraph = (id) => apiGet(`/sources/${id}/graph`);
export const getOntology = (id) => apiGet(`/sources/${id}/ontology`);
export const saveOntology = (id, content, rebuildKg) =>
  apiPost(`/sources/${id}/ontology`, { content, rebuild_kg: rebuildKg });
export const validateOntology = (id) => apiPost(`/sources/${id}/ontology/validate`);

// ── Metadata ──────────────────────────────────────────────────────────────────
export const listEntities = (sourceId) =>
  apiGet(`/metadata/entities${sourceId ? `?source_id=${encodeURIComponent(sourceId)}` : ''}`);
export const getEntity = (metadataId) => apiGet(`/metadata/entities/${metadataId}`);
export const listRedundancies = () => apiGet('/metadata/redundancies');
export const enrichTaxonomy = (id) => apiPost(`/metadata/sources/${id}/enrich-taxonomy`);
export const classifyPII = (id) => apiPost(`/metadata/sources/${id}/classify-pii`);

// ── KG bridges ──────────────────────────────────────────────────────────────
export const listBridges = () => apiGet('/kg-bridges');

// ── Business Glossary ─────────────────────────────────────────────────────────
// All routes proxy through the orchestrator: /metadata/glossary/* → agent-api /glossary/*.
const _G = '/metadata/glossary';
export const listGlossaryTerms = (domain = '') =>
  apiGet(`${_G}/terms${domain ? `?domain=${encodeURIComponent(domain)}` : ''}`);
export const searchGlossaryTerms = (q, domain = '') =>
  apiGet(`${_G}/search?q=${encodeURIComponent(q)}&domain=${encodeURIComponent(domain)}&limit=50`);
export const getGlossaryTerm = (id) => apiGet(`${_G}/terms/${id}`);
export const createGlossaryTerm = (body) => apiPost(`${_G}/terms`, body);
export const updateGlossaryTerm = (id, body) => apiPut(`${_G}/terms/${id}`, body);
export const deleteGlossaryTerm = (id) => apiDelete(`${_G}/terms/${id}`);
export const addGlossarySynonym = (termId, synonym, domainScope = '') =>
  apiPost(`${_G}/terms/${termId}/synonyms`, { synonym, domain_scope: domainScope });
export const removeGlossarySynonym = (synonymId) => apiDelete(`${_G}/synonyms/${synonymId}`);
export const upsertGlossaryThreshold = (termId, body) => apiPut(`${_G}/terms/${termId}/threshold`, body);

// ── KPI Formula Registry ──────────────────────────────────────────────────────
// Served directly by the orchestrator (/kpis/*), backed by kpi_store.
export const listKpis = ({ source_id = '', category = '', status = '' } = {}) => {
  const qs = new URLSearchParams();
  if (source_id) qs.set('source_id', source_id);
  if (category) qs.set('category', category);
  if (status) qs.set('status', status);
  const q = qs.toString();
  return apiGet(`/kpis${q ? `?${q}` : ''}`);
};
export const getKpi = (id) => apiGet(`/kpis/${id}`);
export const createKpi = (body) => apiPost('/kpis', body);          // → { kpi, warnings }
export const updateKpi = (id, body) => apiPatch(`/kpis/${id}`, body); // → { kpi, warnings }
export const deleteKpi = (id) => apiDelete(`/kpis/${id}`);
export const compileKpi = (id, columnContext, model) =>
  apiPost(`/kpis/${id}/compile`, { column_context: columnContext, model });
export const listKpiVersions = (id) => apiGet(`/kpis/${id}/versions`);
export const rollbackKpiVersion = (id, versionNum) =>
  apiPost(`/kpis/${id}/versions/${versionNum}/rollback`, {});
export const activateKpi = (id) => apiPost(`/kpis/${id}/activate`, {});
