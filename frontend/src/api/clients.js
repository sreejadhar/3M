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
    if (res.status === 401) {
      window.dispatchEvent(new CustomEvent('auth:unauthorized'));
    }
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

export const executeSourceSQL = (sourceId, sql, limit = 500, password = null) =>
  apiPost(`/sources/${sourceId}/execute-sql`, {
    sql,
    limit,
    ...(password != null ? { password } : {}),
  });

// SSE — EventSource cannot send Authorization headers; the proxy/middleware
// exempts /index-events from auth, so a bare connection is fine.
export function sourceEvents(sourceId) {
  return new EventSource(`${API}/sources/${sourceId}/index-events`);
}

// ── Sources ───────────────────────────────────────────────────────────────────
// persona=admin keeps the persona_access ABAC filter permissive; per-user
// ownership scoping (who indexed what) is enforced server-side from the JWT.
export const listSources = () => apiGet('/sources?persona=admin');
export const createSource = (payload) => apiPost('/sources', payload);
export const patchSource = (id, payload) => apiPatch(`/sources/${id}`, payload);
export const deleteSource = (id) => apiDelete(`/sources/${id}`);
export const reindexSource = (id) => apiPost(`/sources/${id}/reindex`);
export const testConnection = (payload) => apiPost('/sources/test-connection', payload);
export const getGraph = (id) => apiGet(`/sources/${id}/graph`);
export const getOntology = (id) => apiGet(`/sources/${id}/ontology`);
export const saveOntology = (id, content, rebuildKg) =>
  apiPost(`/sources/${id}/ontology`, { content, rebuild_kg: rebuildKg });
export const validateOntology = (id) => apiPost(`/sources/${id}/validate-ontology`, {});

// ── Metadata ──────────────────────────────────────────────────────────────────
export const listEntities = (sourceId) =>
  apiGet(`/metadata/entities${sourceId ? `?source_id=${encodeURIComponent(sourceId)}` : ''}`);
export const getEntity = (metadataId) => apiGet(`/metadata/entities/${metadataId}`);
export const listRedundancies = () => apiGet('/metadata/redundancies');
export const enrichTaxonomy = (id) => apiPost(`/metadata/sources/${id}/enrich-taxonomy`);
export const classifyPII = (id) => apiPost(`/metadata/sources/${id}/classify-pii`);
export const detectBusiness = (id) => apiPost(`/metadata/sources/${id}/detect-business`);
export const detectDomain = (id) => apiPost(`/metadata/sources/${id}/detect-domain`);

// ── Business Glossary — governed term registry (distinct from the KPI/finance
// glossary above: this one is discovered from schema, cross-source, on demand) ─
export const generateSourceGlossary = (sourceId) =>
  apiPost(`/metadata/sources/${sourceId}/generate-glossary`);
export const getEntityGlossary = (metadataId) => apiGet(`/metadata/entities/${metadataId}/glossary`);
export const listBizGlossaryTerms = (opts = {}) => {
  const qs = new URLSearchParams();
  if (opts.sourceId) qs.set('source_id', opts.sourceId);
  if (opts.status) qs.set('status', opts.status);
  if (opts.domain) qs.set('domain', opts.domain);
  const q = qs.toString();
  return apiGet(`/metadata/glossary-terms${q ? `?${q}` : ''}`);
};
export const getBizGlossaryTerm = (termId) => apiGet(`/metadata/glossary-terms/${termId}`);
export const updateBizGlossaryTerm = (termId, body) => apiPatch(`/metadata/glossary-terms/${termId}`, body);
export const approveBizGlossaryTerm = (termId) => apiPost(`/metadata/glossary-terms/${termId}/approve`);
export const rejectBizGlossaryTerm = (termId) => apiPost(`/metadata/glossary-terms/${termId}/reject`);

// ── Abbreviation Glossary — governed abbreviation<->full-form registry, same
// shape as the Business Glossary above (discovered from schema, per source,
// on demand) ─────────────────────────────────────────────────────────────────
export const generateSourceAbbrevGlossary = (sourceId) =>
  apiPost(`/metadata/sources/${sourceId}/generate-abbreviation-glossary`);
export const listAbbrevGlossarySources = () => apiGet('/metadata/abbreviation-glossary-sources');
export const createAbbrevGlossaryTerm = (body) => apiPost('/metadata/abbreviation-glossary-terms', body);
export const listAbbrevGlossaryTerms = (opts = {}) => {
  const qs = new URLSearchParams();
  if (opts.sourceId) qs.set('source_id', opts.sourceId);
  if (opts.status) qs.set('status', opts.status);
  if (opts.domain) qs.set('domain', opts.domain);
  const q = qs.toString();
  return apiGet(`/metadata/abbreviation-glossary-terms${q ? `?${q}` : ''}`);
};
export const getAbbrevGlossaryTerm = (termId) => apiGet(`/metadata/abbreviation-glossary-terms/${termId}`);
export const updateAbbrevGlossaryTerm = (termId, body) =>
  apiPatch(`/metadata/abbreviation-glossary-terms/${termId}`, body);
export const approveAbbrevGlossaryTerm = (termId) =>
  apiPost(`/metadata/abbreviation-glossary-terms/${termId}/approve`);
export const rejectAbbrevGlossaryTerm = (termId) =>
  apiPost(`/metadata/abbreviation-glossary-terms/${termId}/reject`);

// ── Business Ontology — SKOS+OWL graph generated per-source from that
// source's governed glossary above (distinct artifact from the per-source
// structural ontology, and only offered for sources with a generated glossary) ─
export const listBusinessOntologySources = () => apiGet('/business-ontology/sources');
export const generateBusinessOntology = (sourceId) => apiPost(`/business-ontology/${sourceId}/generate`);
export const getBusinessOntologyDraft = (sourceId) => apiGet(`/business-ontology/${sourceId}/draft`);
export const getBusinessOntologyTermKgLinks = (sourceId, termId) =>
  apiGet(`/business-ontology/${sourceId}/terms/${termId}/kg-links`);
export const saveBusinessOntologyDraftTtl = (sourceId, ttlContent) =>
  apiPut(`/business-ontology/${sourceId}/draft`, { ttl_content: ttlContent });
export const updateBusinessOntologyTerm = (sourceId, termId, body) =>
  apiPatch(`/business-ontology/${sourceId}/terms/${termId}`, body);
export const deleteBusinessOntologyTerm = (sourceId, termId) =>
  apiDelete(`/business-ontology/${sourceId}/terms/${termId}`);
export const addBusinessOntologyRelation = (sourceId, termId, relatedTermId, relationshipType) =>
  apiPost(`/business-ontology/${sourceId}/relations`, {
    term_id: termId, related_term_id: relatedTermId, relationship_type: relationshipType,
  });
export const saveBusinessOntologyVersion = (sourceId, label) =>
  apiPost(`/business-ontology/${sourceId}/versions`, { label });
export const listBusinessOntologyVersions = (sourceId) => apiGet(`/business-ontology/${sourceId}/versions`);
export const getBusinessOntologyVersion = (sourceId, versionId) =>
  apiGet(`/business-ontology/${sourceId}/versions/${versionId}`);
export const restoreBusinessOntologyVersion = (sourceId, versionId) =>
  apiPost(`/business-ontology/${sourceId}/versions/${versionId}/restore`);

// ── KG bridges ──────────────────────────────────────────────────────────────
export const listBridges = () => apiGet('/kg-bridges');

// ── Change Log (CDC) ─────────────────────────────────────────────────────────
export const listChanges = ({ sourceId = '', entityId = '', limit = 200 } = {}) => {
  const qs = new URLSearchParams();
  if (sourceId)  qs.set('source_id',  sourceId);
  if (entityId)  qs.set('entity_id',  entityId);
  if (limit !== 200) qs.set('limit', limit);
  const q = qs.toString();
  return apiGet(`/metadata/changes${q ? `?${q}` : ''}`);
};

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

// ── Document Intelligence (unstructured agent, port 8008) ─────────────────────
// All routes proxy through the orchestrator: /unstructured/* → unstructured-api/*
const _U = '/unstructured';
export const listConnectorTypes = () => apiGet(`${_U}/connectors`);
export const docListSources     = ()          => apiGet(`${_U}/sources`);
export const docCreateSource    = (body)      => apiPost(`${_U}/sources`, body);
export const docDeleteSource    = (id)        => apiDelete(`${_U}/sources/${id}`);
export const docStartIndex      = (id)        => apiPost(`${_U}/sources/${id}/index`);
export const docListJobs        = (id)        => apiGet(`${_U}/sources/${id}/jobs`);
export const docListAssets      = (id)        => apiGet(`${_U}/sources/${id}/assets`);
export const docGetAsset        = (assetId)   => apiGet(`${_U}/assets/${assetId}`);
export const fsBrowse           = (path = '') => apiGet(`${_U}/fs/browse?path=${encodeURIComponent(path)}`);
export const docUploadDocument  = (sourceId, file) => {
  const fd = new FormData();
  fd.append('file', file, file.name);
  return apiUpload(`${_U}/sources/${sourceId}/upload`, fd);
};
