# Metadata Agent — React UI

A React (Vite) port of the Streamlit app (`app.py`). Same design and
functionality, served as a static SPA that talks to the five FastAPI
microservices.

## Run (dev)

The backends must be running (agent-api 8000, ontology-api 8001, kg-api 8002,
dialog-api 8003, conformity-api 8004 — see the repo's `run_services_local.ps1`).

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

Vite proxies each service so the browser never hits CORS:

| Browser path  | Proxied to            | Service        |
| ------------- | --------------------- | -------------- |
| `/agent`      | http://localhost:8000 | agent-api      |
| `/ontology`   | http://localhost:8001 | ontology-api   |
| `/kg`         | http://localhost:8002 | kg-api         |
| `/dialog`     | http://localhost:8003 | dialog-api     |
| `/conformity` | http://localhost:8004 | conformity-api |
| `/auth`       | http://localhost:8005 | orchestrator (login/validate) |

## Login

The app is gated by a login screen. It calls `POST /auth/login` on the
orchestrator (same JWT scheme as the existing HTML UIs): the token is stored in
`sessionStorage.auth_token`, the email in `localStorage.auth_email`, validated
on load via `GET /auth/validate`, and injected as `Authorization: Bearer` on
every same-origin request. Sign out via the **Logout** button at the bottom of
the sidebar. Accounts are seeded by the orchestrator from `SEED_PASSWORD_*`
env vars (or created directly in `data/auth.db`).

Override the targets with env vars (`AGENT_API_URL`, `ONTOLOGY_API_URL`, …)
when starting Vite, or set the `VITE_*_API_URL` vars to absolute URLs for a
production build (`npm run build`).

## Structure

```
src/
  main.jsx            entry
  App.jsx             shell + view routing (state-based, like st.session_state.page)
  state.jsx           app-wide store mirroring Streamlit session_state
  theme.css           ported _inject_css() dark theme + config.toml palette
  constants.js        DB_META, pipeline node lists, nav items
  hooks.js            useHealth, usePolling, useAsync
  api/clients.js      the five API clients (faithful port of app.py)
  lib/utils.js        node-state mapping, highlight, download, formatting
  components/         common.jsx, Sidebar, PipelineNodes, GraphView, DataTable
  views/              Extract, History, Search, Ontology, KnowledgeGraph,
                      Dialog, Conformity
```

`GraphView` uses `vis-network` to replace the Streamlit pyvis graph.
