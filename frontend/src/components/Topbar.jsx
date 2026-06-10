import { useAppState } from '../state.jsx';
import { useAuth } from '../auth.jsx';
import { IconRefresh, IconPlus, IconLogout } from './Icons.jsx';

const TITLES = {
  pipeline: ['Pipeline Monitor', 'All sources'],
  graph: ['Graph Explorer', 'Knowledge graph visualisation'],
  catalog: ['Schema Catalog', 'Tables, columns & statistics'],
  ontology: ['Ontology Viewer', 'OWL / Turtle ontology'],
  redundancy: ['Redundancies', 'Cross-source schema overlap'],
};

export default function Topbar() {
  const { currentView, refreshSources, bumpRefresh, setAddSourceOpen } = useAppState();
  const { email, doLogout } = useAuth();
  const [title, sub] = TITLES[currentView] || ['', ''];

  return (
    <div id="topbar">
      <span className="page-title">{title}</span>
      <span className="page-sub">{sub}</span>
      <div className="topbar-actions">
        <button
          className="btn btn-secondary"
          onClick={() => {
            refreshSources();
            bumpRefresh();
          }}
        >
          <IconRefresh />
          Refresh
        </button>
        <button className="btn btn-primary" onClick={() => setAddSourceOpen(true)}>
          <IconPlus />
          Add Source
        </button>
        <span className="topbar-email" title={email}>
          {email}
        </span>
        <button className="btn btn-logout" title="Sign out" onClick={doLogout}>
          <IconLogout width="14" height="14" />
          Sign out
        </button>
      </div>
    </div>
  );
}
