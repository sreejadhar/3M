import { useAppState } from '../state.jsx';
import {
  IconPipeline,
  IconGraph,
  IconCatalog,
  IconOntology,
  IconRedundancy,
  IconSql,
  IconCdc,
  IconDocuments,
  IconGlossary,
  IconKpi,
} from './Icons.jsx';

const OBSERVE = [
  ['pipeline', IconPipeline, 'Pipeline Monitor'],
  ['graph', IconGraph, 'Graph Explorer'],
];
const INSPECT = [
  ['catalog', IconCatalog, 'Schema Catalog'],
  ['ontology', IconOntology, 'Ontology Viewer'],
  ['redundancy', IconRedundancy, 'Redundancies'],
  ['glossary', IconGlossary, 'Business Glossary'],
  ['dataglossary', IconGlossary, 'Business Glossary (Discovery)'],
  ['kpi', IconKpi, 'KPI Registry'],
  ['documents', IconDocuments, 'Document Intelligence'],
];
const DEVELOP_COMING = [];

function NavItem({ id, Icon, label, badge }) {
  const { currentView, setCurrentView } = useAppState();
  return (
    <div
      className={`nav-item ${currentView === id ? 'active' : ''}`}
      onClick={() => setCurrentView(id)}
    >
      <Icon />
      {label}
      {badge != null && <span className="badge">{badge}</span>}
    </div>
  );
}

export default function Sidebar() {
  const { sources, activeSourceId, setActiveSourceId } = useAppState();

  return (
    <aside id="sidebar">
      <div className="sidebar-brand">
        <div className="logo">DN</div>
        <div>
          <div className="brand-text">DataNanite</div>
          <div className="brand-sub">Engineer Workbench</div>
        </div>
      </div>

      <div className="sidebar-section">
        <div className="sidebar-section-label">Observe</div>
        <NavItem id="pipeline" Icon={IconPipeline} label="Pipeline Monitor" badge={sources.length} />
        <NavItem id="graph" Icon={IconGraph} label="Graph Explorer" />
      </div>

      <div className="sidebar-section">
        <div className="sidebar-section-label">Inspect</div>
        {INSPECT.map(([id, Icon, label]) => (
          <NavItem key={id} id={id} Icon={Icon} label={label} />
        ))}
      </div>

      <div className="sidebar-section">
        <div className="sidebar-section-label">Develop</div>
        <NavItem id="sql" Icon={IconSql} label="SQL Console" />
        <NavItem id="cdc" Icon={IconCdc} label="Change Log" />
        {DEVELOP_COMING.map(([id, Icon, label]) => (
          <div key={id} className="nav-item coming-soon" title="Coming Soon">
            <Icon />
            {label}
            <span className="coming-soon-badge">Coming Soon</span>
          </div>
        ))}
      </div>

      <div className="sidebar-source-selector">
        <label>Active Source</label>
        <select value={activeSourceId} onChange={(e) => setActiveSourceId(e.target.value)}>
          <option value="">— select source —</option>
          {sources.map((s) => (
            <option key={s.id} value={s.id}>
              {s.name}
            </option>
          ))}
        </select>
      </div>
    </aside>
  );
}
