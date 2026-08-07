import { AppStateProvider, useAppState } from './state.jsx';
import { AuthProvider, useAuth } from './auth.jsx';
import Login from './components/Login.jsx';
import Sidebar from './components/Sidebar.jsx';
import Topbar from './components/Topbar.jsx';
import ToastContainer from './components/Toast.jsx';
import AddSourceModal from './components/AddSourceModal.jsx';
import PipelineMonitor from './views/PipelineMonitor.jsx';
import GraphExplorer from './views/GraphExplorer.jsx';
import SchemaCatalog from './views/SchemaCatalog.jsx';
import OntologyViewer from './views/OntologyViewer.jsx';
import Redundancies from './views/Redundancies.jsx';
import Glossary from './views/Glossary.jsx';
import DataGlossary from './views/DataGlossary.jsx';
import Kpi from './views/Kpi.jsx';
import SqlConsole from './views/SqlConsole.jsx';
import ChangeLog from './views/ChangeLog.jsx';
import DocumentIntelligence from './views/DocumentIntelligence.jsx';

const VIEWS = {
  pipeline: PipelineMonitor,
  graph: GraphExplorer,
  catalog: SchemaCatalog,
  ontology: OntologyViewer,
  redundancy: Redundancies,
  glossary: Glossary,
  dataglossary: DataGlossary,
  kpi: Kpi,
  sql: SqlConsole,
  cdc: ChangeLog,
  documents: DocumentIntelligence,
};

function Workbench() {
  const { currentView, addSourceOpen } = useAppState();
  const View = VIEWS[currentView] || PipelineMonitor;
  return (
    <div id="app">
      <Sidebar />
      <div id="main">
        <Topbar />
        <div id="content">
          <View />
        </div>
      </div>
      {addSourceOpen && <AddSourceModal />}
      <ToastContainer />
    </div>
  );
}

function Gate() {
  const { checking, authed } = useAuth();
  if (checking) {
    return (
      <div id="login-overlay">
        <span className="spinner" />
      </div>
    );
  }
  if (!authed) return <Login />;
  return (
    <AppStateProvider>
      <Workbench />
    </AppStateProvider>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Gate />
    </AuthProvider>
  );
}
