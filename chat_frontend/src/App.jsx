import { AppStateProvider, useApp } from './state.jsx';
import { AuthProvider, useAuth } from './auth.jsx';
import Login from './components/Login.jsx';
import Sidebar from './components/Sidebar.jsx';
import Topbar from './components/Topbar.jsx';
import Landing from './components/Landing.jsx';
import ChatView from './components/ChatView.jsx';
import PersonaDropdown from './components/PersonaDropdown.jsx';
import ToastContainer from './components/Toast.jsx';
import Wizard from './components/Wizard.jsx';

function Placeholder({ title, sub }) {
  return (
    <div className="placeholder-screen">
      <h2 style={{ margin: 0 }}>{title}</h2>
      <p style={{ margin: 0 }}>{sub}</p>
    </div>
  );
}

function Shell() {
  const { screen } = useApp();
  return (
    <>
      <Sidebar />
      <div className="main" id="main">
        <Topbar />
        {screen === 'landing' && <Landing />}
        {screen === 'chat' && <ChatView />}
        {screen === 'md' && <Placeholder title="Metadata Catalog" sub="Data Manager view — porting next." />}
        {screen === 'bim' && <Placeholder title="KPI Manager" sub="BI Manager view — porting next." />}
      </div>
      <PersonaDropdown />
      <Wizard />
      <ToastContainer />
    </>
  );
}

function Gate() {
  const { checking, authed } = useAuth();
  if (checking) return <div style={{ margin: 'auto' }}><span className="spinner" /></div>;
  if (!authed) return <Login />;
  return (
    <AppStateProvider>
      <Shell />
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
