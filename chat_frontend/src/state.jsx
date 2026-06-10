import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { listSources, listSessions, createSession } from './api.js';

export const PERSONAS = {
  business_user: { name: 'Business User', icon: '👤', showSQL: false, canConnect: false, screen: 'landing' },
  analyst: { name: 'Business Analyst', icon: '🔬', showSQL: true, canConnect: true, screen: 'landing' },
  admin: { name: 'Data Admin', icon: '⚙️', showSQL: true, canConnect: true, isAdmin: true, screen: 'landing' },
  data_manager: { name: 'Data Manager', icon: '🗂️', screen: 'md' },
  bi_manager: { name: 'BI Manager', icon: '📊', screen: 'bim' },
};

const LS = {
  persona: 'datachat_persona',
  model: 'datachat_llm_model',
  role: 'datachat_analyst_role',
};

const Ctx = createContext(null);
let toastSeq = 0;

export function AppStateProvider({ children }) {
  const [persona, setPersonaState] = useState(localStorage.getItem(LS.persona) || 'business_user');
  const [llmModel, setLlmModelState] = useState(localStorage.getItem(LS.model) || 'claude-sonnet-4-6');
  const [analystRole, setAnalystRoleState] = useState(localStorage.getItem(LS.role) || '');
  const [sources, setSources] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [activeSourceName, setActiveSourceName] = useState('');
  const [screen, setScreen] = useState(PERSONAS[localStorage.getItem(LS.persona) || 'business_user']?.screen || 'landing');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [personaDropdownOpen, setPersonaDropdownOpen] = useState(false);
  const [wizardOpen, setWizardOpen] = useState(false);
  const [toasts, setToasts] = useState([]);
  const seeded = useRef(false);

  const setPersona = useCallback((p) => {
    setPersonaState(p);
    localStorage.setItem(LS.persona, p);
    setScreen(PERSONAS[p]?.screen || 'landing');
  }, []);
  const setLlmModel = useCallback((m) => { setLlmModelState(m); localStorage.setItem(LS.model, m); }, []);
  const setAnalystRole = useCallback((r) => { setAnalystRoleState(r); localStorage.setItem(LS.role, r); }, []);

  const toastRef = useRef(null);

  const refreshSources = useCallback(async () => {
    try {
      const list = await listSources(persona);
      setSources(Array.isArray(list) ? list : []);
      return list;
    } catch { return []; }
  }, [persona]);

  const refreshSessions = useCallback(async () => {
    try {
      const list = await listSessions(persona);
      setSessions(Array.isArray(list) ? list : []);
      return list;
    } catch { return []; }
  }, [persona]);

  useEffect(() => { refreshSources(); refreshSessions(); }, [refreshSources, refreshSessions]);

  // Open a chat session for a source (creates one), resume an existing session,
  // or start fresh — mirrors openSourceSession / resumeSession / newChat.
  const openSource = useCallback(async (source) => {
    try {
      const s = await createSession({ title: source.name, persona, source_id: source.id });
      setActiveSessionId(s.session_id);
      setActiveSourceName(source.name);
      setScreen('chat');
      refreshSessions();
    } catch (e) {
      toastRef.current && toastRef.current(`Could not open source: ${e.message}`, 'error');
    }
  }, [persona]); // eslint-disable-line

  const openSession = useCallback((session) => {
    setActiveSessionId(session.session_id);
    setActiveSourceName(session.title || '');
    setScreen('chat');
  }, []);

  const newChat = useCallback(() => {
    setActiveSessionId(null);
    setActiveSourceName('');
    setScreen(PERSONAS[persona]?.screen || 'landing');
  }, [persona]);

  const toast = useCallback((message, type = 'info', duration = 4000) => {
    const id = ++toastSeq;
    setToasts((t) => [...t, { id, message, type }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), duration);
  }, []);
  toastRef.current = toast;

  const value = {
    persona, setPersona, llmModel, setLlmModel, analystRole, setAnalystRole,
    sources, refreshSources, sessions, refreshSessions,
    activeSessionId, setActiveSessionId, activeSourceName, setActiveSourceName,
    screen, setScreen, sidebarOpen, setSidebarOpen,
    personaDropdownOpen, setPersonaDropdownOpen, wizardOpen, setWizardOpen,
    toasts, toast, seeded,
    openSource, openSession, newChat,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error('useApp must be used within AppStateProvider');
  return ctx;
}
