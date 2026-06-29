import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { listSources, listSessions, createSession } from './api.js';

export const PERSONAS = {
  business_user: {
    name: 'Business User', icon: '\u{1F464}', showSQL: false, canConnect: false, screen: 'landing',
    role: 'Business User — provide clear, jargon-free summaries with key numbers highlighted. Avoid technical SQL details.',
  },
  analyst: {
    name: 'Business Analyst', icon: '\u{1F52C}', showSQL: true, canConnect: true, screen: 'landing',
    role: 'Business Analyst — focus on trends, variance analysis, and data-driven recommendations. Show SQL logic where relevant.',
  },
  admin: {
    name: 'Data Admin', icon: '⚙️', showSQL: true, canConnect: true, isAdmin: true, screen: 'landing',
    role: 'Data Admin — provide technical detail including SQL, row counts, schema notes, and data quality observations.',
  },
  data_manager: {
    name: 'Data Manager', icon: '\u{1F5C2}️', showSQL: false, canConnect: false, screen: 'landing', apiPersona: 'business_user',
    role: 'Data Manager — focus on metadata quality, data lineage, and governance. Highlight missing descriptions, untagged columns, and golden record candidates. Use business-friendly language.',
  },
  bi_manager: {
    name: 'BI Manager', icon: '\u{1F4CA}', showSQL: false, canConnect: false, screen: 'landing', apiPersona: 'business_user',
    role: 'BI Manager — focus on KPI definitions, metric consistency, and taxonomy alignment. Surface calculation logic, dimension hierarchies, and conflicts between business definitions.',
  },
  finance_analyst: {
    name: 'Finance Analyst', icon: '\u{1F4B9}', showSQL: true, canConnect: false, screen: 'landing', apiPersona: 'business_user',
    role: 'Finance Analyst — prioritise financial KPIs: revenue, margins, cost variance, EBITDA, P&L line items, cash flow ratios. Compute derived metrics (margin %, growth %, variance) wherever the schema supports it. Present results in a structured financial format.',
  },
  finance_director: {
    name: 'Finance Director', icon: '\u{1F3DB}️', showSQL: false, canConnect: false, screen: 'landing', apiPersona: 'business_user',
    role: 'Finance Director — deliver concise executive-level insights. Lead with the headline number, then 2-3 key drivers. Flag material risks or anomalies. Avoid raw data dumps; synthesise into strategic takeaways.',
  },
  risk_analyst: {
    name: 'Risk Analyst', icon: '⚠️', showSQL: true, canConnect: false, screen: 'landing', apiPersona: 'business_user',
    role: 'Risk Analyst — focus on credit risk, default rates, delinquency (DPD buckets), LTV ratios, concentration risk (HHI), loss severity, recovery rates, and SRT eligibility. Flag breaches of regulatory thresholds. Use risk-specific terminology (PD, LGD, EAD, ECL).',
  },
  portfolio_manager: {
    name: 'Portfolio Manager', icon: '\u{1F4C2}', showSQL: false, canConnect: false, screen: 'landing', apiPersona: 'business_user',
    role: 'Portfolio Manager — focus on pool-level exposure, tranche waterfall, geographic and sector concentration, outstanding balances vs limits, and portfolio composition. Highlight diversification metrics and any concentration alerts above threshold.',
  },
  credit_analyst: {
    name: 'Credit Analyst', icon: '\u{1F4B3}', showSQL: true, canConnect: false, screen: 'landing', apiPersona: 'business_user',
    role: 'Credit Analyst — focus on loan-level credit quality: origination LTV, credit scores, interest rates, obligor type, vehicle residual value differentials, and default prediction signals. Compute weighted averages and stratification breakdowns across key credit dimensions.',
  },
  hr_manager: {
    name: 'HR Manager', icon: '\u{1F9D1}\u{200D}\u{1F4BC}', showSQL: false, canConnect: false, screen: 'landing', apiPersona: 'business_user',
    role: 'HR Manager — focus on workforce analytics: headcount, attrition, tenure, gender and age diversity, leave patterns, recruitment pipeline, and performance distribution. Present findings in plain language with actionable HR insights.',
  },
  cfo: {
    name: 'CFO', icon: '\u{1F4B0}', showSQL: false, canConnect: false, screen: 'landing', apiPersona: 'business_user',
    role: 'CFO — deliver concise financial intelligence. Lead with P&L headline numbers, cash flow position, EBITDA, and budget vs actuals. Highlight cost overruns, revenue risks, and margin trends. Frame everything in terms of financial impact and strategic decision-making.',
  },
  ceo: {
    name: 'CEO', icon: '\u{1F3E2}', showSQL: false, canConnect: false, screen: 'landing', apiPersona: 'business_user',
    role: 'CEO — provide top-line executive summaries only. Lead with the single most important number or trend, then 2-3 strategic takeaways. No raw data, no SQL, no jargon. Focus on what is working, what is not, and what decision is needed.',
  },
  operations_manager: {
    name: 'Operations Manager', icon: '\u{2699}\u{FE0F}', showSQL: false, canConnect: false, screen: 'landing', apiPersona: 'business_user',
    role: 'Operations Manager — focus on operational efficiency: throughput, cycle times, SLA adherence, capacity utilisation, and bottlenecks. Surface process deviations and highlight areas where performance is below target. Recommend operational improvements where data supports it.',
  },
  sales_manager: {
    name: 'Sales Manager', icon: '\u{1F4C8}', showSQL: false, canConnect: false, screen: 'landing', apiPersona: 'business_user',
    role: 'Sales Manager — focus on pipeline health, win rates, revenue attainment vs target, deal velocity, and rep-level performance. Highlight top performers, stalled deals, and territory gaps. Present in a results-driven format with clear comparisons to targets.',
  },
  compliance_officer: {
    name: 'Compliance Officer', icon: '\u{1F4CB}', showSQL: false, canConnect: false, screen: 'landing', apiPersona: 'business_user',
    role: 'Compliance Officer — focus on regulatory adherence, policy breaches, audit trails, and risk exposure. Flag any data anomalies or threshold breaches that may indicate compliance issues. Use precise, formal language and cite specific metrics and thresholds where relevant.',
  },
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
      const apiPersona = PERSONAS[persona]?.apiPersona || persona;
      const list = await listSources(apiPersona);
      setSources(Array.isArray(list) ? list : []);
      return list;
    } catch { return []; }
  }, [persona]);

  const refreshSessions = useCallback(async () => {
    try {
      const apiPersona = PERSONAS[persona]?.apiPersona || persona;
      const list = await listSessions(apiPersona);
      setSessions(Array.isArray(list) ? list : []);
      return list;
    } catch { return []; }
  }, [persona]);

  useEffect(() => { refreshSources(); refreshSessions(); }, [refreshSources, refreshSessions]);

  const openSource = useCallback(async (source) => {
    try {
      const apiPersona = PERSONAS[persona]?.apiPersona || persona;
      const s = await createSession({ title: source.name, persona: apiPersona, source_id: source.id });
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
