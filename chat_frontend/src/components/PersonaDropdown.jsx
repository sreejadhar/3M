import { useApp, PERSONAS } from '../state.jsx';

const OPTIONS = [
  ['business_user',      '👤',  'Business User',       'Explore data sources and ask questions'],
  ['analyst',            '🔬',  'Business Analyst',    'Schema browser, SQL visibility, connect databases'],
  ['admin',              '⚙️',  'Data Admin',          'Manage all sources, reindex, control access'],
  ['data_manager',       '🗂️',  'Data Manager',        'Review metadata, tag golden records, edit descriptions'],
  ['bi_manager',         '📊',  'BI Manager',          'Define KPIs, manage taxonomy, compile calculation logic'],
  ['hr_manager',         '🧑‍💼', 'HR Manager',          'Workforce analytics, attrition, diversity and recruitment'],
  ['cfo',                '💰',  'CFO',                 'P&L, cash flow, EBITDA and budget vs actuals'],
  ['ceo',                '🏢',  'CEO',                 'Top-line executive summaries and strategic takeaways'],
  ['operations_manager', '⚙️',  'Operations Manager',  'Throughput, SLA adherence, capacity and bottlenecks'],
  ['sales_manager',      '📈',  'Sales Manager',       'Pipeline, win rates, revenue attainment and rep performance'],
  ['compliance_officer', '📋',  'Compliance Officer',  'Regulatory adherence, audit trails and risk exposure'],
  ['finance_analyst',    '💹',  'Finance Analyst',     'Financial data deep-dives with SQL access'],
  ['finance_director',   '🏛️',  'Finance Director',    'Executive view of financial performance and KPIs'],
  ['risk_analyst',       '⚠️',  'Risk Analyst',        'Credit risk, delinquency and loss analysis with SQL'],
  ['portfolio_manager',  '📂',  'Portfolio Manager',   'Pool-level exposure, tranche and concentration views'],
  ['credit_analyst',     '💳',  'Credit Analyst',      'Loan-level credit quality, LTV and default analytics'],
];

export default function PersonaDropdown() {
  const { personaDropdownOpen, setPersonaDropdownOpen, persona, setPersona, refreshSources, refreshSessions } = useApp();
  if (!personaDropdownOpen) return null;
  const choose = (key) => {
    setPersona(key);
    setPersonaDropdownOpen(false);
    refreshSources();
    refreshSessions();
  };
  return (
    <>
      <div style={{ position: 'fixed', inset: 0, zIndex: 199 }} onClick={() => setPersonaDropdownOpen(false)} />
      <div className="persona-dropdown" id="personaDropdown" style={{ display: 'block' }}>
        {OPTIONS.map(([key, icon, name, desc]) => (
          <div
            key={key}
            className={`persona-option${persona === key ? ' active' : ''}`}
            onClick={() => choose(key)}
          >
            <span className="persona-opt-icon">{icon}</span>
            <div>
              <div className="persona-opt-name">{name}</div>
              <div className="persona-opt-desc">{desc}</div>
            </div>
          </div>
        ))}
      </div>
    </>
  );
}
