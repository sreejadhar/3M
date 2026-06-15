import { useState } from 'react';
import { useAuth } from '../auth.jsx';

// Faithful replica of the tech_ui login overlay (light split-panel).
export default function Login() {
  const { doLogin } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [shake, setShake] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await doLogin(email.trim(), password);
    } catch (err) {
      setError(err.message);
      setShake(true);
      setTimeout(() => setShake(false), 400);
    } finally {
      setBusy(false);
    }
  };

  const inputStyle = {
    width: '100%', boxSizing: 'border-box', padding: '11px 12px 11px 42px',
    border: '1.5px solid #e5e7eb', borderRadius: 8, fontSize: '.95rem', outline: 'none',
    transition: 'border-color .2s', background: '#fff',
  };

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 9999, display: 'flex' }}>
      {/* brand panel */}
      <div style={{ flex: '0 0 58%', background: 'linear-gradient(135deg,#0d1b3e 0%,#1a3a6b 50%,#0d2b5a 100%)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: 48, position: 'relative', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', inset: 0, backgroundImage: 'repeating-linear-gradient(0deg,transparent,transparent 60px,rgba(255,255,255,.03) 60px,rgba(255,255,255,.03) 61px),repeating-linear-gradient(90deg,transparent,transparent 60px,rgba(255,255,255,.03) 60px,rgba(255,255,255,.03) 61px)' }} />
        <div style={{ position: 'relative', textAlign: 'center', maxWidth: 380 }}>
          <div style={{ width: 64, height: 64, background: 'linear-gradient(135deg,#4285F4,#9B72CB)', borderRadius: 14, display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 20px', fontSize: '1.6rem', fontWeight: 800, color: '#fff' }}>
            DN
          </div>
          <h1 style={{ color: '#fff', fontSize: '2rem', fontWeight: 700, margin: '0 0 12px' }}>DataNanite</h1>
          <p style={{ color: 'rgba(255,255,255,.65)', fontSize: '1rem', lineHeight: 1.6, margin: 0 }}>
            Engineer Workbench — Pipeline Monitor &amp; Data Tools
          </p>
          <div style={{ display: 'flex', gap: 8, justifyContent: 'center', marginTop: 32 }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#4285F4', opacity: 0.8 }} />
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#9B72CB', opacity: 0.8 }} />
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#D96570', opacity: 0.8 }} />
          </div>
        </div>
      </div>

      {/* form panel */}
      <div style={{ flex: 1, background: '#f8f9fb', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 48 }}>
        <div style={{ width: '100%', maxWidth: 380 }}>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 700, color: '#1a1a2e', margin: '0 0 8px' }}>Welcome back</h2>
          <p style={{ color: '#6b7280', margin: '0 0 32px', fontSize: '.95rem' }}>Sign in to access the Engineer Workbench</p>
          {error && (
            <div style={{ background: '#fef2f2', border: '1px solid #fca5a5', color: '#dc2626', padding: '12px 16px', borderRadius: 8, fontSize: '.9rem', marginBottom: 20 }}>
              {error}
            </div>
          )}
          <form onSubmit={submit} className={shake ? 'login-shake' : ''} autoComplete="on">
            <div style={{ marginBottom: 20 }}>
              <label style={{ display: 'block', fontSize: '.875rem', fontWeight: 500, color: '#374151', marginBottom: 6 }}>Email address</label>
              <div style={{ position: 'relative' }}>
                <svg style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', width: 18, height: 18, color: '#9ca3af' }} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
                  <polyline points="22,6 12,12 2,6" />
                </svg>
                <input type="email" autoComplete="email" placeholder="you@cognizant.com" value={email}
                  onChange={(e) => setEmail(e.target.value)} required style={inputStyle}
                  onFocus={(e) => (e.target.style.borderColor = '#4285F4')} onBlur={(e) => (e.target.style.borderColor = '#e5e7eb')} />
              </div>
            </div>
            <div style={{ marginBottom: 28 }}>
              <label style={{ display: 'block', fontSize: '.875rem', fontWeight: 500, color: '#374151', marginBottom: 6 }}>Password</label>
              <div style={{ position: 'relative' }}>
                <svg style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', width: 18, height: 18, color: '#9ca3af' }} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                  <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                </svg>
                <input type="password" autoComplete="current-password" placeholder="••••••••" value={password}
                  onChange={(e) => setPassword(e.target.value)} required style={inputStyle}
                  onFocus={(e) => (e.target.style.borderColor = '#4285F4')} onBlur={(e) => (e.target.style.borderColor = '#e5e7eb')} />
              </div>
            </div>
            <button type="submit" disabled={busy}
              style={{ width: '100%', padding: 12, background: 'linear-gradient(135deg,#4285F4,#9B72CB)', color: '#fff', border: 'none', borderRadius: 8, fontSize: '1rem', fontWeight: 600, cursor: 'pointer' }}>
              {busy ? 'Signing in…' : 'Sign in'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
