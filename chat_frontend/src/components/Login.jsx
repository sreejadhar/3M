import { useState } from 'react';
import { useAuth } from '../auth.jsx';

// Faithful replica of the chat_ui login overlay (light split-panel).
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

  return (
    <div id="loginOverlay" style={{ position: 'fixed', inset: 0, zIndex: 9999, display: 'flex' }}>
      <div style={{ flex: '0 0 58%', background: 'linear-gradient(135deg,#0d1b3e 0%,#1a3a6b 50%,#0d2b5a 100%)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: 48, position: 'relative', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', inset: 0, backgroundImage: 'repeating-linear-gradient(0deg,transparent,transparent 60px,rgba(255,255,255,.03) 60px,rgba(255,255,255,.03) 61px),repeating-linear-gradient(90deg,transparent,transparent 60px,rgba(255,255,255,.03) 60px,rgba(255,255,255,.03) 61px)' }} />
        <div style={{ position: 'relative', textAlign: 'center', maxWidth: 380 }}>
          <svg viewBox="0 0 80 80" style={{ width: 72, height: 72, marginBottom: 24 }}>
            <defs>
              <linearGradient id="lgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#4285F4" />
                <stop offset="50%" stopColor="#9B72CB" />
                <stop offset="100%" stopColor="#D96570" />
              </linearGradient>
            </defs>
            <polygon points="40,4 76,22 76,58 40,76 4,58 4,22" fill="url(#lgGrad)" opacity="0.9" />
            <polygon points="40,18 62,29 62,51 40,62 18,51 18,29" fill="none" stroke="white" strokeWidth="2" opacity="0.6" />
          </svg>
          <h1 style={{ color: '#fff', fontSize: '2rem', fontWeight: 700, margin: '0 0 12px' }}>DataChat</h1>
          <p style={{ color: 'rgba(255,255,255,.65)', fontSize: '1rem', lineHeight: 1.6, margin: 0 }}>
            AI-powered data exploration and insights platform
          </p>
          <div style={{ display: 'flex', gap: 8, justifyContent: 'center', marginTop: 32 }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#4285F4', opacity: 0.8 }} />
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#9B72CB', opacity: 0.8 }} />
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#D96570', opacity: 0.8 }} />
          </div>
        </div>
      </div>
      <div style={{ flex: 1, background: '#f8f9fb', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 48 }}>
        <div style={{ width: '100%', maxWidth: 380 }}>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 700, color: '#1a1a2e', margin: '0 0 8px' }}>Welcome back</h2>
          <p style={{ color: '#6b7280', margin: '0 0 32px', fontSize: '.95rem' }}>Sign in to your account to continue</p>
          {error && (
            <div style={{ background: '#fef2f2', border: '1px solid #fca5a5', color: '#dc2626', padding: '12px 16px', borderRadius: 8, fontSize: '.9rem', marginBottom: 20 }}>
              {error}
            </div>
          )}
          <form onSubmit={submit} className={shake ? 'login-shake' : ''} autoComplete="on">
            <div style={{ marginBottom: 20 }}>
              <label style={{ display: 'block', fontSize: '.875rem', fontWeight: 500, color: '#374151', marginBottom: 6 }}>Email address</label>
              <input
                type="email"
                autoComplete="email"
                placeholder="you@cognizant.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                style={{ width: '100%', boxSizing: 'border-box', padding: '11px 12px', border: '1.5px solid #e5e7eb', borderRadius: 8, fontSize: '.95rem', outline: 'none', background: '#fff' }}
                required
              />
            </div>
            <div style={{ marginBottom: 28 }}>
              <label style={{ display: 'block', fontSize: '.875rem', fontWeight: 500, color: '#374151', marginBottom: 6 }}>Password</label>
              <input
                type="password"
                autoComplete="current-password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                style={{ width: '100%', boxSizing: 'border-box', padding: '11px 12px', border: '1.5px solid #e5e7eb', borderRadius: 8, fontSize: '.95rem', outline: 'none', background: '#fff' }}
                required
              />
            </div>
            <button
              type="submit"
              disabled={busy}
              style={{ width: '100%', padding: 12, background: 'linear-gradient(135deg,#4285F4,#9B72CB)', color: '#fff', border: 'none', borderRadius: 8, fontSize: '1rem', fontWeight: 600, cursor: 'pointer' }}
            >
              {busy ? 'Signing in…' : 'Sign in'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
