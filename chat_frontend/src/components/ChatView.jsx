import { useEffect, useRef, useState } from 'react';
import { useApp, PERSONAS } from '../state.jsx';
import { sessionEvents, getMessages, sendChat, uploadToSession } from '../api.js';
import Message from './Message.jsx';

const SUGGESTIONS = [
  ['📊 List available tables', 'What tables are available and how many rows does each have?'],
  ['🔍 Preview data', 'Show me the top 10 rows from the first table'],
  ['👥 Headcount by department', 'What is the total headcount by department?'],
  ['💰 Revenue breakdown', 'Show me the revenue breakdown by service line with percentages'],
];

export default function ChatView() {
  const { activeSessionId, llmModel, setLlmModel, analystRole, persona, toast, refreshSessions, sources, activeSourceName } = useApp();
  const [messages, setMessages] = useState([]);
  const [typing, setTyping] = useState(false);
  const [pipeline, setPipeline] = useState(null); // {stage, message, pct}
  const [input, setInput] = useState('');
  const [files, setFiles] = useState([]);
  const [sending, setSending] = useState(false);
  const esRef = useRef(null);
  const areaRef = useRef(null);
  const fileRef = useRef(null);
  const showSQL = PERSONAS[persona]?.showSQL;

  const scrollDown = () => {
    requestAnimationFrame(() => {
      if (areaRef.current) areaRef.current.scrollTop = areaRef.current.scrollHeight;
    });
  };

  // (Re)connect SSE + load history when the session changes.
  useEffect(() => {
    if (esRef.current) { esRef.current.close(); esRef.current = null; }
    setMessages([]);
    setTyping(false);
    setPipeline(null);
    if (!activeSessionId) return undefined;

    // cancelled flag prevents a stale getMessages response (from a previous
    // session that was switched away before the fetch completed) from
    // overwriting state for the current session.
    let cancelled = false;
    getMessages(activeSessionId)
      .then((d) => {
        if (cancelled) return;
        const msgs = Array.isArray(d) ? d : d.messages || [];
        setMessages(msgs.map((m, i) => ({ ...m, id: m.id || `h-${i}` })));
        scrollDown();
      })
      .catch(() => {});

    const es = sessionEvents(activeSessionId);
    esRef.current = es;
    es.onmessage = (e) => {
      let ev;
      try { ev = JSON.parse(e.data); } catch { return; }
      switch (ev.type) {
        case 'heartbeat':
          break;
        case 'thinking':
          setTyping(true);
          scrollDown();
          break;
        case 'chat_response':
          setTyping(false);
          setMessages((m) => [...m, { role: 'assistant', id: ev.msg_id || `a-${Date.now()}`, content: ev.content, results: ev.results, sql: ev.sql, errors: ev.errors, sources: ev.sources, cache_hit: ev.cache_hit }]);
          setSending(false);
          scrollDown();
          break;
        case 'chat_error':
          setTyping(false);
          setMessages((m) => [...m, { role: 'assistant', id: ev.msg_id || `e-${Date.now()}`, error: ev.message }]);
          setSending(false);
          break;
        case 'progress':
          if (!ev.background) setPipeline({ stage: ev.stage, message: ev.message, pct: ev.pct || 0 });
          break;
        case 'ready':
          setPipeline(null);
          refreshSessions();
          toast(ev.message || 'Ready to chat', 'success');
          break;
        case 'error':
          setPipeline(null);
          toast(ev.message || 'Pipeline error', 'error');
          break;
        default:
          break;
      }
    };
    return () => { cancelled = true; es.close(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSessionId]);

  // Build effective analyst_role: persona role + datasource domain context
  const effectiveRole = (() => {
    const personaCfg = PERSONAS[persona];
    const personaRole = personaCfg?.role || '';
    const activeSource = sources.find(s => s.name === activeSourceName);
    const domain = activeSource?.domain ? `Data domain: ${activeSource.domain}.` : '';
    const sourceName = activeSourceName ? `Datasource: ${activeSourceName}.` : '';
    // Manual override takes full precedence
    if (analystRole) return [analystRole, domain, sourceName].filter(Boolean).join(' ');
    return [personaRole, domain, sourceName].filter(Boolean).join(' ');
  })();

  const send = async () => {
    const text = input.trim();
    if (!text || !activeSessionId || sending) return;
    setMessages((m) => [...m, { role: 'user', id: `u-${Date.now()}`, content: text }]);
    setInput('');
    setSending(true);
    setTyping(true);
    scrollDown();
    try {
      await sendChat(activeSessionId, { message: text, analyst_role: effectiveRole, llm_model: llmModel });
    } catch (e) {
      setTyping(false);
      setSending(false);
      setMessages((m) => [...m, { role: 'assistant', id: `e-${Date.now()}`, error: e.message }]);
    }
  };

  const onUpload = async (fileList) => {
    const arr = Array.from(fileList);
    if (!arr.length || !activeSessionId) {
      if (!activeSessionId) toast('Open a chat first, then upload files', 'warn');
      return;
    }
    setFiles(arr);
    setPipeline({ stage: 'uploading', message: `Uploading ${arr.length} file(s)…`, pct: 2 });
    try {
      const fd = new FormData();
      arr.forEach((f) => fd.append('files', f, f.name));
      await uploadToSession(activeSessionId, fd);
    } catch (e) {
      setPipeline(null);
      toast(`Upload failed: ${e.message}`, 'error');
    }
  };

  const onKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  };

  return (
    <div className="chat-view" id="chatView" style={{ display: 'flex' }}>
      <div className="chat-area" id="chatArea" ref={areaRef}>
        {messages.length === 0 && (
          <div className="welcome" id="welcome">
            <h1 className="welcome-title">How can I help with your data?</h1>
            <p className="welcome-sub">
              {activeSessionId ? 'Ask a question about your data below.' : 'Select a data source from the sidebar to start exploring your data.'}
            </p>
            <div className="suggestions">
              {SUGGESTIONS.map(([label, q]) => (
                <button key={label} className="suggestion-chip" onClick={() => setInput(q)}>{label}</button>
              ))}
            </div>
          </div>
        )}
        <div className="messages" id="messages">
          {messages.map((m) => <Message key={m.id} msg={m} showSQL={showSQL} />)}
          {typing && (
            <div className="msg-row assistant">
              <div className="msg-avatar">⬡</div>
              <div className="msg-bubble">
                <div className="typing-indicator"><span /><span /><span /></div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="input-area" id="inputArea">
        <div className="input-area-inner">
          {files.length > 0 && (
            <div className="file-chips">
              {files.map((f, i) => <span className="file-chip" key={i}>📄 {f.name}</span>)}
            </div>
          )}
          {pipeline && (
            <div className="pipeline-bar" style={{ display: 'block' }}>
              <div className="pipeline-bar-inner">
                <span className="spinner" style={{ marginRight: 8 }} />
                {pipeline.stage ? `${pipeline.stage}: ` : ''}{pipeline.message}
                <div style={{ height: 3, background: 'rgba(255,255,255,0.1)', borderRadius: 2, marginTop: 6 }}>
                  <div style={{ height: '100%', width: `${pipeline.pct || 0}%`, background: 'linear-gradient(90deg,#4285F4,#9B72CB)', borderRadius: 2, transition: 'width .3s' }} />
                </div>
              </div>
            </div>
          )}
          <div className="input-row">
            <button className="upload-btn" type="button" title="Upload CSV or Excel files" onClick={() => fileRef.current?.click()}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" />
              </svg>
            </button>
            <input type="file" ref={fileRef} hidden multiple accept=".csv,.xlsx,.xls,.xlsm" onChange={(e) => onUpload(e.target.files)} />
            <textarea
              className="msg-input"
              placeholder="Ask anything about your data…"
              rows="1"
              value={input}
              disabled={!activeSessionId || sending}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKey}
            />
            <select className="model-select" title="Synthesis model" value={llmModel} onChange={(e) => setLlmModel(e.target.value)}>
              <option value="claude-haiku-4-5">⚡ Haiku 4.5</option>
              <option value="claude-sonnet-4-6">✦ Sonnet 4.6</option>
              <option value="claude-opus-4-5">◆ Opus 4.5</option>
            </select>
            <button className="send-btn" disabled={!activeSessionId || sending || !input.trim()} title="Send" onClick={send}>
              <svg viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" /></svg>
            </button>
          </div>
          <div className="input-footer">DataChat can make mistakes. Verify important figures independently.</div>
        </div>
      </div>
    </div>
  );
}
