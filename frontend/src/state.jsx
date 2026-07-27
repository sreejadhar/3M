// build-marker-1
import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { listSources } from './api/clients.js';

// Shared workbench state — current view, the sources list (polled every 10s like
// app.js), the globally-selected "Active Source", and a toast queue.
const AppStateContext = createContext(null);

let toastSeq = 0;

export function AppStateProvider({ children }) {
  const [currentView, setCurrentView] = useState('pipeline');
  const [sources, setSources] = useState([]);
  const [loadingSources, setLoadingSources] = useState(true);
  const [activeSourceId, setActiveSourceId] = useState('');
  const [toasts, setToasts] = useState([]);
  const [refreshTick, setRefreshTick] = useState(0);
  const [addSourceOpen, setAddSourceOpen] = useState(false);
  const autoSelected = useRef(false);

  const bumpRefresh = useCallback(() => setRefreshTick((t) => t + 1), []);

  // Several call sites trigger refreshSources() independently and can overlap
  // (a 10s poll, a debounced SSE refresh, a direct post-reindex refresh, button
  // actions) — on localhost's near-zero latency they virtually always resolve
  // in the order they were issued, but over a real network (e.g. after
  // deployment) an older, slower GET /sources response can resolve AFTER a
  // newer one and silently clobber it with stale data. That's what caused
  // freshly-populated fields (e.g. business/domain right after a reindex) to
  // flash correctly for a moment and then revert. Guard with a generation
  // counter so only the response to the most-recently-issued call is applied.
  const refreshGen = useRef(0);

  const refreshSources = useCallback(async () => {
    const gen = ++refreshGen.current;
    try {
      const list = await listSources();
      if (gen !== refreshGen.current) return [];   // superseded by a newer call
      const arr = Array.isArray(list) ? list : [];
      setSources(arr);
      // auto-select first ready source once (matches app.js init)
      if (!autoSelected.current && !activeSourceId) {
        const ready = arr.find((s) => s.status === 'ready') || arr[0];
        if (ready) {
          setActiveSourceId(ready.id);
          autoSelected.current = true;
        }
      }
      return arr;
    } catch {
      return [];
    } finally {
      if (gen === refreshGen.current) setLoadingSources(false);
    }
  }, [activeSourceId]);

  useEffect(() => {
    refreshSources();
    const id = setInterval(refreshSources, 10000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const toast = useCallback((message, type = 'info', duration = 3500) => {
    const id = ++toastSeq;
    setToasts((t) => [...t, { id, message, type }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), duration);
  }, []);

  const value = {
    currentView,
    setCurrentView,
    sources,
    loadingSources,
    refreshSources,
    activeSourceId,
    setActiveSourceId,
    toasts,
    toast,
    refreshTick,
    bumpRefresh,
    addSourceOpen,
    setAddSourceOpen,
  };
  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState() {
  const ctx = useContext(AppStateContext);
  if (!ctx) throw new Error('useAppState must be used within AppStateProvider');
  return ctx;
}
