import { useEffect, useRef, useState } from 'react';
import { Network } from 'vis-network/standalone/esm/vis-network';
import { useAppState } from '../state.jsx';
import { getGraph, listBridges } from '../api/clients.js';
import { IconRefresh, IconGraph } from '../components/Icons.jsx';

// Classify a node by its label so we can colour/shape it meaningfully.
function nodeKind(label = '') {
  const s = String(label).toUpperCase();
  if (/FACT/.test(s)) return 'fact';
  if (/DIM/.test(s)) return 'dim';
  if (/KPI|SUMMARY|METRIC/.test(s)) return 'kpi';
  return 'other';
}

const KIND_STYLE = {
  fact: { background: '#3b2a0e', border: '#f59e0b', glow: 'rgba(245,158,11,0.25)' },
  dim: { background: '#11233e', border: '#58a6ff', glow: 'rgba(88,166,255,0.20)' },
  kpi: { background: '#2a163a', border: '#bc8cff', glow: 'rgba(188,140,255,0.20)' },
  other: { background: '#15202b', border: '#39c5cf', glow: 'rgba(57,197,207,0.18)' },
};

const LEGEND = [
  ['Fact table', '#f59e0b'],
  ['Dimension', '#58a6ff'],
  ['KPI / summary', '#bc8cff'],
  ['Other', '#39c5cf'],
];

// Edge cardinality → colour. Tokens like "(1:N)", "N:1", "M:N" appear in labels.
const CARD_STYLE = {
  '1:1': { color: '#3fb950', width: 1.4 },
  '1:N': { color: '#58a6ff', width: 1.4 },
  'N:1': { color: '#39c5cf', width: 1.4 },
  'N:N': { color: '#f59e0b', width: 2.4 }, // many-to-many — emphasised
};
const CARD_DEFAULT = { color: 'rgba(120,140,170,0.45)', width: 1.2 };
const CARD_LEGEND = [['1:1', '#3fb950'], ['1:N', '#58a6ff'], ['N:1', '#39c5cf'], ['N:N', '#f59e0b']];

function cardinalityOf(edge) {
  const txt = `${edge.label || ''} ${edge.title || ''}`;
  const m = txt.match(/([1nm])\s*:\s*([1nm])/i);
  if (!m) return null;
  const norm = (c) => (c === '1' ? '1' : 'N'); // M and N both → N
  return `${norm(m[1].toLowerCase())}:${norm(m[2].toLowerCase())}`;
}

export default function GraphExplorer() {
  const { sources, activeSourceId, setActiveSourceId, refreshTick, toast } = useAppState();
  const [stats, setStats] = useState({ nodes: 0, edges: 0 });
  const [hasGraph, setHasGraph] = useState(false);
  const [info, setInfo] = useState(null);
  const [bridges, setBridges] = useState([]);
  const visRef = useRef(null);
  const netRef = useRef(null);

  const load = async (id) => {
    if (!id) return;
    try {
      const g = await getGraph(id);
      const rawNodes = g.nodes || [];
      const rawEdges = g.edges || [];

      // degree → size emphasis
      const degree = {};
      rawEdges.forEach((e) => { degree[e.from] = (degree[e.from] || 0) + 1; degree[e.to] = (degree[e.to] || 0) + 1; });
      const maxDeg = Math.max(1, ...Object.values(degree));

      const nodes = rawNodes.map((n) => {
        const kind = nodeKind(n.label ?? n.id);
        const st = KIND_STYLE[kind];
        const deg = degree[n.id] || 0;
        const isHub = kind === 'fact' || deg >= maxDeg * 0.6;
        return {
          id: n.id,
          label: n.label ?? String(n.id),
          title: n.title || '',
          shape: 'box',
          shapeProperties: { borderRadius: 8 },
          margin: { top: 8, bottom: 8, left: 12, right: 12 },
          borderWidth: isHub ? 3 : 1.5,
          color: {
            background: st.background,
            border: st.border,
            highlight: { background: st.background, border: '#ffffff' },
            hover: { background: st.background, border: '#ffffff' },
          },
          font: { color: '#e6edf3', size: isHub ? 16 : 13, face: 'Inter, sans-serif', bold: isHub ? { color: '#fff' } : false },
          shadow: { enabled: true, color: st.glow, size: isHub ? 22 : 12, x: 0, y: 0 },
          _kind: kind,
        };
      });

      const edges = rawEdges.map((e) => {
        const card = cardinalityOf(e);
        const cs = (card && CARD_STYLE[card]) || CARD_DEFAULT;
        return {
          from: e.from,
          to: e.to,
          label: e.label || '',
          title: e.title || e.label || '',
          arrows: { to: { enabled: true, scaleFactor: card === 'N:N' ? 0.8 : 0.6, type: 'arrow' } },
          color: { color: cs.color, highlight: cs.color, hover: '#ffffff', opacity: 1 },
          width: cs.width,
          selectionWidth: cs.width + 1,
          dashes: card === 'N:N' ? [6, 4] : false, // many-to-many drawn dashed
          smooth: { enabled: true, type: 'dynamic', roundness: 0.5 },
          font: { color: card ? cs.color : '#9fb2c8', size: 10, face: 'Inter, sans-serif', strokeWidth: 4, strokeColor: '#0a0e1a', align: 'middle' },
        };
      });

      setStats({ nodes: nodes.length, edges: edges.length });
      setHasGraph(nodes.length > 0);
      if (netRef.current) { netRef.current.destroy(); netRef.current = null; }
      if (!nodes.length || !visRef.current) return;

      netRef.current = new Network(
        visRef.current,
        { nodes, edges },
        {
          autoResize: true,
          nodes: { borderWidthSelected: 3 },
          edges: { smooth: { type: 'dynamic' } },
          physics: {
            solver: 'forceAtlas2Based',
            forceAtlas2Based: {
              gravitationalConstant: -70,
              centralGravity: 0.008,
              springLength: 150,
              springConstant: 0.08,
              damping: 0.6,
              avoidOverlap: 0.7,
            },
            stabilization: { enabled: true, iterations: 280, fit: true },
            minVelocity: 0.5,
          },
          interaction: {
            hover: true,
            tooltipDelay: 120,
            zoomView: true,
            dragView: true,
            navigationButtons: true,
            keyboard: { enabled: true, bindToWindow: false },
            multiselect: true,
          },
        },
      );

      // Hover/selection emphasis: dim unrelated nodes/edges on select.
      netRef.current.on('click', (params) => {
        if (params.nodes.length) {
          const node = nodes.find((n) => n.id === params.nodes[0]);
          setInfo(node || null);
        } else {
          setInfo(null);
        }
      });
    } catch (e) {
      toast(`Graph load failed: ${e.message}`, 'error');
      setHasGraph(false);
    }
  };

  useEffect(() => {
    listBridges().then((b) => setBridges(Array.isArray(b) ? b : [])).catch(() => setBridges([]));
  }, [refreshTick]);

  useEffect(() => {
    load(activeSourceId);
    return () => { if (netRef.current) { netRef.current.destroy(); netRef.current = null; } };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSourceId, refreshTick]);

  const srcBridges = bridges.filter(
    (b) => b.from_source_id === activeSourceId || b.to_source_id === activeSourceId || !activeSourceId,
  );

  return (
    <div id="view-graph" className="view active">
      <div id="graph-controls">
        <span style={{ fontSize: 12, color: 'var(--text-1)' }}>Source:</span>
        <select className="search-input" style={{ padding: '5px 8px', width: 200 }} value={activeSourceId} onChange={(e) => setActiveSourceId(e.target.value)}>
          <option value="">— select —</option>
          {sources.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
        </select>
        <button className="btn btn-secondary" onClick={() => load(activeSourceId)}>
          <IconRefresh />
          Load
        </button>
        <div style={{ width: 1, height: 20, background: 'var(--border)' }} />
        <button className="btn btn-ghost" onClick={() => netRef.current && netRef.current.fit({ animation: true })}>Fit</button>
        <button className="btn btn-ghost" onClick={() => netRef.current && netRef.current.stabilize()}>Reset Layout</button>

        {/* node legend */}
        <div style={{ display: 'flex', gap: 12, marginLeft: 16, alignItems: 'center' }}>
          {LEGEND.map(([label, color]) => (
            <span key={label} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--text-2)' }}>
              <span style={{ width: 11, height: 11, borderRadius: 3, background: 'transparent', border: `2px solid ${color}` }} />
              {label}
            </span>
          ))}
        </div>
        <div style={{ width: 1, height: 18, background: 'var(--border)', marginLeft: 4 }} />
        {/* cardinality (edge) legend */}
        <div style={{ display: 'flex', gap: 12, marginLeft: 4, alignItems: 'center' }}>
          <span style={{ fontSize: 11, color: 'var(--text-2)' }}>Edges:</span>
          {CARD_LEGEND.map(([label, color]) => (
            <span key={label} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--text-2)' }}>
              <span style={{ width: 16, height: 0, borderTop: `${label === 'N:N' ? '2px dashed' : '2px solid'} ${color}` }} />
              {label}
            </span>
          ))}
        </div>

        <div id="graph-stats">
          <span>◉ {stats.nodes} nodes</span>
          <span>— {stats.edges} edges</span>
        </div>
      </div>

      <div id="graph-container">
        <div id="kg-vis" ref={visRef} />
        {!hasGraph && (
          <div id="graph-empty">
            <IconGraph strokeWidth="1.5" />
            Select a source to explore the knowledge graph
          </div>
        )}
        {info && (
          <div id="graph-info-panel" className="visible">
            <div className="info-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ width: 10, height: 10, borderRadius: 3, border: `2px solid ${KIND_STYLE[info._kind || 'other'].border}` }} />
              {info.label}
            </div>
            <div id="gip-body">
              {String(info.title || '')
                .split('\n')
                .filter(Boolean)
                .map((line, i) => {
                  const [k, ...rest] = line.split(':');
                  return (
                    <div className="info-row" key={i}>
                      <span className="info-label">{rest.length ? k : '•'}</span>
                      <span className="info-value">{rest.length ? rest.join(':').trim() : line}</span>
                    </div>
                  );
                })}
              {!info.title && <div className="info-row"><span className="info-label">type</span><span className="info-value">{info._kind}</span></div>}
            </div>
            <button className="btn btn-ghost" style={{ width: '100%', marginTop: 10, fontSize: 11 }} onClick={() => setInfo(null)}>Dismiss</button>
          </div>
        )}
      </div>

      <div id="graph-bridges">
        <div id="graph-bridges-header">
          🔗 CROSS-SOURCE BRIDGES
          {srcBridges.length > 0 && <span id="graph-bridges-count">{srcBridges.length}</span>}
          <div id="graph-bridges-legend">
            <span className="bridges-legend-item bridges-legend-declared">declared</span>
            <span className="bridges-legend-item bridges-legend-inferred">inferred</span>
            <span className="bridges-legend-item bridges-legend-disabled">disabled</span>
          </div>
        </div>
        <div id="graph-bridges-body" style={{ padding: srcBridges.length ? '8px 14px' : 0 }}>
          {srcBridges.length === 0 ? (
            <div id="graph-bridges-empty">No cross-source bridges for this source.</div>
          ) : (
            srcBridges.map((b, i) => {
              const color = !b.enabled ? 'var(--text-2)' : b.source === 'inferred' ? 'var(--blue)' : 'var(--green)';
              return (
                <div key={i} style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color, padding: '2px 0' }}>
                  {b.from_kg || b.from_entity}.{b.from_column} → {b.to_kg || b.to_entity}.{b.to_column}
                  {b.join_type ? ` · ${b.join_type}` : ''}
                  {b.confidence != null ? ` · ${Math.round(b.confidence * 100)}%` : ''}
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
