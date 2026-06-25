import { useEffect, useRef, useState } from 'react';
import { Network } from 'vis-network/standalone/esm/vis-network';
import { useAppState } from '../state.jsx';
import { getGraph, listBridges } from '../api/clients.js';
import { IconRefresh, IconGraph } from '../components/Icons.jsx';

function nodeKind(label = '') {
  const s = String(label).toUpperCase();
  if (/FACT/.test(s)) return 'fact';
  if (/DIM/.test(s)) return 'dim';
  if (/KPI|SUMMARY|METRIC/.test(s)) return 'kpi';
  return 'other';
}

// Primary source node styles
const KIND_STYLE = {
  fact: { background: '#3b2a0e', border: '#f59e0b', glow: 'rgba(245,158,11,0.25)' },
  dim:  { background: '#11233e', border: '#58a6ff', glow: 'rgba(88,166,255,0.20)' },
  kpi:  { background: '#2a163a', border: '#bc8cff', glow: 'rgba(188,140,255,0.20)' },
  other:{ background: '#15202b', border: '#39c5cf', glow: 'rgba(57,197,207,0.18)' },
};

// Remote source node styles — vivid green backgrounds so they stand out immediately
const KIND_STYLE_REMOTE = {
  fact: { background: '#1c3d18', border: '#4ade80', glow: 'rgba(74,222,128,0.40)' },
  dim:  { background: '#0f3020', border: '#4ade80', glow: 'rgba(74,222,128,0.35)' },
  kpi:  { background: '#1a3820', border: '#4ade80', glow: 'rgba(74,222,128,0.35)' },
  other:{ background: '#132d18', border: '#4ade80', glow: 'rgba(74,222,128,0.30)' },
};


const LEGEND = [
  ['Fact table',    '#f59e0b'],
  ['Dimension',     '#58a6ff'],
  ['KPI / summary', '#bc8cff'],
  ['Other',         '#39c5cf'],
  ['Cross-source',  '#3fb950'],
];

const CARD_STYLE = {
  '1:1': { color: '#3fb950', width: 1.4 },
  '1:N': { color: '#58a6ff', width: 1.4 },
  'N:1': { color: '#39c5cf', width: 1.4 },
  'N:N': { color: '#f59e0b', width: 2.4 },
};
const CARD_DEFAULT  = { color: 'rgba(120,140,170,0.45)', width: 1.2 };
const CARD_LEGEND   = [['1:1','#3fb950'],['1:N','#58a6ff'],['N:1','#39c5cf'],['N:N','#f59e0b']];

function cardinalityOf(edge) {
  const txt = `${edge.label || ''} ${edge.title || ''}`;
  const m = txt.match(/([1nm])\s*:\s*([1nm])/i);
  if (!m) return null;
  const norm = (c) => (c === '1' ? '1' : 'N');
  return `${norm(m[1].toLowerCase())}:${norm(m[2].toLowerCase())}`;
}

function buildVisNodes(rawNodes, rawEdges, { idPrefix = '', remote = false, sourceName = '' } = {}) {
  const degree = {};
  rawEdges.forEach((e) => {
    degree[e.from] = (degree[e.from] || 0) + 1;
    degree[e.to]   = (degree[e.to]   || 0) + 1;
  });
  const maxDeg = Math.max(1, ...Object.values(degree));

  return rawNodes.map((n) => {
    const kind      = nodeKind(n.label ?? n.id);
    const st        = remote ? KIND_STYLE_REMOTE[kind] : KIND_STYLE[kind];
    const deg       = degree[n.id] || 0;
    const isHub     = kind === 'fact' || deg >= maxDeg * 0.6;
    const baseLabel = n.label ?? String(n.id);

    return {
      id:    idPrefix + n.id,
      // ⬡ prefix + source name second line clearly marks remote nodes
      label: remote ? `⬡ ${baseLabel}\n[${sourceName}]` : baseLabel,
      title: remote
        ? `[${sourceName}] ${n.title || baseLabel}\n(remote source node)`
        : (n.title || ''),
      shape: 'box',
      shapeProperties: { borderRadius: 8 },
      // Thick dashed border = remote; solid = primary
      borderDashes: remote ? [10, 5] : false,
      margin: { top: 8, bottom: 8, left: 12, right: 12 },
      borderWidth: remote ? 3 : (isHub ? 3 : 1.5),
      color: {
        background: st.background,
        border:     st.border,
        highlight: { background: st.background, border: '#ffffff' },
        hover:     { background: st.background, border: '#ffffff' },
      },
      font: remote
        ? { color: '#a8f0b8', size: isHub ? 14 : 12, face: 'Inter, sans-serif' }
        : { color: '#e6edf3', size: isHub ? 16 : 13, face: 'Inter, sans-serif',
            bold: isHub ? { color: '#fff' } : false },
      shadow: { enabled: true, color: st.glow, size: isHub ? 24 : 14, x: 0, y: 0 },
      _kind:   kind,
      _prefix: idPrefix,
      _remote: remote,
    };
  });
}

function buildVisEdges(rawEdges, idPrefix = '', muted = false) {
  return rawEdges.map((e, i) => {
    const card = cardinalityOf(e);
    const cs   = muted ? { color: 'rgba(63,185,80,0.30)', width: 1 }
                       : ((card && CARD_STYLE[card]) || CARD_DEFAULT);
    return {
      id:    idPrefix ? `${idPrefix}edge_${i}` : undefined,
      from:  idPrefix + e.from,
      to:    idPrefix + e.to,
      label: muted ? '' : (e.label || ''),
      title: e.title || e.label || '',
      arrows:{ to: { enabled: true, scaleFactor: muted ? 0.4 : (card === 'N:N' ? 0.8 : 0.6), type: 'arrow' } },
      color: { color: cs.color, highlight: cs.color, hover: '#ffffff', opacity: muted ? 0.5 : 1 },
      width: cs.width,
      dashes: muted ? [3, 5] : (card === 'N:N' ? [6, 4] : false),
      smooth: { enabled: true, type: 'dynamic', roundness: 0.5 },
      font:  { color: muted ? 'transparent' : (card ? cs.color : '#9fb2c8'), size: 10, face: 'Inter, sans-serif', strokeWidth: 4, strokeColor: '#0a0e1a', align: 'middle' },
    };
  });
}

export default function GraphExplorer() {
  const { sources, activeSourceId, setActiveSourceId, refreshTick, toast } = useAppState();
  const [stats,          setStats]          = useState({ nodes: 0, edges: 0 });
  const [hasGraph,       setHasGraph]       = useState(false);
  const [info,           setInfo]           = useState(null);
  const [bridges,        setBridges]        = useState([]);
  const [selectedRemoteKg, setSelectedRemoteKg] = useState('');

  const visRef = useRef(null);
  const netRef = useRef(null);

  // Primary graph data (preserved for re-styling + stats)
  const rawNodesRef    = useRef([]);  // vis node objects for primary source
  const rawEdgesRef    = useRef([]);  // vis edge objects for primary source
  const bridgesRef     = useRef([]);  // all enabled bridges involving primary source

  // Remote graph data (added on top of primary, removed on clear)
  const remoteNodesRef  = useRef([]);
  const remoteEdgesRef  = useRef([]);
  const bridgeEdgesRef  = useRef([]);

  // ── Load the primary graph ────────────────────────────────────────────────────
  const load = async (id) => {
    if (!id) return;
    try {
      const [g, allBridges] = await Promise.all([getGraph(id), listBridges().catch(() => [])]);
      const allBridgesArr = Array.isArray(allBridges) ? allBridges : [];
      setBridges(allBridgesArr);

      const rawNodes = g.nodes || [];
      const rawEdges = g.edges || [];

      const nodes = buildVisNodes(rawNodes, rawEdges);
      const edges = buildVisEdges(rawEdges);

      const activeBridges = allBridgesArr.filter(
        (b) => b.enabled && (b.from_kg === id || b.to_kg === id),
      );

      rawNodesRef.current  = nodes;
      rawEdgesRef.current  = edges;
      bridgesRef.current   = activeBridges;
      remoteNodesRef.current  = [];
      remoteEdgesRef.current  = [];
      bridgeEdgesRef.current  = [];

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
            hover: true, tooltipDelay: 120, zoomView: true,
            dragView: true, navigationButtons: true,
            keyboard: { enabled: true, bindToWindow: false },
            multiselect: true,
          },
        },
      );

      netRef.current.on('click', (params) => {
        if (params.nodes.length) {
          const allVisible = [...rawNodesRef.current, ...remoteNodesRef.current];
          setInfo(allVisible.find((n) => n.id === params.nodes[0]) || null);
        } else {
          setInfo(null);
        }
      });

      netRef.current.on('stabilizationIterationsDone', () => {
        if (netRef.current) {
          netRef.current.setOptions({ physics: false });
          netRef.current.fit({ animation: false });
        }
      });
    } catch (e) {
      toast(`Graph load failed: ${e.message}`, 'error');
      setHasGraph(false);
    }
  };

  useEffect(() => {
    setSelectedRemoteKg('');
    load(activeSourceId);
    return () => { if (netRef.current) { netRef.current.destroy(); netRef.current = null; } };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSourceId, refreshTick]);

  // ── Bridges-from: load remote graph + draw bridge edges ──────────────────────
  useEffect(() => {
    if (!netRef.current || !rawNodesRef.current.length) return;

    // Always clean up any previously rendered remote content first
    if (remoteNodesRef.current.length) {
      netRef.current.body.data.nodes.remove(remoteNodesRef.current.map((n) => n.id));
    }
    const staleEdgeIds = [
      ...remoteEdgesRef.current.map((e) => e.id),
      ...bridgeEdgesRef.current.map((e) => e.id),
    ].filter(Boolean);
    if (staleEdgeIds.length) netRef.current.body.data.edges.remove(staleEdgeIds);

    remoteNodesRef.current = [];
    remoteEdgesRef.current = [];
    bridgeEdgesRef.current = [];

    // Restore primary node styles (remove any previous green highlights)
    netRef.current.body.data.nodes.update(
      rawNodesRef.current.map((n) => ({
        id: n.id, borderWidth: n.borderWidth, color: n.color, shadow: n.shadow, title: n.title,
      })),
    );
    setStats({ nodes: rawNodesRef.current.length, edges: rawEdgesRef.current.length });

    if (!selectedRemoteKg) return;

    const remoteName = sources.find((s) => s.id === selectedRemoteKg)?.name || 'Remote';
    const REMOTE_PREFIX = 'r2__';

    getGraph(selectedRemoteKg)
      .then((g) => {
        if (!netRef.current) return;

        const rawRemoteNodes = g.nodes || [];
        const rawRemoteEdges = g.edges || [];

        // Build remote vis nodes — same kind colours, dashed border, source name as second label line
        const remoteNodes = buildVisNodes(rawRemoteNodes, rawRemoteEdges, {
          idPrefix: REMOTE_PREFIX,
          remote: true,
          sourceName: remoteName,
        });

        // Remote internal edges (muted, dashed)
        const remoteEdges = buildVisEdges(rawRemoteEdges, REMOTE_PREFIX, true);

        // Build label → vis-node-id maps for both graphs
        const primaryLabelToId = {};
        rawNodesRef.current.forEach((n) => {
          primaryLabelToId[n.label.toLowerCase()] = n.id;
        });
        const remoteLabelToId = {};
        rawRemoteNodes.forEach((n) => {
          remoteLabelToId[(n.label ?? String(n.id)).toLowerCase()] = REMOTE_PREFIX + n.id;
        });

        // Filter bridges to only this pair
        const pairBridges = bridgesRef.current.filter(
          (b) =>
            (b.from_kg === activeSourceId && b.to_kg === selectedRemoteKg) ||
            (b.to_kg   === activeSourceId && b.from_kg === selectedRemoteKg),
        );

        // De-duplicate: one bridge edge per (primaryNode, remoteNode) pair
        const seen = new Set();
        const bridgeEdges = [];
        pairBridges.forEach((b, i) => {
          const isPrimaryFrom = b.from_kg === activeSourceId;
          const primaryEntity = (isPrimaryFrom ? b.from_entity : b.to_entity) || '';
          const remoteEntity  = (isPrimaryFrom ? b.to_entity  : b.from_entity) || '';
          const primaryCol    = isPrimaryFrom ? b.from_column : b.to_column;
          const remoteCol     = isPrimaryFrom ? b.to_column   : b.from_column;

          const fromId = primaryLabelToId[primaryEntity.toLowerCase()];
          const toId   = remoteLabelToId[remoteEntity.toLowerCase()];
          if (!fromId || !toId) return;

          const pairKey = `${fromId}|${toId}`;
          const edgeLabel = seen.has(pairKey) ? '' : `${primaryCol} → ${remoteCol}`;
          seen.add(pairKey);

          bridgeEdges.push({
            id: `bridge__${i}`,
            from: fromId,
            to:   toId,
            label: edgeLabel,
            title: `🔗 Bridge\n${primaryEntity}.${primaryCol} → ${remoteEntity}.${remoteCol}\n${b.join_type ? b.join_type + ' · ' : ''}${Math.round((b.confidence || 0) * 100)}% confidence`,
            arrows: { to: { enabled: true, scaleFactor: 0.75, type: 'arrow' } },
            color:  { color: '#3fb950', highlight: '#5ccd6e', hover: '#5ccd6e', opacity: 0.95 },
            width:  2.2,
            dashes: [9, 4],
            smooth: { enabled: true, type: 'curvedCW', roundness: 0.25 },
            font:   { color: '#3fb950', size: 10, face: 'Inter, sans-serif', strokeWidth: 4, strokeColor: '#0a0e1a', align: 'middle' },
          });
        });

        // Highlight primary nodes that have bridges to the remote source
        const bridgedPrimaryIds = new Set(
          pairBridges.map((b) => {
            const e = b.from_kg === activeSourceId ? b.from_entity : b.to_entity;
            return primaryLabelToId[e?.toLowerCase()];
          }).filter(Boolean),
        );
        const primaryUpdates = rawNodesRef.current.map((n) => {
          if (!bridgedPrimaryIds.has(n.id)) return { id: n.id };
          const st = KIND_STYLE[n._kind || 'other'];
          return {
            id: n.id,
            borderWidth: 3,
            color: { background: st.background, border: '#4ade80', highlight: { background: st.background, border: '#ffffff' }, hover: { background: st.background, border: '#ffffff' } },
            shadow: { enabled: true, color: 'rgba(74,222,128,0.55)', size: 28, x: 0, y: 0 },
          };
        });

        // Commit to vis-network
        netRef.current.body.data.nodes.add(remoteNodes);
        netRef.current.body.data.edges.add([...remoteEdges, ...bridgeEdges]);
        netRef.current.body.data.nodes.update(primaryUpdates);

        // Enable physics briefly so remote nodes spread out, then freeze again
        netRef.current.setOptions({ physics: { enabled: true, stabilization: { iterations: 150 } } });
        netRef.current.once('stabilizationIterationsDone', () => {
          if (netRef.current) {
            netRef.current.setOptions({ physics: false });
            netRef.current.fit({ animation: { duration: 600, easingFunction: 'easeInOutQuad' } });
          }
        });
        netRef.current.stabilize(150);

        remoteNodesRef.current = remoteNodes;
        remoteEdgesRef.current = remoteEdges;
        bridgeEdgesRef.current = bridgeEdges;

        setStats({
          nodes: rawNodesRef.current.length + remoteNodes.length,
          edges: rawEdgesRef.current.length + remoteEdges.length + bridgeEdges.length,
        });
      })
      .catch((e) => toast(`Failed to load bridge graph: ${e.message}`, 'error'));

  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedRemoteKg]);

  const srcBridges = bridges.filter(
    (b) => b.from_kg === activeSourceId || b.to_kg === activeSourceId || !activeSourceId,
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
        <span style={{ fontSize: 12, color: 'var(--text-1)' }}>Bridges from:</span>
        <select
          className="search-input"
          style={{ padding: '5px 8px', width: 160 }}
          value={selectedRemoteKg}
          onChange={(e) => setSelectedRemoteKg(e.target.value)}
        >
          <option value="">— none —</option>
          {[...new Set(
            bridges
              .filter((b) => b.from_kg === activeSourceId || b.to_kg === activeSourceId)
              .map((b) => (b.from_kg === activeSourceId ? b.to_kg : b.from_kg))
          )].filter((kid) => sources.some((s) => s.id === kid) && kid !== activeSourceId)
            .map((kid) => {
            const name = sources.find((s) => s.id === kid)?.name;
            return <option key={kid} value={kid}>{name}</option>;
          })}
        </select>
        <div style={{ width: 1, height: 20, background: 'var(--border)' }} />
        <button className="btn btn-ghost" onClick={() => netRef.current && netRef.current.fit({ animation: true })}>Fit</button>
        <button className="btn btn-ghost" onClick={() => { if (netRef.current) { netRef.current.setOptions({ physics: true }); netRef.current.stabilize(); } }}>Reset Layout</button>

        <div style={{ display: 'flex', gap: 12, marginLeft: 16, alignItems: 'center' }}>
          {LEGEND.map(([label, color]) => (
            <span key={label} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--text-2)' }}>
              <span style={{ width: 11, height: 11, borderRadius: 3, background: 'transparent', border: `2px solid ${color}` }} />
              {label}
            </span>
          ))}
        </div>
        <div style={{ width: 1, height: 18, background: 'var(--border)', marginLeft: 4 }} />
        <div style={{ display: 'flex', gap: 12, marginLeft: 4, alignItems: 'center' }}>
          <span style={{ fontSize: 11, color: 'var(--text-2)' }}>Edges:</span>
          {CARD_LEGEND.map(([label, color]) => (
            <span key={label} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--text-2)' }}>
              <span style={{ width: 16, height: 0, borderTop: `${label === 'N:N' ? '2px dashed' : '2px solid'} ${color}` }} />
              {label}
            </span>
          ))}
          {selectedRemoteKg && (
            <span style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--text-2)' }}>
              <span style={{ width: 16, height: 0, borderTop: '2px dashed #3fb950' }} />
              bridge
            </span>
          )}
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
              <span style={{ width: 10, height: 10, borderRadius: 3, border: `2px solid ${info._remote ? '#3fb950' : KIND_STYLE[info._kind || 'other'].border}` }} />
              {info.label}
              {info._remote && <span style={{ fontSize: 10, color: '#3fb950', marginLeft: 4 }}>remote</span>}
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
              const isSelected = selectedRemoteKg &&
                ((b.from_kg === activeSourceId && b.to_kg === selectedRemoteKg) ||
                 (b.to_kg === activeSourceId && b.from_kg === selectedRemoteKg));
              const color = !b.enabled ? 'var(--text-2)' : b.source === 'inferred' ? 'var(--blue)' : 'var(--green)';
              return (
                <div key={i} style={{
                  fontFamily: 'var(--font-mono)', fontSize: 11, color,
                  padding: '2px 0',
                  opacity: selectedRemoteKg && !isSelected ? 0.35 : 1,
                  fontWeight: isSelected ? 600 : 'normal',
                }}>
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
