import { useEffect, useRef, useState } from 'react';
import { Network } from 'vis-network/standalone/esm/vis-network';
import { useAppState } from '../state.jsx';
import { getOntology, saveOntology, validateOntology } from '../api/clients.js';

// Light Turtle/OWL parser — enough to populate the explorer tree + class graph.
function parseOntology(ttl) {
  const text = ttl || '';
  const grab = (re) => {
    const out = new Set();
    let m;
    while ((m = re.exec(text))) out.add(m[1]);
    return [...out];
  };
  const classes = grab(/(\S+)\s+(?:a|rdf:type)\s+owl:Class/g);
  const objProps = grab(/(\S+)\s+(?:a|rdf:type)\s+owl:ObjectProperty/g);
  const dataProps = grab(/(\S+)\s+(?:a|rdf:type)\s+owl:DatatypeProperty/g);
  const subOf = [];
  const re = /(\S+)\s+rdfs:subClassOf\s+(\S+)\s*[.;]/g;
  let m;
  while ((m = re.exec(text))) subOf.push([m[1], m[2].replace(/[.;]$/, '')]);
  return { classes, objProps, dataProps, subOf };
}

const short = (iri) => String(iri).replace(/^.*[#/:]/, '').replace(/[<>]/g, '') || iri;

function Section({ title, dotClass, items }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="onto-section">
      <div className="onto-section-hdr" onClick={() => setOpen((o) => !o)}>
        <span className="onto-section-toggle">{open ? '▾' : '▸'}</span>
        {title} ({items.length})
      </div>
      <div className={`onto-section-body ${open ? '' : 'collapsed'}`}>
        {items.map((it, i) => (
          <div className="onto-item" key={i}>
            <span className={`onto-dot ${dotClass}`} />
            <span className="onto-item-label">{short(it)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function OntologyViewer() {
  const { sources, activeSourceId, setActiveSourceId, toast } = useAppState();
  const [content, setContent] = useState('');
  const [tab, setTab] = useState('ttl');
  const [sparql, setSparql] = useState(
    'PREFIX owl: <http://www.w3.org/2002/07/owl#>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n\nSELECT ?class WHERE {\n  ?class a owl:Class .\n}',
  );
  const [parsed, setParsed] = useState({ classes: [], objProps: [], dataProps: [], subOf: [] });
  const [vizCollapsed, setVizCollapsed] = useState(false);
  const visRef = useRef(null);
  const netRef = useRef(null);

  useEffect(() => {
    if (!activeSourceId) {
      setContent('');
      setParsed({ classes: [], objProps: [], dataProps: [], subOf: [] });
      return;
    }
    getOntology(activeSourceId)
      .then((d) => {
        const ttl = d.content || d.ontology_content || '';
        setContent(ttl);
        setParsed(parseOntology(ttl));
      })
      .catch(() => {
        setContent('');
        setParsed({ classes: [], objProps: [], dataProps: [], subOf: [] });
      });
  }, [activeSourceId]);

  // Render the class graph in the viz panel.
  useEffect(() => {
    if (vizCollapsed || !visRef.current || parsed.classes.length === 0) {
      if (netRef.current) {
        netRef.current.destroy();
        netRef.current = null;
      }
      return;
    }
    const nodes = parsed.classes.map((c) => ({
      id: short(c),
      label: short(c),
      shape: 'dot',
      size: 14,
      color: { background: '#111827', border: '#58a6ff' },
      font: { color: '#e6edf3', size: 12 },
    }));
    const ids = new Set(nodes.map((n) => n.id));
    const edges = parsed.subOf
      .filter(([a, b]) => ids.has(short(a)) && ids.has(short(b)))
      .map(([a, b]) => ({ from: short(a), to: short(b), dashes: true, color: '#3a4a6a' }));
    if (netRef.current) netRef.current.destroy();
    netRef.current = new Network(
      visRef.current,
      { nodes, edges },
      {
        physics: { stabilization: { iterations: 120 } },
        edges: { smooth: { type: 'dynamic' }, arrows: { to: { enabled: true, scaleFactor: 0.6 } } },
      },
    );
    return () => {
      if (netRef.current) {
        netRef.current.destroy();
        netRef.current = null;
      }
    };
  }, [parsed, vizCollapsed]);

  const save = async () => {
    try {
      await saveOntology(activeSourceId, content, true);
      toast('Ontology saved — rebuilding KG', 'success');
    } catch (e) {
      toast(`Save failed: ${e.message}`, 'error');
    }
  };
  const validate = async () => {
    try {
      const r = await validateOntology(activeSourceId);
      toast(r && r.valid ? 'Validation passed' : 'Validation found issues', r && r.valid ? 'success' : 'warn');
    } catch (e) {
      toast(`Validate failed: ${e.message}`, 'error');
    }
  };

  return (
    <div id="view-ontology" className="view active">
      <div id="onto-layout">
        <div id="onto-explorer">
          <div className="onto-explorer-hdr">ONTOLOGY EXPLORER</div>
          <div style={{ padding: '6px 10px', borderBottom: '1px solid var(--border)' }}>
            <select
              value={activeSourceId}
              onChange={(e) => setActiveSourceId(e.target.value)}
              style={{ width: '100%', background: 'var(--bg-3)', border: '1px solid var(--border)', color: 'var(--text-0)', padding: '5px 8px', borderRadius: 'var(--radius)', fontSize: 12 }}
            >
              <option value="">— select source —</option>
              {sources.map((s) => (
                <option key={s.id} value={s.id}>{s.name}</option>
              ))}
            </select>
          </div>
          <div id="onto-tree">
            {parsed.classes.length === 0 ? (
              <div className="empty-state" style={{ height: 120, fontSize: 11 }}>Load a source</div>
            ) : (
              <>
                <Section title="Classes" dotClass="class-dot" items={parsed.classes} />
                <Section title="Object Properties" dotClass="prop-dot" items={parsed.objProps} />
                <Section title="Data Properties" dotClass="data-dot" items={parsed.dataProps} />
              </>
            )}
          </div>
        </div>

        <div id="onto-right">
          <div id="onto-toolbar">
            <div id="onto-tabs">
              <button className={`onto-tab ${tab === 'ttl' ? 'active' : ''}`} onClick={() => setTab('ttl')}>
                ONTOLOGY.TTL
              </button>
              <button className={`onto-tab ${tab === 'sparql' ? 'active' : ''}`} onClick={() => setTab('sparql')}>
                SPARQL
              </button>
            </div>
            <button className="btn btn-ghost" onClick={() => { navigator.clipboard.writeText(content); toast('Copied', 'success'); }}>
              Copy
            </button>
            <button className="btn btn-ghost" onClick={validate} disabled={!activeSourceId}>
              Validate
            </button>
            <button className="btn btn-primary" onClick={save} disabled={!activeSourceId}>
              Save &amp; Rebuild KG
            </button>
          </div>

          <div id="onto-editor-area">
            <div id="onto-ttl-wrap" className={tab === 'ttl' ? '' : 'onto-tab-hidden'}>
              <textarea
                id="ontology-editor"
                spellCheck={false}
                value={content}
                onChange={(e) => setContent(e.target.value)}
                placeholder="-- OWL/Turtle ontology will appear here after indexing…"
              />
            </div>
            <div id="onto-sparql-wrap" className={tab === 'sparql' ? '' : 'onto-tab-hidden'}>
              <textarea id="sparql-editor" spellCheck={false} value={sparql} onChange={(e) => setSparql(e.target.value)} />
            </div>
          </div>

          <div id="onto-viz-panel" className={vizCollapsed ? 'collapsed' : ''}>
            <div className="onto-viz-hdr">
              <span>ONTOLOGY CLASS GRAPH &amp; PROPERTY VISUALIZER</span>
              <div style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
                <button className="btn btn-ghost" style={{ padding: '2px 8px', fontSize: 10 }} onClick={() => setVizCollapsed((c) => !c)}>
                  {vizCollapsed ? '▴' : '▾'}
                </button>
              </div>
            </div>
            <div id="onto-vis" ref={visRef} />
          </div>
        </div>
      </div>
    </div>
  );
}
