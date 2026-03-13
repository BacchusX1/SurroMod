import { useEffect, useRef, useCallback, useState, lazy, Suspense } from 'react';
import useStore from '../store';
import type { HPTunerNodeData } from '../types';

const HPTunerAnalytics = lazy(() => import('./HPTunerAnalytics'));

// ─── SSE connection hook ────────────────────────────────────────────────────

function useLogStream() {
  const addLogMessage = useStore((s) => s.addLogMessage);

  useEffect(() => {
    let eventSource: EventSource | null = null;
    let retryTimeout: ReturnType<typeof setTimeout> | null = null;

    function connect() {
      eventSource = new EventSource('/api/logs/stream');

      eventSource.onmessage = (event) => {
        addLogMessage(event.data);
      };

      eventSource.onerror = () => {
        // Auto-reconnect after 3 seconds
        eventSource?.close();
        eventSource = null;
        retryTimeout = setTimeout(connect, 3000);
      };
    }

    connect();

    return () => {
      eventSource?.close();
      if (retryTimeout) clearTimeout(retryTimeout);
    };
  }, [addLogMessage]);
}

// ─── HP Tuning Results Sub-Panel ────────────────────────────────────────────

function HPTuningResultsPanel() {
  const { nodes, selectedNodeId } = useStore();
  const [subTab, setSubTab] = useState<'table' | 'analytics'>('table');

  // Find the selected HP tuner node (or the first HP tuner node with results)
  const hpNode = (() => {
    if (selectedNodeId) {
      const n = nodes.find((nd) => nd.id === selectedNodeId);
      if (n && (n.data as any).category === 'hp_tuner') return n;
    }
    // Fallback: find any HP tuner node with results
    return nodes.find(
      (n) => (n.data as any).category === 'hp_tuner' && (n.data as HPTunerNodeData).tuningResults?.length,
    );
  })();

  if (!hpNode) {
    return (
      <div className="output-panel__empty">
        No HP tuning results yet. Select an HP Tuner node and run tuning.
      </div>
    );
  }

  const data = hpNode.data as HPTunerNodeData;
  const results = data.tuningResults;

  if (!results || results.length === 0) {
    return (
      <div className="output-panel__empty">
        {data.tuningStatus === 'running'
          ? 'Tuning in progress… results will appear here when done.'
          : 'No results yet. Start tuning from the Inspector panel.'}
      </div>
    );
  }

  return (
    <div className="hp-results-panel">
      {/* Sub-tab bar */}
      <div className="hp-results-panel__tabs">
        <button
          className={`hp-results-panel__tab${subTab === 'table' ? ' hp-results-panel__tab--active' : ''}`}
          onClick={() => setSubTab('table')}
        >
          Table ({results.length})
        </button>
        <button
          className={`hp-results-panel__tab${subTab === 'analytics' ? ' hp-results-panel__tab--active' : ''}`}
          onClick={() => setSubTab('analytics')}
        >
          📊 Analytics
        </button>
      </div>

      {subTab === 'table' && (
        <div className="hp-results-panel__table-wrap">
          <table className="hp-results-panel__table">
            <thead>
              <tr>
                <th>#</th>
                <th>Score</th>
                {results[0]?.holdout_score != null && <th>Holdout</th>}
                {results[0]?.n_params != null && <th>Params</th>}
                <th>Config</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r) => {
                const isBest =
                  data.bestScore != null && Math.abs(r.score - data.bestScore) < 1e-10;
                return (
                  <tr key={r.iteration} className={isBest ? 'hp-results-panel__row--best' : ''}>
                    <td>{r.iteration}</td>
                    <td className="mono">{r.score.toFixed(6)}</td>
                    {results[0]?.holdout_score != null && (
                      <td className="mono">
                        {r.holdout_score != null ? r.holdout_score.toFixed(6) : '—'}
                      </td>
                    )}
                    {results[0]?.n_params != null && (
                      <td className="mono">{r.n_params != null ? r.n_params.toLocaleString() : '—'}</td>
                    )}
                    <td className="dim">
                      {Object.entries(r.config)
                        .map(
                          ([k, v]) =>
                            `${k}=${typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(4)) : v}`,
                        )
                        .join(', ')}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {subTab === 'analytics' && (
        <Suspense fallback={<p style={{ padding: 12, opacity: 0.6 }}>Loading analytics…</p>}>
          <HPTunerAnalytics
            history={results}
            tunableParams={data.tunableParams ?? []}
            scoringMetric={String(data.hyperparams.scoring_metric || 'r2')}
          />
        </Suspense>
      )}
    </div>
  );
}

// ─── Output Panel Component ─────────────────────────────────────────────────

export default function OutputPanel() {
  const {
    outputPanelOpen,
    outputPanelHeight,
    logMessages,
    toggleOutputPanel,
    setOutputPanelHeight,
    clearLogs,
  } = useStore();

  useLogStream();

  const [activeTab, setActiveTab] = useState<'logs' | 'hp-results'>('logs');
  const scrollRef = useRef<HTMLDivElement>(null);
  const isResizing = useRef(false);
  const autoScroll = useRef(true);

  // Auto-scroll to bottom when new messages arrive (if user hasn't scrolled up)
  useEffect(() => {
    if (activeTab === 'logs' && autoScroll.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logMessages, activeTab]);

  // Track whether user has scrolled away from bottom
  const handleScroll = useCallback(() => {
    if (!scrollRef.current) return;
    const el = scrollRef.current;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    autoScroll.current = atBottom;
  }, []);

  // Resize handle (drag to resize vertically)
  const onResizeMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      isResizing.current = true;

      const startY = e.clientY;
      const startHeight = outputPanelHeight;

      const onMouseMove = (ev: MouseEvent) => {
        if (!isResizing.current) return;
        const delta = startY - ev.clientY;
        setOutputPanelHeight(startHeight + delta);
      };

      const onMouseUp = () => {
        isResizing.current = false;
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      };

      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
      document.body.style.cursor = 'row-resize';
      document.body.style.userSelect = 'none';
    },
    [outputPanelHeight, setOutputPanelHeight],
  );

  // Colorize log levels
  const colorize = (line: string): string => {
    if (line.includes('ERROR')) return 'output-panel__line--error';
    if (line.includes('WARNING')) return 'output-panel__line--warning';
    if (line.includes('DEBUG')) return 'output-panel__line--debug';
    return '';
  };

  return (
    <div className="output-panel-wrapper">
      {/* Toggle bar (always visible) */}
      <div className="output-panel__toggle-bar" onClick={toggleOutputPanel}>
        <span className="output-panel__toggle-icon">
          {outputPanelOpen ? '▼' : '▲'}
        </span>
        <span className="output-panel__toggle-label">Output</span>
        <span className="output-panel__toggle-count">
          {logMessages.length > 0 && `(${logMessages.length})`}
        </span>
        <span className="output-panel__toggle-spacer" />
        {outputPanelOpen && (
          <button
            className="output-panel__clear-btn"
            onClick={(e) => {
              e.stopPropagation();
              clearLogs();
            }}
            title="Clear output"
          >
            🗑
          </button>
        )}
      </div>

      {/* Expandable panel */}
      {outputPanelOpen && (
        <>
          <div className="output-panel__resize-handle" onMouseDown={onResizeMouseDown} />

          {/* Tab bar */}
          <div className="output-panel__tab-bar">
            <button
              className={`output-panel__tab${activeTab === 'logs' ? ' output-panel__tab--active' : ''}`}
              onClick={() => setActiveTab('logs')}
            >
              Logs
            </button>
            <button
              className={`output-panel__tab${activeTab === 'hp-results' ? ' output-panel__tab--active' : ''}`}
              onClick={() => setActiveTab('hp-results')}
            >
              HP Tuning
            </button>
          </div>

          {activeTab === 'logs' && (
            <div
              className="output-panel__body"
              style={{ height: outputPanelHeight }}
              ref={scrollRef}
              onScroll={handleScroll}
            >
              {logMessages.length === 0 ? (
                <div className="output-panel__empty">
                  No output yet. Run a pipeline to see backend logs here.
                </div>
              ) : (
                logMessages.map((msg, i) => (
                  <div key={i} className={`output-panel__line ${colorize(msg)}`}>
                    {msg}
                  </div>
                ))
              )}
            </div>
          )}

          {activeTab === 'hp-results' && (
            <div className="output-panel__body" style={{ height: outputPanelHeight }}>
              <HPTuningResultsPanel />
            </div>
          )}
        </>
      )}
    </div>
  );
}
