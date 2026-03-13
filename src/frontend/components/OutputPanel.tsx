import { useEffect, useRef, useCallback } from 'react';
import useStore from '../store';

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

  const scrollRef = useRef<HTMLDivElement>(null);
  const isResizing = useRef(false);
  const autoScroll = useRef(true);

  // Auto-scroll to bottom when new messages arrive (if user hasn't scrolled up)
  useEffect(() => {
    if (autoScroll.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logMessages]);

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
        </>
      )}
    </div>
  );
}
