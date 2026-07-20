import { useState } from "react";
import type { ConnStatus } from "../lib/client";

interface Props {
  url: string;
  status: ConnStatus;
  running: boolean;
  onConnect: (url: string) => void;
  onDisconnect: () => void;
  onRun: () => void;
  onExplain: () => void;
  onCancel: () => void;
}

const STATUS_LABEL: Record<ConnStatus, string> = {
  idle: "disconnected",
  connecting: "connecting…",
  open: "connected",
  closed: "disconnected",
  error: "error",
};

export default function Toolbar({
  url, status, running, onConnect, onDisconnect, onRun, onExplain, onCancel,
}: Props) {
  const [field, setField] = useState(url);
  const connected = status === "open";

  return (
    <div className="toolbar">
      <div className="conn">
        <span className={`dot ${status}`} title={STATUS_LABEL[status]} />
        <input
          className="url"
          value={field}
          onChange={(e) => setField(e.target.value)}
          disabled={connected}
          placeholder="ws://127.0.0.1:5430"
        />
        {connected ? (
          <button onClick={onDisconnect}>Disconnect</button>
        ) : (
          <button onClick={() => onConnect(field)}>Connect</button>
        )}
      </div>
      <div className="run-group">
        <button className="primary" onClick={onRun} disabled={!connected || running} title="Ctrl/Cmd+Enter">
          {running ? "Running…" : "Run"}
        </button>
        <button onClick={onExplain} disabled={!connected || running}>Explain</button>
        {running && <button onClick={onCancel}>Cancel</button>}
      </div>
    </div>
  );
}