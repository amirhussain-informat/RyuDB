import { useCallback, useEffect, useRef, useState } from "react";
import { RyuDBClient, type ConnStatus } from "../lib/client";
import type { Request, Result } from "../lib/types";

/** Owns the RyuDBClient + connection lifecycle and exposes the op helpers as
 * memoized callbacks. The UI reads `status` to render the connection badge. */
export function useServer() {
  const [status, setStatus] = useState<ConnStatus>("idle");
  const clientRef = useRef<RyuDBClient | null>(null);

  const connect = useCallback(async (url: string) => {
    clientRef.current?.disconnect();
    const c = new RyuDBClient(url);
    c.onStatus = setStatus;
    clientRef.current = c;
    await c.connect();
  }, []);

  const disconnect = useCallback(() => {
    clientRef.current?.disconnect();
  }, []);

  const op = useCallback(async (req: Request): Promise<Result> => {
    const c = clientRef.current;
    if (!c) throw new Error("not connected");
    return c.request(req);
  }, []);

  /** Upload a parquet file from the browser (two-frame text+binary ingest).
   *  `maxBytes` is pre-checked client-side so an oversized file is refused
   *  before sending (the transport would otherwise close with 1009). */
  const upload = useCallback(async (
    name: string, bytes: Uint8Array, maxBytes?: number,
  ): Promise<Result> => {
    const c = clientRef.current;
    if (!c) throw new Error("not connected");
    return c.upload(name, bytes, "parquet", maxBytes);
  }, []);

  // Tear down on unmount.
  useEffect(() => () => clientRef.current?.disconnect(), []);

  return { status, connect, disconnect, op, upload };
}