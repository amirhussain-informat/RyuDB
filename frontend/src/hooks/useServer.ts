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

  // Tear down on unmount.
  useEffect(() => () => clientRef.current?.disconnect(), []);

  return { status, connect, disconnect, op };
}