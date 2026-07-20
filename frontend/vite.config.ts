import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The worksheet UI talks to ryudb-server (default 127.0.0.1:5430) over a
    // raw WebSocket; no Vite dev proxy is needed because the server accepts
    // WebSocket connections directly and the browser connects by absolute URL.
  },
  build: { outDir: "dist", sourcemap: true },
});