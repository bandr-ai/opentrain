import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built into the Python package; the OpenTrain server serves it at /.
// `pnpm dev` proxies the API to a running server (opentrain serve, default port 7477).
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: { outDir: "../src/opentrain/web/app", emptyOutDir: true, assetsDir: "assets", sourcemap: false },
  server: {
    proxy: {
      "/api": { target: process.env.OPENTRAIN_URL || "http://127.0.0.1:7477", changeOrigin: true },
      "/auth": { target: process.env.OPENTRAIN_URL || "http://127.0.0.1:7477", changeOrigin: true },
    },
  },
});
