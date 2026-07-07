import { defineConfig } from 'vite';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
let ortPkgPath = '';
try { ortPkgPath = require.resolve('onnxruntime-web/package.json'); } catch {}
const ortPkgRoot = ortPkgPath ? path.dirname(ortPkgPath) : path.resolve(__dirname, '../../node_modules/onnxruntime-web');
const ortDistFs = path.join(ortPkgRoot, 'dist');
const ORT_WASM_BASE_DEV = `/@fs/${ortDistFs}/`;

import fs from 'node:fs';
import pathModule from 'node:path';

/**
 * Vite plugin to serve .onnx files as raw binary with proper MIME type.
 * ONNX Runtime Web requires binary files to be served as application/octet-stream.
 * This plugin bypasses Vite's asset transformation to prevent binary corruption.
 */
function onnxMimePlugin() {
  return {
    name: 'onnx-mime-plugin',
    configureServer(server) {
      // Middleware runs BEFORE Vite's default file serving to short-circuit .onnx requests
      server.middlewares.use((req, res, next) => {
        if (req.url && req.url.endsWith('.onnx')) {
          // Resolve the file path relative to the project root (examples/vanilla-worker)
          const projectRoot = pathModule.resolve(__dirname);
          // Strip query params if any, then resolve path
          const urlPath = req.url.split('?')[0].replace(/^\//, '');
          const filePath = pathModule.resolve(projectRoot, urlPath);

          console.log(`[onnx-mime] Request: ${req.url}`);
          console.log(`[onnx-mime] Resolved path: ${filePath}`);

          if (fs.existsSync(filePath)) {
            const stat = fs.statSync(filePath);
            console.log(`[onnx-mime] Serving ${filePath} (${stat.size} bytes)`);
            res.setHeader('Content-Type', 'application/octet-stream');
            res.setHeader('Content-Length', stat.size);
            res.setHeader('Cache-Control', 'no-cache');
            const readStream = fs.createReadStream(filePath);
            readStream.on('error', (err) => {
              console.error(`[onnx-mime] Stream error: ${err.message}`);
              if (!res.headersSent) {
                res.statusCode = 500;
                res.end('File read error');
              }
            });
            readStream.pipe(res);
            return; // Short-circuit - don't call next()
          } else {
            console.log(`[onnx-mime] File NOT found at ${filePath}, falling through to Vite`);
          }
        }
        next();
      });
    },
  };
}

export default defineConfig(({ mode }) => ({
  root: '.',
  base: process.env.BASE_PATH && mode === 'production' ? process.env.BASE_PATH : '/',
  plugins: [onnxMimePlugin()],
  server: {
    port: 5173,
    open: '/',
    fs: { allow: [path.resolve(__dirname, '../..')] },
  },
  worker: {
    format: 'es',
  },
  optimizeDeps: {
    exclude: ['web-txt2img'],
  },
  build: {
    // production: minify=true, sourcemap=false
    // development: minify=false, sourcemap=false
    // debug: minify=false, sourcemap='inline'
    minify: mode === 'production',
    sourcemap: mode === 'debug',
  },
  define: {
    __ORT_WASM_BASE_DEV__: JSON.stringify(ORT_WASM_BASE_DEV),
    __MODE__: JSON.stringify(mode),
  },
}));
