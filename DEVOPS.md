# DEVOPS — Build & Deployment Guide

## Build Modes

| Mode | Minification | Source Maps | Use Case |
|------|-------------|-------------|----------|
| **development** | Off | Off | Local development, quick iteration |
| **debug** | Off | Inline | Debugging runtime issues, stack traces |
| **production** | On | Off | Deployment, optimized bundle size |

## Prerequisites

- **Node.js** >= 18
- **npm** >= 9

```powershell
# Install workspace dependencies
npm install
```

## Build Commands

### Library (`packages/web-txt2img`)

```powershell
# Build the library (TypeScript compilation)
npm run build:lib

# Type-check without emitting
npm run typecheck

# Run unit tests (Jest + ts-jest)
npm test

# Run tests with watch mode
npm run test:watch
```

### Example App (`examples/vanilla-worker`)

#### Development (Dev Server)

```powershell
# Start Vite dev server (no minification, HMR)
npm run dev:vanilla

# Start with debug mode (no minification, inline source maps)
npm run dev:vanilla:debug
```

#### Production Build

```powershell
# Build for production (minified, optimized)
npm run build:vanilla:prod

# Build for development (no minification, no source maps)
npm run build:vanilla:dev
```

## Build Output

| Mode | Output Directory | Characteristics |
|------|-----------------|-----------------|
| `build:vanilla:dev` | `examples/vanilla-worker/dist/` | Unminified JS, no source maps |
| `build:vanilla:prod` | `examples/vanilla-worker/dist/` | Minified JS, tree-shaken, optimized |
| `dev:vanilla:debug` | Served in-memory by Vite | Inline source maps, readable stack traces |

## Vite Configuration

The Vite config (`examples/vanilla-worker/vite.config.ts`) is mode-aware:

```typescript
build: {
  minify: mode === 'production',      // true only for production
  sourcemap: mode === 'debug',        // true only for debug
}
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `BASE_PATH` | Base URL for production builds | `/` |

Example:
```powershell
# Build with custom base path for subdirectory deployment
$env:BASE_PATH="/txt2img/"; npm run build:vanilla:prod
```

## Workflow

### Local Development
```powershell
npm run dev:vanilla
# → http://localhost:5173
```

### Debugging Runtime Issues
```powershell
npm run dev:vanilla:debug
# → Inline source maps for readable stack traces
```

### Production Deployment
```powershell
npm run build:vanilla:prod
# → Serve examples/vanilla-worker/dist/ with any static file server
```

## Clean Build

```powershell
# Remove all build artifacts
npm run clean

# Full rebuild
npm run clean && npm run build:vanilla:prod
```

## Notes

- The library (`packages/web-txt2img`) is always compiled via TypeScript (`tsc`) — it does not use Vite bundling.
- The example app (`examples/vanilla-worker`) uses Vite for bundling and serves as the reference implementation.
- ONNX Runtime Web assets (`.wasm`, `.data` files) are copied automatically by the `predev`/`prebuild` hooks.
- The scheduler module (`src/scheduler/`) contains 9 schedulers, 5 sigma schedule types, noise utilities, and preset configurations. Tests are in `src/scheduler/__tests__/`.
