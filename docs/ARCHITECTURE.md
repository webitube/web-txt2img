# Architecture

Module-level architecture of **web-txt2img** — a browser-only text-to-image library using WebGPU-accelerated AI models.

---

## Table of Contents

- [System Overview](#system-overview)
- [Module Architecture](#module-architecture)
- [Worker System](#worker-system)
- [Model Adapter System](#model-adapter-system)
- [Communication Protocol](#communication-protocol)
- [Data Flow](#data-flow)
- [State Management](#state-management)
- [Dependency Injection](#dependency-injection)
- [Cache Layer](#cache-layer)

---

## System Overview

```mermaid
graph TB
    subgraph "Browser (Main Thread)"
        App["Application Code"]
        Client["Txt2ImgWorkerClient"]
    end

    subgraph "Web Worker Thread"
        Host["Worker Host"]
        Protocol["Message Protocol"]
    end

    subgraph "Worker: Model Adapters"
        SD["SD-Turbo Adapter"]
        Janus["Janus-Pro-1B Adapter"]
    end

    subgraph "External Runtimes"
        ORT["ONNX Runtime Web"]
        TF["Transformers.js"]
        HF["HuggingFace CDN"]
    end

    subgraph "Browser APIs"
        WebGPU["WebGPU"]
        Cache["Cache Storage"]
    end

    App --> Client
    Client -->|postMessage| Host
    Host -->|postMessage| Client
    Host --> Protocol
    Protocol --> SD
    Protocol --> Janus
    SD --> ORT
    Janus --> TF
    ORT --> WebGPU
    TF --> WebGPU
    ORT --> HF
    TF --> HF
    ORT --> Cache
    TF --> Cache
```

The library runs entirely in the browser. All heavy computation (model loading, inference) is offloaded to a Web Worker thread to keep the main thread responsive.

---

## Module Architecture

```mermaid
graph TB
    subgraph "packages/web-txt2img/src/"
        subgraph "Public API"
            Index["index.ts<br/>Entry Point"]
        end

        subgraph "Core Modules"
            Types["types.ts<br/>Type Definitions"]
            Registry["registry.ts<br/>Model Registry"]
            Capabilities["capabilities.ts<br/>Feature Detection"]
            Cache["cache.ts<br/>Cache Storage"]
        end

        subgraph "Worker System"
            Client["worker/client.ts<br/>Main Thread Client"]
            Host["worker/host.ts<br/>Worker Thread"]
            Protocol["worker/protocol.ts<br/>Message Types"]
        end

        subgraph "Adapters"
            SD["adapters/sd-turbo.ts<br/>ONNX Runtime"]
            Janus["adapters/janus-pro.ts<br/>Transformers.js"]
        end
    end

    Index --> Types
    Index --> Registry
    Index --> Capabilities
    Index --> Cache
    Index --> Client
    Client --> Protocol
    Host --> Protocol
    Host --> Index
    Registry --> SD
    Registry --> Janus
    SD --> Cache
    Janus --> Cache
```

### Module Responsibilities

| Module | Path | Responsibility |
|---|---|---|
| **index.ts** | `src/index.ts` | Public API entry point; exports all functions and types |
| **types.ts** | `src/types.ts` | Core type definitions: `ModelId`, `BackendId`, `Adapter`, `GenerateParams`, etc. |
| **registry.ts** | `src/registry.ts` | Model metadata, factory functions, backend preferences |
| **capabilities.ts** | `src/capabilities.ts` | Browser feature detection (WebGPU, WASM, shader-f16) |
| **cache.ts** | `src/cache.ts` | Cache Storage wrapper for model assets |
| **client.ts** | `src/worker/client.ts` | Main-thread client for worker communication |
| **host.ts** | `src/worker/host.ts` | Worker thread: message routing, job queue, lifecycle |
| **protocol.ts** | `src/worker/protocol.ts` | Request/response type definitions for worker messages |
| **sd-turbo.ts** | `src/adapters/sd-turbo.ts` | SD-Turbo model adapter using ONNX Runtime Web |
| **janus-pro.ts** | `src/adapters/janus-pro.ts` | Janus-Pro-1B adapter using Transformers.js |

---

## Worker System

### Architecture

```mermaid
sequenceDiagram
    participant App as Application
    participant Client as Txt2ImgWorkerClient
    participant Host as Worker Host
    participant Adapter as Model Adapter

    App->>Client: load('sd-turbo', options)
    Client->>Host: postMessage({ kind: 'load', model: 'sd-turbo' })
    Host->>Adapter: loadModel('sd-turbo', options)
    Adapter-->>Host: LoadResult
    Host-->>Client: postMessage({ type: 'result', ok: true })
    Client-->>App: Promise resolves

    App->>Client: generate({ prompt: '...' })
    Client->>Host: postMessage({ kind: 'generate', params })
    Host->>Adapter: generateImage(params)
    Adapter-->>Host: progress events
    Host-->>Client: postMessage({ type: 'progress' })
    Client-->>App: onProgress callback
    Adapter-->>Host: GenerateResult
    Host-->>Client: postMessage({ type: 'result', blob })
    Client-->>App: Promise resolves
```

### Worker Lifecycle

```mermaid
stateDiagram-v2
    [*] --> idle
    idle --> running: generate() called
    running --> idle: generation complete
    running --> aborting: abort() called
    aborting --> idle: abort successful
    aborting --> running: abort timeout, continue
    idle --> queued: generate() while busy
    queued --> running: previous job finishes
    queued --> idle: superseded by newer request
```

### Job Queue Policy

The worker maintains a **single-flight, single-slot queue**:

1. **Idle** → Start generation immediately
2. **Busy + `reject`** → Return error immediately
3. **Busy + `queue`** → Queue the request; run after current job finishes
4. **Busy + `abort_and_queue`** → Abort current job; queue new request
5. **Debouncing** → Delay start by `debounceMs` to coalesce rapid inputs

---

## Model Adapter System

### Adapter Interface

All adapters implement the `Adapter` interface from `types.ts`:

```typescript
interface Adapter {
  readonly id: ModelId;
  checkSupport(capabilities: Capabilities): BackendId[];
  load(options: LoadOptions): Promise<LoadResult>;
  isLoaded(): boolean;
  generate(params: Omit<GenerateParams, 'model'>): Promise<GenerateResult>;
  unload(): Promise<void>;
  purgeCache(): Promise<void>;
}
```

### Registry Pattern

```mermaid
classDiagram
    class Adapter {
        <<interface>>
        +id: ModelId
        +checkSupport(capabilities) BackendId[]
        +load(options) Promise~LoadResult~
        +isLoaded() boolean
        +generate(params) Promise~GenerateResult~
        +unload() Promise~void~
        +purgeCache() Promise~void~
    }

    class RegistryEntry {
        +id: ModelId
        +displayName: string
        +task: string
        +supportedBackends: BackendId[]
        +sizeBytesApprox: number
        +createAdapter() Adapter
    }

    class SDTurboAdapter {
        -loaded: boolean
        -ort: ORT
        -sessions: object
        +load() Promise~LoadResult~
        +generate() Promise~GenerateResult~
    }

    class JanusProAdapter {
        -loaded: boolean
        -hf: HF
        -processor: any
        -model: any
        +load() Promise~LoadResult~
        +generate() Promise~GenerateResult~
    }

    Adapter <|-- SDTurboAdapter
    Adapter <|-- JanusProAdapter
    RegistryEntry ..> Adapter : creates
```

### Model Comparison

| Feature | SD-Turbo | Janus-Pro-1B |
|---|---|---|
| **Runtime** | ONNX Runtime Web | Transformers.js |
| **Backend** | WebGPU (WASM experimental) | WebGPU only |
| **Model Size** | ~2.34 GB | ~2.25 GB |
| **Seed Support** | Yes | No |
| **Resolution** | 512×512 fixed | Variable |
| **Inference Steps** | 1 (turbo) | Autoregressive |

---

## Communication Protocol

### Message Flow

```mermaid
sequenceDiagram
    participant M as Main Thread
    participant W as Worker Thread

    M->>W: WorkerRequest { id, kind, ... }
    W->>W: Route by kind
    alt success
        W->>W: Execute operation
        W-->>M: WorkerResponse { id, type: 'result', ok: true, ... }
    else error
        W-->>M: WorkerResponse { id, type: 'result', ok: false, reason, ... }
    end

    Note over M,W: Progress events for long operations
    W-->>M: WorkerProgress { id, type: 'progress', event }
    W-->>M: WorkerState { type: 'state', value }
```

### Request Types

| Kind | Direction | Purpose |
|---|---|---|
| `detect` | → Worker | Detect browser capabilities |
| `listModels` | → Worker | List available models |
| `listBackends` | → Worker | List available backends |
| `load` | → Worker | Load a model into memory |
| `unload` | → Worker | Unload current model |
| `purge` | → Worker | Purge model cache |
| `purgeAll` | → Worker | Purge all caches |
| `generate` | → Worker | Generate image from prompt |
| `abort` | → Worker | Abort current generation |

### Response Types

| Type | Description |
|---|---|
| `result` | Final result (success or error) |
| `progress` | Progress update during operation |
| `accepted` | Acknowledgment for queued operations |
| `state` | Worker state change notification |

---

## Data Flow

### Image Generation Flow

```mermaid
flowchart TD
    A["User calls generate prompt"] --> B["Client sends request to Worker"]
    B --> C{Worker busy?}
    C -->|No| D["Start generation immediately"]
    C -->|Yes| E{Busy Policy}
    E -->|reject| F["Return error"]
    E -->|queue| G["Queue request"]
    E -->|abort_and_queue| H["Abort current, queue new"]
    G --> I["Wait for current job"]
    I --> D
    H --> J["Abort timeout check"]
    J --> D

    D --> K["Adapter.generate()"]
    K --> L["Tokenize prompt"]
    L --> M["Run model inference"]
    M --> N["Decode to image"]
    N --> O["Return Blob"]
    O --> P["Worker posts result"]
    P --> Q["Client resolves promise"]
    Q --> R["User receives image"]
```

### Model Loading Flow

```mermaid
flowchart TD
    A["User calls load model"] --> B["Client sends load request"]
    B --> C["Worker routes to load handler"]
    C --> D{Model already loaded?}
    D -->|Yes| E["Reject with busy"]
    D -->|No| F["Resolve adapter from registry"]
    F --> G["Check backend support"]
    G --> H{Backend available?}
    H -->|No| I["Reject with backend_unavailable"]
    H -->|Yes| J["Download model assets"]
    J --> K["Cache in Cache Storage"]
    K --> L["Initialize runtime sessions"]
    L --> M["Set loaded flag"]
    M --> N["Post success result"]
```

---

## State Management

### Worker State

The worker maintains these state variables:

```typescript
// Current execution
let currentJob: CurrentJob | null = null;
let pendingJob: PendingJob | null = null;
let aborting = false;

// Model state
let loadedModel: ModelId | null = null;
let loadInFlight = false;

// Timers
let abortTimer: number | null = null;
let debounceTimer: number | null = null;
```

### Adapter State

Each adapter tracks:

```typescript
// Per-adapter state
private loaded = false;
private backendUsed: BackendId | null = null;
// Runtime-specific handles (ORT sessions, Transformers.js model, etc.)
```

---

## Dependency Injection

The library supports dependency injection for runtime flexibility:

```mermaid
graph LR
    subgraph "Injection Points"
        A["onnxruntime-web instance"]
        B["Tokenizer provider function"]
        C["WASM asset paths"]
        D["Model base URL override"]
    end

    subgraph "Load Options"
        E["LoadOptions.ort"]
        F["LoadOptions.tokenizerProvider"]
        G["LoadOptions.wasmPaths"]
        H["LoadOptions.modelBaseUrl"]
    end

    A --> E
    B --> F
    C --> G
    D --> H
```

### Resolution Order

For each dependency, the adapter tries:

1. **Injected value** — Passed via `LoadOptions`
2. **Dynamic import** — `import('onnxruntime-web')` etc.
3. **Global fallback** — `globalThis.ort`, `globalThis.transformers`

---

## Cache Layer

### Cache Architecture

```mermaid
graph TD
    subgraph "Cache Storage"
        Cache1["web-txt2img-v1"]
    end

    subgraph "URL Tracker"
        Tracker["Map~ModelId, Set~URL~~"]
    end

    Fetch["fetchWithCache()"] --> Cache1
    Fetch --> Tracker
    Purge["purgeModelCache()"] --> Tracker
    Purge --> Cache1
    PurgeAll["purgeAllCaches()"] --> Cache1
```

### Cache Behavior

- **Name**: `web-txt2img-v1` (Cache Storage API)
- **Strategy**: Cache-first with network fallback
- **Scope**: Per-model URL tracking for targeted purging
- **Progress**: Streaming downloads report byte-level progress
- **Fallback**: Simulated progress when streaming is unavailable

### Cache Operations

| Function | Purpose |
|---|---|
| `fetchWithCache()` | Fetch with Cache Storage caching |
| `fetchArrayBufferWithCacheProgress()` | Fetch binary with progress reporting |
| `purgeModelCache(id)` | Delete cached assets for specific model |
| `purgeAllCaches()` | Delete all cached assets |
| `noteModelUrl(id, url)` | Track URL association with model |
