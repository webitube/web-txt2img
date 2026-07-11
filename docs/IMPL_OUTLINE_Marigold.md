# Marigold Browser Implementation Outline

## Overview

This document outlines how to port Marigold (monocular depth estimation via diffusion) to run in-browser using the web-txt2img architecture pattern: WebGPU-accelerated ONNX inference, worker-based execution, and type-safe client API.

---

## Phase 1: Foundation — Types and Registry

### 1.1 Extend Model Types

Add Marigold model IDs and new task types to the shared type system.

```typescript
// packages/web-txt2img/src/types.ts

export type ModelId = 'sd-turbo' | 'janus-pro-1b' | 'marigold-depth-v1-1' | 'marigold-normals-v1-1';

export type TaskType = 'text-to-image' | 'depth-estimation' | 'normal-estimation';

export interface ModelInfo {
  id: ModelId;
  displayName: string;
  task: TaskType;
  supportedBackends: BackendId[];
  notes?: string;
  sizeBytesApprox?: number;
  sizeGBApprox?: number;
  sizeNotes?: string;
}
```

### 1.2 Marigold-Specific Generate Params

Depth estimation takes an image input, not a text prompt.

```typescript
// packages/web-txt2img/src/types.ts

export interface MarigoldGenerateParams {
  model: ModelId;
  inputImage: ImageBitmap | HTMLImageElement | HTMLCanvasElement | OffscreenCanvas;
  denoisingSteps?: number;         // default: 4 for v1.1
  ensembleSize?: number;           // default: 1 (0–5 recommended in-browser)
  processingRes?: number;          // default: 768
  matchInputRes?: boolean;         // default: true
  onProgress?: (event: MarigoldProgressEvent) => void;
  signal?: AbortSignal;
}

export type MarigoldProgressPhase =
  | 'preprocessing'
  | 'encoding'
  | 'denoising'
  | 'decoding'
  | 'ensembling'
  | 'postprocessing'
  | 'complete';

export interface MarigoldProgressEvent {
  phase: MarigoldProgressPhase;
  pct?: number;
  step?: number;
  totalSteps?: number;
  ensembleIndex?: number;
  ensembleTotal?: number;
}

export type MarigoldGenerateResult =
  | { ok: true; blob: Blob; depthMap: Float32Array; width: number; height: number; timeMs: number }
  | { ok: false; reason: ErrorCode; message?: string };
```

### 1.3 Registry Entry

```typescript
// packages/web-txt2img/src/registry.ts

import { MarigoldDepthAdapter } from './adapters/marigold-depth.js';

// In REGISTRY array:
{
  id: 'marigold-depth-v1-1',
  displayName: 'Marigold Depth v1.1 (ONNX Runtime WebGPU)',
  task: 'depth-estimation',
  supportedBackends: ['webgpu'],
  notes: 'Monocular depth estimation via diffusion. Ensemble size 1–5 recommended.',
  sizeBytesApprox: 2600 * 1024 * 1024,
  sizeGBApprox: 2.6,
  sizeNotes: 'UNet ~1.6GB, VAE encoder+decoder ~500MB, CLIP text encoder ~500MB',
  createAdapter: () => new MarigoldDepthAdapter(),
},
```

---

## Phase 2: Marigold Adapter Implementation

### 2.1 Adapter Skeleton

```typescript
// packages/web-txt2img/src/adapters/marigold-depth.ts

import type {
  Adapter,
  BackendId,
  Capabilities,
  ErrorCode,
  LoadOptions,
  LoadProgress,
  ModelId,
} from '../types.js';
import type {
  MarigoldGenerateParams,
  MarigoldGenerateResult,
  MarigoldProgressEvent,
} from '../types.js';
import { InferenceSession, Tensor, ImageBitmapInput } from 'onnxruntime-web/webgpu';

export class MarigoldDepthAdapter implements Adapter {
  readonly id: ModelId = 'marigold-depth-v1-1';

  private _unetSession: InferenceSession | null = null;
  private _vaeEncoderSession: InferenceSession | null = null;
  private _vaeDecoderSession: InferenceSession | null = null;
  private _clipSession: InferenceSession | null = null;
  private _emptyTextEmbed: Tensor | null = null;
  private _loaded = false;

  // Latent scale factor from Marigold
  private static readonly LATENT_SCALE_FACTOR = 0.1821;

  checkSupport(capabilities: Capabilities): BackendId[] {
    return capabilities.webgpu ? ['webgpu'] : [];
  }

  isLoaded(): boolean {
    return this._loaded;
  }

  async load(options: LoadOptions & { backendPreference: BackendId[] }): Promise<LoadResult> {
    // ... implementation below
  }

  async generate(params: MarigoldGenerateParams): Promise<MarigoldGenerateResult> {
    // ... implementation below
  }

  async unload(): Promise<void> {
    await this._unetSession?.dispose();
    await this._vaeEncoderSession?.dispose();
    await this._vaeDecoderSession?.dispose();
    await this._clipSession?.dispose();
    this._unetSession = null;
    this._vaeEncoderSession = null;
    this._vaeDecoderSession = null;
    this._clipSession = null;
    this._emptyTextEmbed = null;
    this._loaded = false;
  }

  async purgeCache(): Promise<void> {
    // Clear Cache Storage entries for Marigold models
    for (const key of await caches.keys()) {
      if (key.includes('marigold')) {
        await caches.delete(key);
      }
    }
  }
}
```

### 2.2 Model Loading

Load ONNX sessions for UNet, VAE, and CLIP. Use the same dynamic import + cache storage pattern as SD-Turbo.

```typescript
async load(options: LoadOptions & { backendPreference: BackendId[] }): Promise<LoadResult> {
  const onProgress = options.onProgress;
  const reportProgress = (p: LoadProgress) => onProgress?.(p);

  try {
    const modelBaseUrl = options.modelBaseUrl ?? 'https://huggingface.co/prs-eth/marigold-depth-v1-1/resolve/main/';

    // --- Load UNet ---
    reportProgress({ phase: 'loading', message: 'Loading UNet...', pct: 0 });
    const unetPath = await this._resolveModelPath(modelBaseUrl, 'unet/model.onnx');
    this._unetSession = await InferenceSession.create(unetPath, {
      executionProviders: ['webgpu'],
    });
    reportProgress({ phase: 'loading', message: 'UNet loaded', pct: 30 });

    // --- Load VAE Encoder ---
    reportProgress({ phase: 'loading', message: 'Loading VAE encoder...', pct: 30 });
    const vaeEncoderPath = await this._resolveModelPath(modelBaseUrl, 'vae_encoder/model.onnx');
    this._vaeEncoderSession = await InferenceSession.create(vaeEncoderPath, {
      executionProviders: ['webgpu'],
    });
    reportProgress({ phase: 'loading', message: 'VAE encoder loaded', pct: 50 });

    // --- Load VAE Decoder ---
    reportProgress({ phase: 'loading', message: 'Loading VAE decoder...', pct: 50 });
    const vaeDecoderPath = await this._resolveModelPath(modelBaseUrl, 'vae_decoder/model.onnx');
    this._vaeDecoderSession = await InferenceSession.create(vaeDecoderPath, {
      executionProviders: ['webgpu'],
    });
    reportProgress({ phase: 'loading', message: 'VAE decoder loaded', pct: 70 });

    // --- Load CLIP Text Encoder ---
    reportProgress({ phase: 'loading', message: 'Loading CLIP text encoder...', pct: 70 });
    const clipPath = await this._resolveModelPath(modelBaseUrl, 'text_encoder/model.onnx');
    this._clipSession = await InferenceSession.create(clipPath, {
      executionProviders: ['webgpu'],
    });

    // Pre-compute empty text embedding
    await this._encodeEmptyText();
    reportProgress({ phase: 'loading', message: 'CLIP loaded, empty embed cached', pct: 95 });

    this._loaded = true;
    reportProgress({ phase: 'loading', message: 'Marigold ready', pct: 100 });

    return { ok: true, backendUsed: 'webgpu' };
  } catch (err) {
    return {
      ok: false,
      reason: 'internal_error',
      message: `Failed to load Marigold: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
}

private async _encodeEmptyText(): Promise<void> {
  // Tokenize empty string: CLIP tokenizer produces all-zero input IDs for empty input
  // CLIP ViT-B/32 expects shape [1, 1, 1024] output
  const inputIds = new Tensor('int64', [1, 1], new BigInt64Array([0]));
  const attentionMask = new Tensor('int64', [1, 1], new BigInt64Array([1]));

  const outputs = await this._clipSession!.run({
    input_ids: inputIds,
    attention_mask: attentionMask,
  });

  // Last hidden state: [1, 1, 768] — reshape to [1, 1, 768]
  this._emptyTextEmbed = new Tensor(
    'float32',
    [1, 1, 768],
    outputs.last_hidden_state.data
  );
}
```

### 2.3 Image Preprocessing

Convert input image to normalized tensor `[-1, 1]`, resize to processing resolution.

```typescript
private async _preprocessImage(
  inputImage: ImageBitmap | HTMLImageElement | HTMLCanvasElement | OffscreenCanvas,
  processingRes: number
): Promise<{ tensor: Float32Array; width: number; height: number; originalWidth: number; originalHeight: number }> {
  // Draw to offscreen canvas for uniform processing
  const canvas = new OffscreenCanvas(
    inputImage.width,
    inputImage.height
  );
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(inputImage, 0, 0);
  const bitmap = canvas.transferToImageBitmap();

  const originalWidth = bitmap.width;
  const originalHeight = bitmap.height;

  // Compute target resolution (preserve aspect ratio)
  const scale = processingRes > 0
    ? Math.min(processingRes / originalWidth, processingRes / originalHeight, 1)
    : 1;
  const targetWidth = Math.round(originalWidth * scale);
  const targetHeight = Math.round(originalHeight * scale);

  // Resize using canvas
  const resizedCanvas = new OffscreenCanvas(targetWidth, targetHeight);
  const resizedCtx = resizedCanvas.getContext('2d')!;
  resizedCtx.drawImage(bitmap, 0, 0, targetWidth, targetHeight);
  const resizedBitmap = resizedCanvas.transferToImageBitmap();

  // Convert to Float32Array [1, 3, H, W] normalized to [-1, 1]
  const { data } = createImageBitmap(resizedBitmap);
  // createImageBitmap returns RGBA — extract RGB
  const rgbData = new Float32Array(3 * targetHeight * targetWidth);
  for (let i = 0; i < targetHeight * targetWidth; i++) {
    rgbData[3 * i]     = (data[i * 4]     / 255.0) * 2.0 - 1.0;  // R
    rgbData[3 * i + 1] = (data[i * 4 + 1] / 255.0) * 2.0 - 1.0;  // G
    rgbData[3 * i + 2] = (data[i * 4 + 2] / 255.0) * 2.0 - 1.0;  // B
  }

  return {
    tensor: rgbData,
    width: targetWidth,
    height: targetHeight,
    originalWidth,
    originalHeight,
  };
}
```

### 2.4 VAE Encoding

```typescript
private async _encodeRgb(rgbTensor: Tensor): Promise<Tensor> {
  // VAE encoder: [B, 3, H, W] → [B, 8, H/4, W/4] (moments)
  const outputs = await this._vaeEncoderSession!.run({
    '0': rgbTensor,  // ONNX input name varies; inspect model for actual name
  });

  // Split moments into mean and log_var
  const moments = outputs['latent'];  // shape: [B, 8, h, w]
  const [mean, _logVar] = this._splitMoments(moments);

  // Scale latent
  const scaledData = new Float32Array(mean.dims.reduce((a, b) => a * b, 1));
  for (let i = 0; i < scaledData.length; i++) {
    scaledData[i] = mean.data[i] * MarigoldDepthAdapter.LATENT_SCALE_FACTOR;
  }

  return new Tensor('float32', mean.dims, scaledData);
}

private _splitMoments(moments: Tensor): [Float32Array, Float32Array] {
  // Moments tensor has 8 channels: first 4 = mean, last 4 = log_var
  const [_, channels, h, w] = moments.dims;
  const halfCh = channels / 2;
  const meanData = new Float32Array(halfCh * h * w);
  const logVarData = new Float32Array(halfCh * h * w);

  for (let d = 0; d < h * w; d++) {
    for (let c = 0; c < halfCh; c++) {
      meanData[d * halfCh + c] = moments.data[d * channels + c];
      logVarData[d * halfCh + c] = moments.data[d * channels + halfCh + c];
    }
  }

  return [meanData, logVarData];
}
```

### 2.5 Denoising Loop

The core diffusion loop. This is the most performance-critical section.

```typescript
private async _singleInfer(
  rgbLatent: Tensor,
  numSteps: number,
  onProgress: (pct: number, step: number) => void,
  signal?: AbortSignal
): Promise<Tensor> {
  const batchSize = rgbLatent.dims[0];
  const latentShape = [batchSize, 4, rgbLatent.dims[2], rgbLatent.dims[3]];

  // Generate random noise for target latent
  let targetLatentData = new Float32Array(latentShape.reduce((a, b) => a * b, 1));
  for (let i = 0; i < targetLatentData.length; i++) {
    targetLatentData[i] = Math.random() * 2 - 1;  // Uniform [-1, 1] as initial noise
  }
  let targetLatent = new Tensor('float32', latentShape, targetLatentData);

  // Batch empty text embedding
  const batchEmptyEmbed = this._repeatTensor(this._emptyTextEmbed!, batchSize);

  // DDIM scheduler state (simplified — full implementation needs alpha_cumprod schedule)
  const alphas = this._getAlphaCumprod(numSteps);

  for (let step = 0; step < numSteps; step++) {
    if (signal?.aborted) {
      throw new DOMException('Aborted', 'AbortError');
    }

    const t = alphas[step];

    // Concatenate [rgb_latent, target_latent] along channel dim
    const unetInput = await this._concatTensors(rgbLatent, targetLatent);

    // Run UNet
    const unetOutputs = await this._unetSession!.run({
      sample: unetInput,
      timestep: new Tensor('int64', [1], new BigInt64Array([BigInt(Math.round(t * 999))])),
      encoder_hidden_states: batchEmptyEmbed,
    });

    const noisePred = unetOutputs['sample'];  // [B, 4, h, w]

    // DDIM step: compute prev_sample
    targetLatent = this._ddimStep(noisePred, targetLatent, t, alphas, step);

    onProgress(((step + 1) / numSteps) * 100, step + 1);
  }

  return targetLatent;
}
```

### 2.6 Depth Decoding

```typescript
private async _decodeDepth(depthLatent: Tensor): Promise<Tensor> {
  // Unscla

private _ddimStep(
  noisePred: Tensor,
  currentLatent: Tensor,
  t: number,
  alphas: number[],
  step: number
): Tensor {
  // Simplified DDIM step — full implementation needs proper scheduler
  // x_{t-1} = sqrt(alpha_{t-1}) * (x_t - sqrt(1 - alpha_t) * epsilon) / sqrt(alpha_t)
  //          + sqrt(1 - alpha_{t-1}) * epsilon

  const alphaT = alphas[step];
  const alphaPrev = alphas[Math.min(step + 1, alphas.length - 1)];

  const result = new Float32Array(currentLatent.dims.reduce((a, b) => a * b, 1));

  for (let i = 0; i < result.length; i++) {
    const coeff1 = Math.sqrt(alphaPrev) * (currentLatent.data[i] - Math.sqrt(1 - alphaT) * noisePred.data[i]) / Math.sqrt(alphaT);
    const coeff2 = Math.sqrt(1 - alphaPrev) * noisePred.data[i];
    result[i] = coeff1 + coeff2;
  }

  return new Tensor('float32', currentLatent.dims, result);
}
```

> **Note:** For production, integrate a full DDIM scheduler port rather than the simplified version above. Consider porting the `diffusers` DDIMScheduler logic to TypeScript.

### 2.7 Main Generate Method

The orchestration method that ties preprocessing → encoding → denoising → decoding → postprocessing together.

```typescript
async generate(params: MarigoldGenerateParams): Promise<MarigoldGenerateResult> {
  const t0 = performance.now();

  if (!this._loaded) {
    return { ok: false, reason: 'model_not_loaded', message: 'Marigold model not loaded' };
  }

  const onProgress = params.onProgress;
  const reportProgress = (phase: MarigoldProgressPhase, pct?: number) => {
    onProgress?.({ phase, pct });
  };

  try {
    // --- Preprocessing ---
    reportProgress('preprocessing', 0);
    const processingRes = params.processingRes ?? 768;
    const { tensor: rgbData, width, height } = await this._preprocessImage(
      params.inputImage,
      processingRes
    );

    const rgbTensor = new Tensor('float32', [1, 3, height, width], rgbData);

    // --- VAE Encoding ---
    reportProgress('encoding', 10);
    const rgbLatent = await this._encodeRgb(rgbTensor);

    // --- Denoising ---
    reportProgress('denoising', 20);
    const numSteps = params.denoisingSteps ?? 4;
    const ensembleSize = params.ensembleSize ?? 1;

    let depthLatent: Tensor;

    if (ensembleSize > 1) {
      // Run multiple inferences with different random seeds
      const depthLatents: Tensor[] = [];

      for (let i = 0; i < ensembleSize; i++) {
        reportProgress('denoising', 20 + (i / ensembleSize) * 60);

        const latent = await this._singleInfer(
          rgbLatent,
          numSteps,
          (pct, step) => {
            reportProgress('denoising', 20 + (i / ensembleSize) * 60 + (pct / 100) * (60 / ensembleSize));
          },
          params.signal
        );
        depthLatents.push(latent);
      }

      // Ensemble: reduce via median (more robust to outliers)
      depthLatent = this._medianReduce(depthLatents);
    } else {
      // Single inference
      depthLatent = await this._singleInfer(
        rgbLatent,
        numSteps,
        (pct, step) => reportProgress('denoising', 20 + (pct / 100) * 60),
        params.signal
      );
    }

    // --- Depth Decoding ---
    reportProgress('decoding', 85);
    const depthOutput = await this._decodeDepth(depthLatent);

    // --- Postprocessing ---
    reportProgress('postprocessing', 95);

    // Sigmoid to get [0, 1] depth range
    const depthData = new Float32Array(depthOutput.dims.reduce((a, b) => a * b, 1));
    for (let i = 0; i < depthData.length; i++) {
      depthData[i] = 1 / (1 + Math.exp(-depthOutput.data[i]));  // sigmoid
    }

    // Resize to original input resolution if requested
    let finalDepth = depthData;
    let finalWidth = width;
    let finalHeight = height;

    if (params.matchInputRes) {
      const originalWidth = params.inputImage.width;
      const originalHeight = params.inputImage.height;
      ({ depth: finalDepth, width: finalWidth, height: finalHeight } =
        await this._resizeDepthMap(depthData, width, height, originalWidth, originalHeight));
    }

    // Create Blob for output
    const depthBlob = new Blob([finalDepth.buffer], { type: 'application/octet-stream' });

    const timeMs = performance.now() - t0;
    reportProgress('complete', 100);

    return {
      ok: true,
      blob: depthBlob,
      depthMap: finalDepth,
      width: finalWidth,
      height: finalHeight,
      timeMs,
    };
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      return { ok: false, reason: 'aborted', message: 'Generation aborted' };
    }
    return {
      ok: false,
      reason: 'internal_error',
      message: `Generation failed: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
}
```

### 2.8 Test-Time Ensembling (Advanced)

Marigold supports test-time ensembling for higher-quality results. For in-browser use, limit ensemble size to 1–5 due to memory constraints.

```typescript
private _medianReduce(latents: Tensor[]): Tensor {
  // Compute element-wise median across ensemble members
  const [batchSize, channels, h, w] = latents[0].dims;
  const totalElements = batchSize * channels * h * w;
  const result = new Float32Array(totalElements);

  for (let i = 0; i < totalElements; i++) {
    const values = latents.map(l => l.data[i]).sort();
    const mid = Math.floor(values.length / 2);
    result[i] = values.length % 2 !== 0
      ? values[mid]
      : (values[mid - 1] + values[mid]) / 2;
  }

  return new Tensor('float32', latents[0].dims, result);
}

// Optional: depth alignment via BFGS (for advanced ensembling)
// This is the most complex part of Marigold's ensembling pipeline
// For in-browser, consider skipping BFGS alignment and using raw median reduction
// which still provides quality improvements without the optimization overhead
```

> **In-browser recommendation:** Skip BFGS depth alignment for simplicity. Raw median reduction across ensemble members still improves quality significantly. BFGS alignment adds ~100ms+ per alignment step and requires porting an optimizer.

---

## Phase 3: Worker Protocol Integration

Add Marigold-specific message types to the worker protocol.

### 3.1 Protocol Messages

```typescript
// packages/web-txt2img/src/worker/protocol.ts

// New request type for Marigold
export interface MarigoldDepthRequest {
  type: 'marigold_depth';
  id: string;
  params: {
    inputImage: ArrayBuffer;  // Image data passed as ArrayBuffer
    inputWidth: number;
    inputHeight: number;
    denoisingSteps?: number;
    ensembleSize?: number;
    processingRes?: number;
    matchInputRes?: boolean;
  };
}

// New result type
export interface MarigoldDepthResult {
  type: 'marigold_depth_result';
  id: string;
  ok: boolean;
  depthMap?: Float32Array;
  width?: number;
  height?: number;
  timeMs?: number;
  reason?: ErrorCode;
  message?: string;
}

// Extend existing union types
export type HostRequest = LoadModelRequest | GenerateRequest | MarigoldDepthRequest | ...;
export type HostResult = LoadModelResult | GenerateResult | MarigoldDepthResult | ...;
```

### 3.2 Host Handler

```typescript
// In host.ts message handler

case 'marigold_depth': {
  const { params } = request as MarigoldDepthRequest;

  // Reconstruct ImageBitmap from ArrayBuffer
  // Note: this requires passing image data through a Blob → createImageBitmap
  const blob = new Blob([params.inputImage]);
  const imageBitmap = await createImageBitmap(
    new Blob([params.inputImage], { type: 'image/png' }),
    params.inputWidth,
    params.inputHeight
  );

  const result = await this._adapter.generate({
    model: 'marigold-depth-v1-1',
    inputImage: imageBitmap,
    denoisingSteps: params.denoisingSteps,
    ensembleSize: params.ensembleSize,
    processingRes: params.processingRes,
    matchInputRes: params.matchInputRes,
    onProgress: (event) => {
      this._postMessage({
        type: 'progress',
        id: request.id,
        phase: event.phase,
        pct: event.pct,
      });
    },
    signal: this._abortController?.signal,
  });

  if (result.ok) {
    this._postMessage({
      type: 'marigold_depth_result',
      id: request.id,
      ok: true,
      depthMap: result.depthMap,
      width: result.width,
      height: result.height,
      timeMs: result.timeMs,
    });
  } else {
    this._postMessage({
      type: 'marigold_depth_result',
      id: request.id,
      ok: false,
      reason: result.reason,
      message: result.message,
    });
  }
  break;
}
```

---

## Phase 4: Client API

Expose Marigold depth estimation through the client.

### 4.1 Client Method

```typescript
// packages/web-txt2img/src/worker/client.ts

async generateDepth(
  inputImage: ImageBitmap | HTMLImageElement | HTMLCanvasElement | OffscreenCanvas,
  options: {
    denoisingSteps?: number;
    ensembleSize?: number;
    processingRes?: number;
    matchInputRes?: boolean;
    onProgress?: (event: MarigoldProgressEvent) => void;
    signal?: AbortSignal;
  } = {}
): Promise<MarigoldGenerateResult> {
  // Serialize image to ArrayBuffer
  const canvas = new OffscreenCanvas(inputImage.width, inputImage.height);
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(inputImage, 0, 0);
  const bitmap = canvas.transferToImageBitmap();
  const blob = await createImageBitmap(bitmap);

  // Convert to PNG ArrayBuffer
  const pngBlob = await createImageBitmap(canvas.transferToImageBitmap())
    .then(b => b instanceof Blob ? b : canvas.convertToBlob({ type: 'image/png' }))
    .then(b => b.arrayBuffer());

  const request: MarigoldDepthRequest = {
    type: 'marigold_depth',
    id: crypto.randomUUID(),
    params: {
      inputImage: pngBlob,
      inputWidth: inputImage.width,
      inputHeight: inputImage.height,
      denoisingSteps: options.denoisingSteps,
      ensembleSize: options.ensembleSize,
      processingRes: options.processingRes,
      matchInputRes: options.matchInputRes,
    },
  };

  // Send to worker and await result
  return this._sendAndAwait<MarigoldDepthResult>(request, options.signal);
}
```

---

## Phase 5: Example Application

### 5.1 HTML Structure

```html
<!-- examples/vanilla-worker/index.html -->

<div class="depth-estimator">
  <input type="file" id="imageInput" accept="image/*" />
  <div class="controls">
    <label>
      Denoising Steps:
      <input type="range" id="stepsSlider" min="1" max="10" value="4" />
      <span id="stepsValue">4</span>
    </label>
    <label>
      Ensemble Size:
      <input type="range" id="ensembleSlider" min="1" max="5" value="1" />
      <span id="ensembleValue">1</span>
    </label>
    <button id="estimateBtn">Estimate Depth</button>
  </div>
  <div class="output">
    <canvas id="originalCanvas"></canvas>
    <canvas id="depthCanvas"></canvas>
  </div>
  <div id="progress"></div>
</div>
```

### 5.2 JavaScript Integration

```javascript
// examples/vanilla-worker/main.js

import { createClient } from 'web-txt2img';

const client = createClient(new URL('../../packages/web-txt2img/dist/worker/host.js', import.meta.url));

const imageInput = document.getElementById('imageInput');
const estimateBtn = document.getElementById('estimateBtn');
const stepsSlider = document.getElementById('stepsSlider');
const ensembleSlider = document.getElementById('ensembleSlider');
const progressDiv = document.getElementById('progress');

let currentImage = null;

imageInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const bitmap = await createImageBitmap(file);
  currentImage = bitmap;

  // Show original
  const canvas = document.getElementById('originalCanvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(bitmap, 0, 0);
});

estimateBtn.addEventListener('click', async () => {
  if (!currentImage) {
    alert('Please select an image first');
    return;
  }

  estimateBtn.disabled = true;
  progressDiv.textContent = 'Loading model...';

  try {
    // Load Marigold model
    await client.loadModel('marigold-depth-v1-1', {
      onProgress: (event) => {
        progressDiv.textContent = `Loading: ${event.message} (${Math.round(event.pct ?? 0)}%)`;
      },
    });

    progressDiv.textContent = 'Estimating depth...';

    // Generate depth map
    const result = await client.generateDepth(currentImage, {
      denoisingSteps: parseInt(stepsSlider.value),
      ensembleSize: parseInt(ensembleSlider.value),
      processingRes: 768,
      matchInputRes: true,
      onProgress: (event) => {
        progressDiv.textContent = `${event.phase}: ${Math.round(event.pct ?? 0)}%`;
      },
    });

    if (!result.ok) {
      throw new Error(result.message);
    }

    // Render depth map
    renderDepthMap(result.depthMap, result.width, result.height);

    progressDiv.textContent = `Done in ${Math.round(result.timeMs)}ms`;
  } catch (err) {
    progressDiv.textContent = `Error: ${err.message}`;
  } finally {
    estimateBtn.disabled = false;
  }
});

function renderDepthMap(depthData, width, height) {
  const canvas = document.getElementById('depthCanvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  const imageData = ctx.createImageData(width, height);
  const data = imageData.data;

  // Map depth to colormap (viridis-like)
  for (let i = 0; i < depthData.length; i++) {
    const depth = depthData[i];
    const [r, g, b] = depthToColor(depth);

    data[i * 4]     = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
    data[i * 4 + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);
}

// Simple viridis-like colormap
function depthToColor(depth) {
  // Simplified — use d3-scale-chromatic for production
  const t = depth * 3;
  if (t < 0.33) {
    return [Math.round(255 * t * 3), 0, Math.round(255 * (1 - t * 3))];
  } else if (t < 0.66) {
    return [0, Math.round(255 * ((t - 0.33) * 3)), Math.round(255 * (1 - (t - 0.33) * 3))];
  } else {
    return [Math.round(255 * ((t - 0.66) * 3)), 255, 0];
  }
}
```

---

## Implementation Checklist

### Priority Matrix

| Priority | Task | Complexity | Notes |
|----------|------|------------|-------|
| **P0** | Types and registry entries | Low | Foundation for everything else |
| **P0** | Adapter skeleton + model loading | Medium | 4 ONNX sessions (UNet, VAE enc/dec, CLIP) |
| **P0** | Image preprocessing | Low | Resize + normalize to [-1, 1] |
| **P0** | VAE encoding | Medium | Split moments, scale latents |
| **P0** | Denoising loop (single inference) | High | DDIM scheduler + UNet inference |
| **P0** | Depth decoding + postprocessing | Medium | Sigmoid activation, resize |
| **P1** | Main generate method | Medium | Orchestration of P0 components |
| **P1** | Worker protocol messages | Low | MarigoldDepthRequest/Result types |
| **P1** | Client API method | Low | `generateDepth()` wrapper |
| **P2** | Test-time ensembling | Medium | Median reduction, optional BFGS |
| **P2** | Example application | Low | File input + canvas rendering |
| **P3** | LCMScheduler support | Medium | Faster inference (1–2 steps) |
| **P3** | Normals pipeline variant | Medium | Same UNet, different VAE decoder |
| **P3** | IID pipeline variant | Medium | Same UNet, different decoder |
| **P4** | Depth alignment (BFGS) | High | Complex optimization, optional |

### ONNX Model Preparation

Before implementation, export Marigold components to ONNX:

```python
# export_marigold_onnx.py

import torch
from diffusers import MarigoldPipeline
from marigold.models.unet import MarigldModel

pipeline = MarigoldPipeline.from_pretrained('prs-eth/marigold-depth-v1-1')

# Export UNet
unet = pipeline.unet
unet.eval()

dummy_sample = torch.randn(1, 4, 96, 96)  # 768/8 = 96
dummy_timestep = torch.tensor(0, dtype=torch.long)
dummy_encoder_hidden_states = torch.randn(1, 1, 768)

torch.onnx.export(
    unet,
    (dummy_sample, dummy_timestep, dummy_encoder_hidden_states),
    'unet/model.onnx',
    input_names=['sample', 'timestep', 'encoder_hidden_states'],
    output_names=['sample'],
    dynamic_axes={
        'sample': {0: 'batch', 2: 'height', 3: 'width'},
        'encoder_hidden_states': {0: 'batch'},
    },
    opset_version=17,
)

# Export VAE encoder
vae_encoder = pipeline.vae.encoder
vae_encoder.eval()

dummy_rgb = torch.randn(1, 3, 768, 768)

torch.onnx.export(
    vae_encoder,
    (dummy_rgb,),
    'vae_encoder/model.onnx',
    input_names=['sample'],
    output_names=['latent'],
    dynamic_axes={'sample': {0: 'batch', 2: 'height', 3: 'width'}},
    opset_version=17,
)

# Export VAE decoder (depth-specific)
vae_decoder = pipeline.vae.decoder
vae_decoder.eval()

dummy_latent = torch.randn(1, 4, 96, 96)

torch.onnx.export(
    vae_decoder,
    (dummy_latent,),
    'vae_decoder/model.onnx',
    input_names=['latent'],
    output_names=['sample'],
    dynamic_axes={'latent': {0: 'batch', 2: 'height', 3: 'width'}},
    opset_version=17,
)

# Export CLIP text encoder
text_encoder = pipeline.text_encoder
text_encoder.eval()

dummy_input_ids = torch.tensor([[0]], dtype=torch.long)
dummy_attention_mask = torch.tensor([[1]], dtype=torch.long)

torch.onnx.export(
    text_encoder,
    {'input_ids': dummy_input_ids, 'attention_mask': dummy_attention_mask},
    'text_encoder/model.onnx',
    input_names=['input_ids', 'attention_mask'],
    output_names=['last_hidden_state'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'sequence'},
        'attention_mask': {0: 'batch', 1: 'sequence'},
    },
    opset_version=17,
)
```

### Memory Budget Estimates

| Component | Model Size (ONNX) | GPU Memory (WebGPU) | Notes |
|-----------|-------------------|---------------------|-------|
| UNet | ~1.6 GB | ~2 GB | Largest component, 868M params |
| VAE Encoder | ~250 MB | ~300 MB | SD-style VAE encoder |
| VAE Decoder | ~250 MB | ~300 MB | Depth-specific decoder |
| CLIP Text Encoder | ~500 MB | ~600 MB | ViT-B/32 |
| **Total** | **~2.6 GB** | **~3.2 GB** | Plus latent activations |

> **WebGPU memory constraint:** On devices with < 4 GB GPU memory, consider loading UNet lazily or using a distilled variant.

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Model load time | < 30s | Depends on network + GPU upload |
| Single inference (4 steps, 768px) | < 10s | WebGPU, 4 GB+ GPU memory |
| Ensemble (3×, 4 steps) | < 30s | 3× single inference + reduction |
| Memory peak | < 4 GB | UNet + VAE + activations |

---

## Known Challenges

1. **WebGPU Memory Limits:** Marigold's UNet is large (~868M params). Test on target devices for OOM errors.
2. **DDIM Scheduler Port:** Need accurate TypeScript implementation of DDIM scheduler with proper alpha_cumprod scheduling.
3. **ONNX Export:** Marigold's custom UNet may require export fixes (dynamic shapes, custom ops).
4. **Image Passing:** Transferring images to worker requires serialization (Blob → ArrayBuffer → ImageBitmap).
5. **Ensembling Memory:** Each ensemble member allocates latent tensors. Limit ensemble size based on available memory.
6. **Processing Resolution:** 768px is the default. Lower resolutions (512px) trade quality for speed/memory.

