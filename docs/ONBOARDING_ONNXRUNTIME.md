# ONNX Runtime Web Onboarding Guide

This guide helps new engineers understand how ONNX Runtime Web is used in web-txt2img for browser-based text-to-image inference.

## What is ONNX Runtime Web?

ONNX Runtime Web (`onnxruntime-web`) is a JavaScript port of Microsoft's ONNX Runtime that runs pre-trained ML models in the browser. It supports multiple backends:

| Backend | Status in Project | Performance | Notes |
|---------|-------------------|-------------|-------|
| **WebGPU** | ✅ Primary | Fastest | GPU acceleration via WebGPU API |
| **WASM** | ⚠️ Experimental | Slower | SIMD + threads available; not recommended for production |

In web-txt2img, ONNX Runtime powers the **SD-Turbo** adapter (`packages/web-txt2img/src/adapters/sd-turbo.ts`), which generates images from text prompts using a Stable Diffusion Turbo pipeline.

## How It's Used in This Project

### 1. Dynamic Import & Dependency Resolution

ONNX Runtime is **never** a static dependency. It's loaded dynamically at runtime using a 3-tier resolution strategy:

```typescript
// Tier 1: Injected via LoadOptions (for testing/dependency injection)
if (this.ortProvider) this.ort = await this.ortProvider();

// Tier 2: Dynamic import (try webgpu build first, then default)
else this.ort = await import('onnxruntime-web/webgpu');

// Tier 3: Global fallback (for environments where import fails)
else this.ort = (globalThis as any).onnxRuntimeWeb;
```

This pattern allows:
- **Tree-shaking**: Code that doesn't use SD-Turbo never imports ONNX Runtime
- **Flexibility**: Tests can inject mock implementations
- **Resilience**: Multiple fallback paths for different build environments

### 2. Session Creation

The SD-Turbo pipeline requires **three separate ONNX sessions**, each running a different part of the model:

```typescript
const sessions = {
  text_encoder: await ort.InferenceSession.create(textEncoderBuffer, sessionOptions),
  unet: await ort.InferenceSession.create(unetBuffer, sessionOptions),
  vae_decoder: await ort.InferenceSession.create(vaeDecoderBuffer, sessionOptions),
};
```

#### Session Options (Critical for Performance)

```typescript
const sessionOptions = {
  enableMemPattern: false,        // Disable memory pattern optimization (reduces init time)
  enableCpuMemArena: false,       // Disable CPU memory arena (WebGPU doesn't use it)
  epContext: { disable_prepacking: true },  // Skip input prepacking (saves GPU memory)
  device: 'webgpu',              // Force WebGPU backend
  executionProviders: [{ name: 'webgpu' }], // Explicit provider selection
};
```

#### Dynamic Shape Handling

ONNX models expect fixed input shapes, but text prompts vary in length. Use `freeDimensionOverrides` to allow dynamic batch/sequence dimensions:

```typescript
const sessionOptions = {
  // ...other options
  freeDimensionOverrides: {
    'input_ids': { 0: 1, 1: 77 },   // batch=1, seq_len=77 (max tokens)
    'sample': { 0: 1, 1: 4, 2: 64, 3: 64 }, // latent shape
  },
};
```

### 3. Model Download & Caching

Model files (~2.34 GB total) are downloaded once and cached using the browser Cache Storage API:

```typescript
// Fetch with progress tracking and caching
const buffer = await fetchArrayBufferWithCacheProgress(
  modelUrl,
  modelId,
  (progress) => {
    onProgress({
      phase: 'loading',
      message: `Downloading ${modelUrl.split('/').pop()}`,
      bytesDownloaded: progress.bytesLoaded,
      totalBytesExpected: GRAND_APPROX,
      asset: modelUrl,
    });
  }
);
```

Key patterns:
- **Cumulative byte tracking**: Sum bytes across all assets for overall progress
- **Per-asset progress**: Each download reports its own progress, aggregated at the adapter level
- **Cache-first**: Subsequent loads read from Cache Storage, skipping network

### 4. Inference Pipeline

The SD-Turbo inference follows a 4-stage pipeline:

```
Text Prompt → Text Encoder → UNet → VAE Decoder → PNG Image
```

#### Stage 1: Tokenization
```typescript
const { input_ids } = await tokenizer(prompt, {
  padding: true,
  max_length: 77,
  truncation: true,
  return_tensor: false,
});
```

#### Stage 2: Text Encoding
```typescript
const encOut = await sessions.text_encoder.run({
  input_ids: new ort.Tensor('int32', input_ids, [1, input_ids.length]),
});
const text_embeddings = encOut.last_hidden_state ?? encOut;
```

#### Stage 3: UNet Denoising
```typescript
// Prepare latent noise tensor
const latent = new ort.Tensor(randn_latents([1, 4, 64, 64], sigma, seed), [1, 4, 64, 64]);
const latent_model_input = scale_model_inputs(ort, latent, sigma);

// Run UNet
const feed = {
  sample: latent_model_input,
  timestep: new ort.Tensor('int64', [999n], [1]),
  encoder_hidden_states: text_embeddings,
};
const unetOutput = await sessions.unet.run(feed);
```

#### Stage 4: VAE Decoding & Image Output
```typescript
const new_latents = step(ort, unetOutput, latent, sigma, vae_scaling_factor);
const vaeOut = await sessions.vae_decoder.run({ latent_sample: new_latents });
const blob = await tensorToPngBlob(vaeOut.sample ?? vaeOut);
```

### 5. Tensor Manipulation

ONNX Runtime uses `ort.Tensor` objects for data flow between sessions. Key patterns:

```typescript
// Create tensor from Float32Array
const tensor = new ort.Tensor('float32', data, [batch, channels, height, width]);

// Access tensor data and shape
const { data, dims } = tensor;

// Create tensor from Int32Array (for token IDs)
const idsTensor = new ort.Tensor('int32', ids, [1, ids.length]);

// Create tensor from BigInt (for timesteps)
const timestep = new ort.Tensor('int64', [999n], [1]);
```

#### Tensor-to-Image Conversion

The VAE decoder outputs a `[1, 3, H, W]` tensor (NCHW format). Convert to PNG:

```typescript
async function tensorToPngBlob(t: any): Promise<Blob> {
  const [n, c, h, w] = t.dims;
  const data: Float32Array = t.data;
  const out = new Uint8ClampedArray(w * h * 4);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const r = data[0 * h * w + y * w + x];
      const g = data[1 * h * w + y * w + x];
      const b = data[2 * h * w + y * w + x];

      // Denormalize: [-1, 1] → [0, 255]
      const clamp = (v: number) => {
        let x = v / 2 + 0.5;
        return Math.round(Math.max(0, Math.min(1, x)) * 255);
      };

      out[idx++] = clamp(r);
      out[idx++] = clamp(g);
      out[idx++] = clamp(b);
      out[idx++] = 255; // Alpha
    }
  }

  // Create PNG blob from pixel data
  // ... (uses OffscreenCanvas or canvas.toBlob())
}
```

### 6. Memory Management

ONNX Runtime sessions hold GPU memory. **Always clean up** to prevent leaks:

```typescript
async unload(): Promise<void> {
  try {
    (this.sessions.unet as any)?.release?.();
    (this.sessions.text_encoder as any)?.release?.();
    (this.sessions.vae_decoder as any)?.release?.();
  } finally {
    this.sessions = {};
    this.ort = null;
    this.loaded = false;
  }
}
```

Also dispose intermediate tensors when done:

```typescript
if (typeof tensor.dispose === 'function') tensor.dispose();
```

### 7. Progress Reporting

Standardized progress events help the UI show meaningful feedback:

```typescript
onProgress?.({ phase: 'tokenizing', pct: 10 });
onProgress?.({ phase: 'encoding', pct: 25 });
onProgress?.({ phase: 'denoising', pct: 70 });
onProgress?.({ phase: 'decoding', pct: 95 });
onProgress?.({ phase: 'complete', pct: 100, timeMs });
```

Each stage reports:
- `phase`: Current operation name
- `pct`: Percentage (0-100)
- `timeMs`: Total time when complete

### 8. Abort Handling

Check `signal.aborted` between pipeline stages to allow cancellation:

```typescript
if (signal?.aborted) {
  onProgress?.({ phase: 'complete', aborted: true, pct: 0 });
  return { ok: false, reason: 'cancelled' };
}
```

## Common Gotchas

### WebGPU Backend Selection
- WebGPU is **required** for reliable SD-Turbo operation
- WASM backend exists but is experimental and significantly slower
- Always check WebGPU support before attempting to load: `navigator.gpu !== undefined`

### Session Initialization Time
- Creating sessions from large models can take seconds
- Use `enableMemPattern: false` to reduce initialization overhead
- Time the session creation and report it as a progress event

### Tensor Shape Mismatches
- ONNX models are strict about input shapes
- Use `freeDimensionOverrides` for dynamic dimensions
- Ensure tensor shapes match model expectations exactly

### Memory Leaks
- Forgetting `session.release()` is the #1 cause of memory leaks
- Always null out references after unload
- Dispose intermediate tensors to free GPU memory

### Model File Sizes
- SD-Turbo downloads ~2.34 GB total
- Always use Cache Storage to avoid re-downloading
- Report cumulative bytes for accurate progress bars

## Debugging Tips

### Check WebGPU Support
```typescript
if (!navigator.gpu) {
  console.error('WebGPU not supported in this browser');
}
```

### Inspect Session Outputs
```typescript
const output = await session.run(inputs);
console.log('Output keys:', Object.keys(output));
console.log('Output shapes:', Object.fromEntries(
  Object.entries(output).map(([k, v]) => [k, (v as any).dims])
));
```

### Profile Inference Time
```typescript
const start = performance.now();
const result = await session.run(inputs);
console.log(`Inference took ${performance.now() - start}ms`);
```

### Check GPU Memory
```typescript
// In Chrome DevTools > Rendering > GPU memory
// Or use the Memory panel to track ONNX Runtime allocations
```

## References

- [ONNX Runtime Web Documentation](https://onnxruntime.ai/docsexecution-providers/web.html)
- [SD-Turbo Adapter Source](../packages/web-txt2img/src/adapters/sd-turbo.ts)
- [Model Type Definitions](../packages/web-txt2img/src/types.ts)
- [Cache Storage Implementation](../packages/web-txt2img/src/cache.ts)
