# Transformers.js Onboarding Guide

This guide helps new engineers understand how Transformers.js is used in web-txt2img for browser-based text-to-image inference with the Janus-Pro-1B model.

## What is Transformers.js?

Transformers.js is a JavaScript port of Hugging Face's Transformers library that runs ML models in the browser. In web-txt2img, it's used via the `@huggingface/transformers` package (or `@xenova/transformers` as a fallback) to power the **Janus-Pro-1B** adapter.

Unlike ONNX Runtime (which requires manual session orchestration), Transformers.js provides high-level APIs for model loading, tokenization, and inference.

## How It's Used in This Project

### 1. Dynamic Import & Dependency Resolution

Transformers.js is loaded dynamically using a 3-tier resolution:

```typescript
// Tier 1: Bare-specifier import (preferred)
try {
  this.hf = await import('@huggingface/transformers');
} catch {
  // Tier 2: import.meta.resolve for dynamic resolution
  const resolved = import.meta.resolve('@huggingface/transformers', import.meta.url);
  this.hf = await import(resolved);
}

// Tier 3: Global fallback
this.hf = (globalThis as any).transformers;
```

This allows:
- **Optional dependency**: Only loaded when using Janus-Pro-1B
- **Package aliasing**: Supports both `@huggingface/transformers` and `@xenova/transformers`
- **Testing**: Inject mock implementations via `LoadOptions`

### 2. Model Loading

Janus-Pro-1B is a multimodal model loaded using `MultiModalityCausalLM`:

```typescript
const { AutoProcessor, MultiModalityCausalLM } = this.hf;

// Load model with per-submodel configuration
this.model = await MultiModalityCausalLM.from_pretrained('Janus/Janus-Pro-1B', {
  dtype: {
    // Quantization levels per submodel
    'language_model': 'q4',     // 4-bit quantized
    'language_model.norm': 'q4',
    'lm_head': 'fp16',         // Full precision for output layer
    'vision_model': 'q4',
    'vision_model.vision_tower': 'q4f16', // Mixed 4-bit + fp16
    'multi_modality': 'q4',
    'lm_head.decoder': 'fp32', // Full precision for decoder
  },
  device: 'webgpu',           // Run on GPU
  progress_callback: (progress) => {
    // Track per-asset download progress
    onProgress({
      phase: 'loading',
      message: `Loading ${progress.file ?? 'model'}`,
      bytesDownloaded: progress.loaded,
      totalBytesExpected: progress.total,
      asset: progress.file,
    });
  },
});

// Load processor for tokenization
this.processor = await AutoProcessor.from_pretrained('Janus/Janus-Pro-1B', {
  progress_callback: (progress) => { /* same pattern */ },
});
```

### 3. Per-Submodel Configuration

Transformers.js allows fine-grained control over dtype and device per submodel:

#### Dtype Options
| Dtype | Precision | Memory | Speed |
|-------|-----------|--------|-------|
| `'fp32'` | Full 32-bit float | Highest | Slowest |
| `'fp16'` | 16-bit float | Medium | Fast |
| `'q4'` | 4-bit quantized | Lowest | Fastest |
| `'q4f16'` | Mixed 4-bit + fp16 | Low | Fast |

#### Device Options
- `'webgpu'`: GPU acceleration (primary)
- `'wasm'`: WebAssembly fallback (not recommended)

#### Configuration Pattern
```typescript
dtype: {
  '*': 'q4',                    // Default: 4-bit quantization
  'lm_head': 'fp16',           // Override: full precision for output
  'vision_model.vision_tower': 'q4f16', // Mixed precision for vision
}
```

### 4. fp16 Capability Detection

Not all GPUs support fp16. Check before using:

```typescript
async function hasFp16Support(): Promise<boolean> {
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return adapter.features.has('shader-f16');
  } catch {
    return false;
  }
}
```

Use this to conditionally select dtypes:
```typescript
const dtype = hasFp16 ? 'fp16' : 'fp32';
```

### 5. Progress Callback Aggregation

Transformers.js reports progress per-asset. Aggregate for monotonic overall progress:

```typescript
// Track per-asset progress to avoid backward jumps
const assetProgress = new Map<string, number>();

function onTransformersProgress(progress: any) {
  const file = progress.file ?? 'unknown';
  const prev = assetProgress.get(file) ?? 0;

  // Only update if progress has advanced
  if (progress.loaded > prev) {
    assetProgress.set(file, progress.loaded);

    // Calculate total across all assets
    const totalDownloaded = Array.from(assetProgress.values()).reduce((a, b) => a + b, 0);

    onProgress({
      phase: 'loading',
      message: `Loading ${file}`,
      bytesDownloaded: totalDownloaded,
      totalBytesExpected: GRAND_APPROX, // ~2.25 GB for Janus
      asset: file,
    });
  }
}
```

### 6. Inference Pipeline

Janus-Pro-1B generates images through a text-to-image pipeline:

```typescript
// 1. Build conversation
const conversation = [
  { role: '<|User|>', content: prompt.trim() },
];

// 2. Process inputs with AutoProcessor
const inputs = await processor(conversation, {
  chat_template: 'text_to_image',
});

// 3. Generate images
const num_image_tokens = processor.num_image_tokens;
const outputs = await model.generate_images({
  ...inputs,
  min_new_tokens: num_image_tokens,
  max_new_tokens: num_image_tokens,
  do_sample: true,
  streamer: streamer, // For progress reporting
});

// 4. Convert output to blob
const blob = await outputs[0].toBlob();
```

### 7. Progress Streaming

Use a custom streamer to report progress during image generation:

```typescript
const StreamerBase = hf.BaseStreamer;

class ProgressStreamer extends StreamerBase {
  total: number;
  on_progress: (p: any) => void;
  count: number = 0;

  constructor(total: number, on_progress: (p: any) => void) {
    super();
    this.total = total;
    this.on_progress = on_progress;
  }

  put(_value: any) {
    // Check for abort
    if (signal?.aborted) {
      throw new Error('JANUS_STOP'); // Sentinel for cancellation
    }

    this.count++;
    this.on_progress({
      count: this.count,
      total: this.total,
      progress: this.count / this.total,
      time: performance.now() - this.start_time,
    });
  }

  end() { /* no-op */ }
}
```

### 8. Cache Management

Transformers.js uses Cache Storage for model downloads. Clear cache when needed:

```typescript
async purgeCache(): Promise<void> {
  await purgeModelCache(this.id);
}
```

### 9. Memory Management

Transformers.js models hold GPU memory. Clean up on unload:

```typescript
async unload(): Promise<void> {
  this.model = null;
  this.processor = null;
  this.hf = null;
  this.loaded = false;
}
```

Note: Transformers.js doesn't have explicit `release()` methods like ONNX Runtime. Setting references to `null` allows GC to collect GPU buffers.

## Common Gotchas

### Package Name Variations
- `@huggingface/transformers` is the current package
- `@xenova/transformers` is an older alias (still works)
- Always try both in your import resolution

### Model Size
- Janus-Pro-1B downloads ~2.25 GB
- Always use progress callbacks to show download status
- Cache Storage prevents re-downloads

### fp16 Support
- Not all GPUs support `shader-f16`
- Always detect before using fp16 dtypes
- Fall back to fp32 if unavailable

### Quantization Trade-offs
- `q4` reduces memory but may reduce quality
- Use higher precision for output layers (`lm_head`, `decoder`)
- Test quality vs. memory for your use case

### Chat Templates
- Janus uses `text_to_image` chat template
- Other models may use different templates
- Check model card for correct template name

### Abort Limitations
- Transformers.js doesn't support mid-inference abort
- Check `signal.aborted` before starting
- Use sentinel errors to unwind the call stack

## Debugging Tips

### Check WebGPU Support
```typescript
if (!navigator.gpu) {
  console.error('WebGPU not supported');
}
```

### Inspect Model Submodels
```typescript
console.log('Model config:', this.model.config);
console.log('Model submodels:', Object.keys(this.model.model));
```

### Check GPU Memory
```typescript
// Chrome DevTools > Rendering > GPU memory
// Or use the Memory panel
```

### Profile Inference Time
```typescript
const start = performance.now();
const outputs = await model.generate_images(inputs);
console.log(`Generation took ${performance.now() - start}ms`);
```

### Debug Progress Callbacks
```typescript
progress_callback: (p) => {
  console.log('Progress:', {
    file: p.file,
    loaded: p.loaded,
    total: p.total,
    progress: p.progress,
  });
}
```

## Differences from ONNX Runtime

| Feature | ONNX Runtime | Transformers.js |
|---------|--------------|-----------------|
| Session management | Manual `InferenceSession.create()` | High-level `from_pretrained()` |
| Multi-model pipelines | Manual orchestration | Single model with submodels |
| Progress callbacks | Per-asset, manual aggregation | Built-in per-asset callbacks |
| Memory cleanup | Explicit `session.release()` | GC via null references |
| Dtype control | Limited | Per-submodel fine-grained |
| Tokenization | Separate tokenizer library | Built-in `AutoProcessor` |
| Inference API | `session.run(inputs)` | `model.generate_images()` |

## References

- [Transformers.js Documentation](https://huggingface.co/docs/transformers.js)
- [Janus-Pro Adapter Source](../packages/web-txt2img/src/adapters/janus-pro.ts)
- [Model Type Definitions](../packages/web-txt2img/src/types.ts)
- [Hugging Face Model Card](https://huggingface.co/Janus/Janus-Pro-1B)
