# ONNX Runtime Web Use Cases

This document covers ONNX Runtime Web use cases beyond text-to-image generation, based on patterns from the [onnxruntime-web-demo](https://github.com/microsoft/onnxruntime-web-demo) project. The demos showcase five distinct model categories: image classification, emotion recognition, object detection, and handwritten digit recognition.

## Table of Contents

- [Overview](#overview)
- [Execution Providers](#execution-providers)
- [Session Management Patterns](#session-management-patterns)
- [Use Case 1: Image Classification (MobileNet, SqueezeNet)](#use-case-1-image-classification-mobilenet-squeezenet)
- [Use Case 2: Emotion Recognition (FER+)](#use-case-2-emotion-recognition-fer)
- [Use Case 3: Object Detection (YOLO)](#use-case-3-object-detection-yolo)
- [Use Case 4: Handwritten Digit Recognition (MNIST)](#use-case-4-handwritten-digit-recognition-mnist)
- [Tensor Operations and Postprocessing](#tensor-operations-and-postprocessing)
- [Input Preprocessing Patterns](#input-preprocessing-patterns)
- [Best Practices](#best-practices)

---

## Overview

ONNX Runtime Web supports a wide range of inference scenarios:

| Use Case | Model | Input | Output | Key Pattern |
|----------|-------|-------|--------|-------------|
| Image Classification | MobileNet v2, SqueezeNet | Image (224×224) | Class probabilities | Softmax + top-K |
| Emotion Recognition | FER+ | Face (64×64 grayscale) | Emotion label | Softmax + argmax |
| Object Detection | YOLO v2 | Image (416×416) | Bounding boxes | Transpose + postprocess |
| Handwritten Digits | MNIST | Drawing (28×28) | Digit (0-9) | Center crop + resize |

---

## Execution Providers

ONNX Runtime Web supports multiple execution backends. The demo demonstrates runtime provider selection with graceful fallback:

### Available Providers

| Provider | Key | Description |
|----------|-----|-------------|
| WebGL | `'webgl'` | GPU acceleration via WebGL (preferred for performance) |
| WebAssembly | `'wasm'` | CPU fallback via WASM |
| WebGPU | `'webgpu'` | Next-gen GPU API (newer versions) |

### Provider Selection Pattern

```typescript
import { InferenceSession } from 'onnxruntime-web';

// GPU session (WebGL)
const gpuSession = await InferenceSession.create(modelBuffer, {
  executionProviders: ['webgl'],
});

// CPU session (WASM)
const cpuSession = await InferenceSession.create(modelBuffer, {
  executionProviders: ['wasm'],
});
```

### Graceful Fallback

Try WebGL first, fall back to WASM if unavailable:

```typescript
let session: InferenceSession;
try {
  session = await InferenceSession.create(modelBuffer, {
    executionProviders: ['webgl'],
  });
} catch {
  session = await InferenceSession.create(modelBuffer, {
    executionProviders: ['wasm'],
  });
}
```

### Dual-Session Caching

For apps that let users switch backends, cache both sessions:

```typescript
class ModelManager {
  gpuSession: InferenceSession | undefined;
  cpuSession: InferenceSession | undefined;
  activeSession: InferenceSession | undefined;

  async switchBackend(provider: 'webgl' | 'wasm') {
    if (provider === 'webgl' && this.gpuSession) {
      this.activeSession = this.gpuSession;
      return;
    }
    if (provider === 'wasm' && this.cpuSession) {
      this.activeSession = this.cpuSession;
      return;
    }
    // Create new session if not cached
    this.activeSession = await InferenceSession.create(modelBuffer, {
      executionProviders: [provider],
    });
  }
}
```

---

## Session Management Patterns

### Model Fetching

Fetch model as `ArrayBuffer` before creating sessions:

```typescript
const response = await fetch('/models/mobilenetv2-7.onnx');
const modelBuffer = await response.arrayBuffer();
const session = await InferenceSession.create(modelBuffer, {
  executionProviders: ['webgl'],
});
```

### Warmup Inference

Run a dummy inference before real use to compile shaders (WebGL) or initialize WASM:

```typescript
async function warmupModel(session: InferenceSession, dims: number[]) {
  const size = dims.reduce((a, b) => a * b);
  const warmupTensor = new Tensor('float32', new Float32Array(size), dims);

  // Fill with random values [-1.0, 1.0)
  for (let i = 0; i < size; i++) {
    (warmupTensor.data as Float32Array)[i] = Math.random() * 2.0 - 1.0;
  }

  await session.run({ [session.inputNames[0]]: warmupTensor });
}

// Warmup with model's expected input shape
await warmupModel(session, [1, 3, 224, 224]);
```

**Why warmup matters:**
- WebGL: Compiles GPU shaders on first run (can take hundreds of ms)
- WASM: Initializes SIMD routines and memory buffers
- Without warmup, the first real inference has a cold-start penalty

### Session Lifecycle

```typescript
// Component/model lifecycle
async created() {
  this.modelFile = await fetch(this.modelPath).then(r => r.arrayBuffer());
  await this.initSession();
}

beforeDestroy() {
  this.session = undefined; // Release reference for GC
}
```

---

## Use Case 1: Image Classification (MobileNet, SqueezeNet)

### Architecture

```
Image → Canvas → Preprocess → Tensor → Inference → Softmax → Top-K Classes
```

### Input Preprocessing

Image classification models expect normalized RGB tensors in NCHW format:

```typescript
import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import { Tensor } from 'onnxruntime-web';

function preprocessImage(ctx: CanvasRenderingContext2D): Tensor {
  const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
  const { data, width, height } = imageData;

  // Source tensor: [H, W, 4] (RGBA)
  const srcTensor = ndarray(new Float32Array(data), [width, height, 4]);

  // Target tensor: [1, 3, H, W] (NCHW format)
  const dstTensor = ndarray(new Float32Array(width * height * 3), [1, 3, width, height]);

  // Extract RGB channels (drop alpha)
  ops.assign(dstTensor.pick(0, 0, null, null), srcTensor.pick(null, null, 0)); // R
  ops.assign(dstTensor.pick(0, 1, null, null), srcTensor.pick(null, null, 1)); // G
  ops.assign(dstTensor.pick(0, 2, null, null), srcTensor.pick(null, null, 2)); // B

  // Normalize to [0, 1]
  ops.divseq(dstTensor, 255);

  // ImageNet mean subtraction
  ops.subseq(dstTensor.pick(0, 0, null, null), 0.485); // R
  ops.subseq(dstTensor.pick(0, 1, null, null), 0.456); // G
  ops.subseq(dstTensor.pick(0, 2, null, null), 0.406); // B

  // ImageNet std division
  ops.divseq(dstTensor.pick(0, 0, null, null), 0.229); // R
  ops.divseq(dstTensor.pick(0, 1, null, null), 0.224); // G
  ops.divseq(dstTensor.pick(0, 2, null, null), 0.225); // B

  return new Tensor('float32', dstTensor.data as Float32Array, [1, 3, width, height]);
}
```

### Key Preprocessing Steps

1. **Channel extraction**: RGBA → RGB (drop alpha)
2. **Layout transform**: NHWC → NCHW (ONNX expects channels-first)
3. **Normalization**: Divide by 255 to get [0, 1] range
4. **Mean subtraction**: Subtract ImageNet per-channel means
5. **Std division**: Divide by ImageNet per-channel standard deviations

### Output Postprocessing

Apply softmax and extract top-K predictions:

```typescript
import _ from 'lodash';

// Softmax activation
function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - max)); // Numerical stability
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

// Top-K class extraction
function getTopK(probabilities: number[], classNames: string[], k = 5) {
  const indexed = probabilities.map((prob, idx) => [prob, idx]);
  const sorted = _.sortBy(indexed, x => x[0]).reverse();
  const topK = sorted.slice(0, k);

  return topK.map(([prob, idx]) => ({
    class: classNames[idx],
    probability: prob,
    index: idx,
  }));
}

// Usage
const output = await session.run({ input: tensor });
const logits = Array.from(output[Object.keys(output)[0]].data);
const probabilities = softmax(logits);
const predictions = getTopK(probabilities, imagenetClasses, 5);
```

### Libraries Used

- **`ndarray`** + **`ndarray-ops`**: Efficient tensor manipulation with broadcasting
- **`lodash`**: Sorting and array utilities for top-K extraction

---

## Use Case 2: Emotion Recognition (FER+)

### Architecture

```
Webcam/Image → Face Crop → Grayscale → Normalize → Inference → Softmax → Emotion Label
```

### Input Preprocessing

Emotion recognition uses grayscale input with different normalization:

```typescript
function preprocessEmotion(ctx: CanvasRenderingContext2D): Tensor {
  // Scale to 64×64
  const scaledCtx = getTempCanvas().getContext('2d');
  scaledCtx.canvas.width = 64;
  scaledCtx.canvas.height = 64;
  scaledCtx.drawImage(ctx.canvas, 0, 0, 64, 64);

  const { data } = scaledCtx.getImageData(0, 0, 64, 64);

  // Convert to grayscale and normalize to [-1, 1]
  const grayscale: number[] = [];
  for (let i = 0; i < data.length; i += 4) {
    // ITU-R Rec. BT.601 luminance formula
    const luminance = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
    // Normalize: [0, 255] → [-1, 1] with zero point at 127.5
    grayscale.push((luminance - 127.5) / 127.5);
  }

  return new Tensor('float32', new Float32Array(grayscale), [1, 1, 64, 64]);
}
```

### Key Differences from Classification

| Aspect | Image Classification | Emotion Recognition |
|--------|---------------------|---------------------|
| Channels | 3 (RGB) | 1 (grayscale) |
| Input shape | `[1, 3, 224, 224]` | `[1, 1, 64, 64]` |
| Normalization | ImageNet mean/std | `[-1, 1]` range |
| Output | 1000 classes | 8 emotions |

### Output Postprocessing

```typescript
const EMOTION_LABELS = [
  'neutral', 'happiness', 'surprise', 'sadness',
  'anger', 'disgust', 'fear', 'contempt',
];

function postprocessEmotion(output: Tensor) {
  const probabilities = softmax(Array.from(output.data));
  const maxIndex = probabilities.indexOf(Math.max(...probabilities));

  return {
    emotion: EMOTION_LABELS[maxIndex],
    confidence: probabilities[maxIndex],
    allProbabilities: probabilities.map((p, i) => ({
      emotion: EMOTION_LABELS[i],
      probability: p,
    })),
  };
}
```

---

## Use Case 3: Object Detection (YOLO)

### Architecture

```
Image → Canvas → Preprocess → Inference → Transpose → Decode Boxes → NMS → Bounding Boxes
```

### Input Preprocessing

YOLO uses simpler preprocessing (no mean/std normalization):

```typescript
function preprocessYolo(ctx: CanvasRenderingContext2D): Tensor {
  const { data, width, height } = ctx.getImageData(0, 0, 416, 416);

  const src = ndarray(new Float32Array(data), [width, height, 4]);
  const dst = ndarray(new Float32Array(width * height * 3), [1, 3, width, height]);

  // Extract RGB channels
  ops.assign(dst.pick(0, 0, null, null), src.pick(null, null, 0));
  ops.assign(dst.pick(0, 1, null, null), src.pick(null, null, 1));
  ops.assign(dst.pick(0, 2, null, null), src.pick(null, null, 2));

  return new Tensor('float32', dst.data as Float32Array, [1, 3, 416, 416]);
}
```

### Output Postprocessing

YOLO requires significant postprocessing to convert raw tensor output to bounding boxes:

```typescript
import { Tensor } from 'onnxruntime-web';

// Transpose from [1, 125, 13, 13] to [1, 13, 13, 125]
function transpose(x: Tensor, perm: number[]): Tensor {
  // Reorder dimensions according to permutation
  // Implementation handles stride calculation and element copying
  return transposedTensor;
}

// Decode YOLO output into bounding boxes
async function postprocessYOLO(tensor: Tensor, numClasses: number) {
  // 1. Transpose to NHWC layout
  const transposed = transpose(tensor, [0, 2, 3, 1]);

  // 2. Apply sigmoid to objectness scores
  // 3. Apply softmax to class probabilities
  // 4. Decode box coordinates from anchor boxes
  // 5. Apply confidence threshold
  // 6. Run Non-Maximum Suppression (NMS)

  return boundingBoxes; // Array of { x, y, width, height, class, confidence }
}
```

### Custom Tensor Operations

YOLO postprocessing requires a mini tensor library:

| Operation | Purpose | Example |
|-----------|---------|---------|
| `transpose()` | Reorder tensor dimensions | `[B, C, H, W]` → `[B, H, W, C]` |
| `reshape()` | Change tensor shape | `[13, 13, 125]` → `[13*13*125]` |
| `sigmoid()` | Activation for objectness | `σ(x) = 1 / (1 + e^(-x))` |
| `softmax()` | Activation for class probs | Per-box class distribution |
| `concat()` | Join tensors along axis | Merge multiple feature maps |
| `add()`, `sub()`, `mul()`, `div()` | Element-wise arithmetic | Box coordinate decoding |

### Tensor Broadcasting

Binary operations support broadcasting (like NumPy):

```typescript
import ndarray from 'ndarray';
import { Tensor } from 'onnxruntime-web';

function add(t1: Tensor, t2: Tensor): Tensor {
  const a = ndarray(t1.data, t1.dims);
  const b = ndarray(t2.data, t2.dims);
  const result = ndarray.ops.broadcast((x, y) => x + y, a, b);
  return new Tensor(t1.type, result.data, result.shape);
}
```

---

## Use Case 4: Handwritten Digit Recognition (MNIST)

### Architecture

```
Canvas Drawing → Center Crop → Resize to 28×28 → Alpha Channel → Inference → Argmax → Digit
```

### Input Preprocessing

MNIST demonstrates canvas-based input with geometric preprocessing:

```typescript
function preprocessMNIST(ctx: CanvasRenderingContext2D): Tensor {
  // Step 1: Center crop based on alpha channel
  const cropped = centerCrop(ctx.getImageData(0, 0, 300, 300));

  // Step 2: Resize to 28×28 using canvas scaling
  const scaledCanvas = document.createElement('canvas');
  scaledCanvas.width = 28;
  scaledCanvas.height = 28;
  const scaledCtx = scaledCanvas.getContext('2d');
  scaledCtx.drawImage(croppedCanvas, 0, 0, 28, 28);

  // Step 3: Extract alpha channel as grayscale
  const { data } = scaledCtx.getImageData(0, 0, 28, 28);
  const input = new Float32Array(784);
  for (let i = 0; i < data.length; i += 4) {
    input[i / 4] = data[i + 3] / 255; // Alpha as intensity
  }

  return new Tensor('float32', input, [1, 1, 28, 28]);
}

// Find bounding box of drawn content using alpha channel
function centerCrop(imageData: ImageData) {
  const { data, width, height } = imageData;

  // Find min/max coordinates where alpha > 0
  let [xmin, ymin] = [width, height];
  let [xmax, ymax] = [-1, -1];

  for (let i = 0; i < width; i++) {
    for (let j = 0; j < height; j++) {
      if (data[(i + j * width) * 4 + 3] > 0) {
        xmin = Math.min(xmin, i);
        xmax = Math.max(xmax, i);
        ymin = Math.min(ymin, j);
        ymax = Math.max(ymax, j);
      }
    }
  }

  // Make bounding box square and add padding
  // ... crop and return
}
```

### Key Techniques

1. **Alpha-channel-based cropping**: Find the drawn content bounds using the alpha channel
2. **Square bounding box**: Pad to make the crop square before resizing
3. **Canvas scaling**: Use browser's built-in image scaling for resize
4. **Alpha as intensity**: Use alpha channel as the single grayscale input channel

### Output Postprocessing

```typescript
function postprocessMNIST(output: Tensor): number {
  const probabilities = softmax(Array.from(output.data));
  return probabilities.indexOf(Math.max(...probabilities)); // Argmax
}
```

---

## Tensor Operations and Postprocessing

### ndarray and ndarray-ops

The demo uses `ndarray` for efficient tensor manipulation:

```typescript
import ndarray from 'ndarray';
import ops from 'ndarray-ops';

// Create tensor with specific shape
const t = ndarray(new Float32Array(224 * 224 * 3), [3, 224, 224]);

// Pick slices (like NumPy indexing)
const channel0 = t.pick(0, null, null); // First channel

// Assign between tensors
ops.assign(dst.pick(0, 0, null, null), src.pick(null, null, 0));

// Element-wise operations with broadcasting
ops.divseq(tensor, 255);           // Divide all elements by 255
ops.subseq(tensor, 0.5);           // Subtract 0.5 from all elements
```

### Common Postprocessing Patterns

| Pattern | Use Case | Implementation |
|---------|----------|----------------|
| Softmax | Classification | `exp(x - max) / sum(exp(x - max))` |
| Argmax | Single-label prediction | `indexOf(Math.max(...))` |
| Top-K | Multi-label ranking | Sort + slice |
| Transpose | Layout conversion | `[N,C,H,W]` ↔ `[N,H,W,C]` |
| Sigmoid | Binary activation | `1 / (1 + exp(-x))` |
| NMS | Object detection | Suppress overlapping boxes |

---

## Input Preprocessing Patterns

### Canvas-Based Preprocessing

All demos use HTML Canvas for image preprocessing:

```
Source (file/webcam/drawing) → Canvas → getImageData() → Tensor manipulation → ONNX Tensor
```

### Common Preprocessing Pipeline

```typescript
// 1. Get pixel data from canvas
const imageData = ctx.getImageData(0, 0, width, height);

// 2. Reshape and reorganize channels
const src = ndarray(new Float32Array(imageData.data), [width, height, 4]);
const dst = ndarray(new Float32Array(width * height * 3), [1, 3, width, height]);

// 3. Apply normalization
ops.divseq(dst, 255);

// 4. Create ONNX Tensor
const tensor = new Tensor('float32', dst.data, [1, 3, width, height]);
```

### Webcam Integration

For real-time inference, capture frames from webcam:

```typescript
async function startWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  const video = document.getElementById('webcam') as HTMLVideoElement;
  video.srcObject = stream;
  await video.play();

  // Capture frame to canvas for preprocessing
  function captureFrame() {
    const canvas = document.getElementById('input-canvas');
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, 416, 416);
    return ctx;
  }
}
```

---

## Best Practices

### 1. Always Warmup Sessions

```typescript
// BAD: First inference is slow due to cold start
const result = await session.run(feeds);

// GOOD: Warmup before real inference
await warmupModel(session, [1, 3, 224, 224]);
const result = await session.run(feeds);
```

### 2. Cache Sessions Across Backend Switches

```typescript
// Create both sessions upfront, switch references
this.gpuSession = await InferenceSession.create(model, { executionProviders: ['webgl'] });
this.cpuSession = await InferenceSession.create(model, { executionProviders: ['wasm'] });
```

### 3. Use ndarray for Batch Operations

```typescript
// BAD: Manual loops for normalization
for (let i = 0; i < data.length; i++) data[i] /= 255;

// GOOD: Vectorized operations
ops.divseq(tensor, 255);
```

### 4. Handle Provider Availability

```typescript
// Try preferred provider, fall back gracefully
const providers = ['webgl', 'wasm'];
for (const provider of providers) {
  try {
    session = await InferenceSession.create(model, { executionProviders: [provider] });
    break;
  } catch {
    continue;
  }
}
```

### 5. Measure Inference Time

```typescript
const start = performance.now();
const output = await session.run(feeds);
const inferenceTime = performance.now() - start;
console.log(`Inference: ${inferenceTime.toFixed(1)}ms`);
```

### 6. Clean Up on Component Destroy

```typescript
beforeDestroy() {
  this.session = undefined;
  this.gpuSession = undefined;
  this.cpuSession = undefined;
}
```

### 7. Use Hidden Canvases for Intermediate Processing

```html
<!-- Hidden canvases for preprocessing steps -->
<canvas id="center-crop-canvas" style="display: none"></canvas>
<canvas id="scaled-canvas" width="28" height="28" style="display: none"></canvas>
```

### 8. Debounce Rapid Inference

For webcam/drawing inputs, debounce to avoid overwhelming the inference pipeline:

```typescript
import _ from 'lodash';

// Debounce inference to 100ms
const debouncedRun = _.debounce(() => {
  this.runInference();
}, 100);
```

---

## References

- [ONNX Runtime Web Demo Source](https://github.com/microsoft/onnxruntime-web-demo)
- [ONNX Runtime Web Documentation](https://onnxruntime.ai/docsexecution-providers/web.html)
- [MobileNet ONNX Model](https://github.com/onnx/models/tree/main/vision/classification/mobilenet)
- [YOLO ONNX Model](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/tiny-yolov2)
- [ndarray Documentation](https://github.com/scijs/ndarray)
- [ndarray-ops Documentation](https://github.com/scijs/ndarray-ops)
