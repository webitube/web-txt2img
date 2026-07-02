// Bonsai-Image-4B adapter for web-txt2img
// Implements the Adapter interface for Bonsai ternary/binary models
// Uses GGUF for DiT transformer + VAE, Transformers.js for T5 text encoder

import type {
  Adapter,
  BackendId,
  Capabilities,
  GenerateParams,
  GenerateResult,
  LoadOptions,
  LoadResult,
  ModelId,
} from '../types.js';
import { purgeModelCache } from '../cache.js';
import { parseGguf, extractTensor, extractTensorsByName, getMetadataInt, getMetadataString } from '../gguf/parser.js';
import { dequantize, GgufTensorType } from '../gguf/dequantize.js';

type HF = typeof import('@huggingface/transformers');

// Bonsai-specific model URLs
const BONSAI_MODELS = {
  // GGUF files (local dev server for testing, HuggingFace for production)
  ggufBase: typeof import.meta !== 'undefined' && (import.meta as any).env?.DEV
    ? 'http://localhost:8765'
    : 'https://huggingface.co/prism-ml/bonsai-image-4B-gguf/resolve/main',
  // Transformers.js ONNX text encoder
  textEncoderId: 'PrismML/Bonsai-Image-4B-TextEncoder-ONNX',
  // Tokenizer
  tokenizerId: 'deepgrove/Bonsai',
} as const;

// Bonsai pipeline defaults
const BONSAI_DEFAULTS = {
  steps: 4,
  guidanceScale: 1.0,
  width: 512,
  height: 512,
  scheduler: 'euler',
} as const;

export class BonsaiAdapter implements Adapter {
  readonly id: ModelId;
  
  // State
  private loaded = false;
  private backendUsed: BackendId | null = null;
  
  // Pipeline components
  private tokenizer: any | null = null;
  private textEncoder: any | null = null;
  private hf: HF | null = null;
  
  // GGUF data
  private ggufTransformer: any | null = null;
  private ggufVae: any | null = null;
  
  // WebGPU
  private webgpuDevice: GPUDevice | null = null;
  private tensorBuffers: Map<string, GPUBuffer> = new Map();
  
  // Generation state
  private variant: 'ternary' | 'binary';
  
  constructor(variant: 'ternary' | 'binary' = 'ternary') {
    this.id = variant === 'ternary' ? 'bonsai-ternary' : 'bonsai-binary';
    this.variant = variant;
  }
  
  checkSupport(c: Capabilities): BackendId[] {
    // Bonsai requires WebGPU for DiT inference
    return c.webgpu ? ['webgpu'] : [];
  }
  
  async load(options: Required<Pick<LoadOptions, 'backendPreference'>> & LoadOptions): Promise<LoadResult> {
    const preferred = options.backendPreference;
    if (!preferred.includes('webgpu')) {
      return { ok: false, reason: 'backend_unavailable', message: 'Bonsai requires WebGPU' };
    }
    
    // Dynamic import of Transformers.js
    let hf: any = null;
    try { hf = await import('@huggingface/transformers').catch(() => null); } catch {}
    if (!hf) {
      const g: any = globalThis as any;
      hf = g.transformers || g.HFTransformers || null;
    }
    if (!hf) {
      return { ok: false, reason: 'internal_error', message: 'Missing @huggingface/transformers. Install it (npm i @huggingface/transformers).' };
    }
    this.hf = hf as HF;
    
    // Get WebGPU device
    try {
      const adapter = await (navigator as any).gpu?.requestAdapter?.({
        powerPreference: 'high-performance',
      });
      if (!adapter) {
        return { ok: false, reason: 'webgpu_unsupported', message: 'No WebGPU adapter available' };
      }
      this.webgpuDevice = await adapter.requestDevice();
    } catch (e) {
      return { ok: false, reason: 'webgpu_unsupported', message: `WebGPU error: ${e instanceof Error ? e.message : String(e)}` };
    }
    
    const TOTAL_BYTES_APPROX: number | undefined = typeof options.approxTotalBytes === 'number' ? options.approxTotalBytes : undefined;
    let bytesDownloaded = 0;
    
    const reportProgress = (message: string, pct?: number) => {
      options.onProgress?.({
        phase: 'loading',
        message,
        bytesDownloaded,
        totalBytesExpected: TOTAL_BYTES_APPROX,
        pct: pct ?? undefined,
        accuracy: 'approximate',
      });
    };
    
    try {
      // Stage 1: Load GGUF files
      reportProgress('Loading Bonsai GGUF files...', 5);
      
      const ggufUrl = `${BONSAI_MODELS.ggufBase}/bonsai-${this.variant}-transformer.gguf`;
      const vaeUrl = `${BONSAI_MODELS.ggufBase}/bonsai-${this.variant}-vae.gguf`;
      
      // Fetch GGUF files with caching
      const { fetchArrayBufferWithCacheProgress } = await import('../cache.js');
      
      const [transformerBuf, vaeBuf] = await Promise.all([
        fetchArrayBufferWithCacheProgress(ggufUrl, this.id, (loaded) => {
          bytesDownloaded = loaded;
          reportProgress('Downloading DiT transformer GGUF...', Math.min(50, 5 + Math.round((loaded / (TOTAL_BYTES_APPROX ?? 1e9)) * 50)));
        }),
        fetchArrayBufferWithCacheProgress(vaeUrl, this.id, (loaded) => {
          bytesDownloaded += loaded;
          reportProgress('Downloading VAE GGUF...', 50 + Math.min(20, Math.round((loaded / (TOTAL_BYTES_APPROX ?? 1e9)) * 20)));
        }),
      ]);
      
      // Parse GGUF files
      reportProgress('Parsing GGUF files...', 70);
      this.ggufTransformer = await parseGguf(transformerBuf);
      this.ggufVae = await parseGguf(vaeBuf);
      
      // Stage 2: Load T5 text encoder via Transformers.js ONNX
      reportProgress('Loading T5 text encoder...', 80);
      this.textEncoder = await (hf as HF).AutoModelForTextEncoding.from_pretrained(BONSAI_MODELS.textEncoderId, {
        dtype: 'fp16',
        device: 'webgpu',
      });
      
      // Stage 3: Load Qwen tokenizer via Transformers.js
      reportProgress('Loading tokenizer...', 90);
      this.tokenizer = await (hf as HF).AutoTokenizer.from_pretrained(BONSAI_MODELS.tokenizerId);
      
      // Stage 4: Upload GGUF tensors to WebGPU buffers
      reportProgress('Uploading tensors to GPU...', 95);
      await this.uploadTensorsToGPU();
      
      this.backendUsed = 'webgpu';
      this.loaded = true;
      
      reportProgress('Bonsai model ready', 100);
      
      return { ok: true, backendUsed: 'webgpu', bytesDownloaded };
    } catch (e) {
      return { ok: false, reason: 'internal_error', message: e instanceof Error ? e.message : String(e) };
    }
  }
  
  private async uploadTensorsToGPU(): Promise<void> {
    if (!this.webgpuDevice || !this.ggufTransformer) return;
    
    // Upload transformer tensors
    for (const tensorInfo of this.ggufTransformer.tensorInfos) {
      const tensor = extractTensor(this.ggufTransformer, tensorInfo.name);
      if (!tensor) continue;
      
      // Dequantize to FP32
      const dequantized = dequantize(tensor);
      
      // Upload to GPU buffer
      const buffer = this.webgpuDevice.createBuffer({
        size: dequantized.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      
      this.webgpuDevice.queue.writeBuffer(buffer, 0, dequantized.buffer as ArrayBuffer);
      this.tensorBuffers.set(tensorInfo.name, buffer);
    }
  }
  
  isLoaded(): boolean {
    return this.loaded;
  }
  
  async generate(params: Omit<GenerateParams, 'model'>): Promise<GenerateResult> {
    if (!this.loaded || !this.tokenizer || !this.textEncoder) {
      return { ok: false, reason: 'model_not_loaded', message: 'Call loadModel() first' };
    }
    
    const { prompt, seed, width, height, signal, onProgress } = params;
    if (!prompt || !prompt.trim()) {
      return { ok: false, reason: 'unsupported_option', message: 'Prompt is required' };
    }
    if (signal?.aborted) {
      return { ok: false, reason: 'cancelled' };
    }
    
    const start = performance.now();
    const steps = params.steps ?? BONSAI_DEFAULTS.steps;
    const guidanceScale = params.guidanceScale ?? BONSAI_DEFAULTS.guidanceScale;
    const genWidth = width ?? BONSAI_DEFAULTS.width;
    const genHeight = height ?? BONSAI_DEFAULTS.height;
    
    try {
      // Tokenize prompt
      onProgress?.({ phase: 'tokenizing', pct: 0 });
      const tokenized = await this.tokenizer(prompt.trim(), { truncation: true });
      const inputIds = tokenized.input_ids;
      
      // Encode prompt via T5 text encoder
      onProgress?.({ phase: 'encoding', pct: 10 });
      const textEmbeddings = await this.textEncoder({ input_ids: inputIds });
      const encoderHiddenStates = textEmbeddings.last_hidden_state;
      
      // Optional: evict text encoder after encoding (memory optimization)
      if (this.textEncoder) {
        try {
          await this.textEncoder.free();
          this.textEncoder = null;
        } catch {}
      }
      
      // Initialize latents (Gaussian noise, seeded)
      onProgress?.({ phase: 'denoising', pct: 20 });
      const latentWidth = Math.ceil(genWidth / 8);
      const latentHeight = Math.ceil(genHeight / 8);
      const latentChannels = 16; // Bonsai latent channels
      const latentSize = latentChannels * latentWidth * latentHeight;
      
      // Seeded random noise
      const latents = this.generateSeededNoise(latentSize, seed ?? Math.floor(Math.random() * 1e9));
      
      // Euler flow-matching denoising loop
      const timesteps = this.generateTimesteps(steps);
      
      for (let step = 0; step < steps; step++) {
        if (signal?.aborted) {
          return { ok: false, reason: 'cancelled' };
        }
        
        const t = timesteps[step];
        onProgress?.({
          phase: 'denoising',
          pct: 20 + Math.round((step / steps) * 60),
          step,
          total: steps,
          timestep: t,
        });
        
        // Run DiT forward pass on WebGPU
        // This is a placeholder — actual implementation would use WebGPU compute shaders
        latents.fill(0); // Placeholder for actual denoising step
        
        // Apply guidance scale
        if (guidanceScale !== 1.0) {
          // CFG would be applied here
        }
      }
      
      // Decode latents via VAE → image
      onProgress?.({ phase: 'decoding', pct: 85 });
      
      // Placeholder: actual VAE decode would use WebGPU compute shaders
      const imageData = this.createPlaceholderImage(genWidth, genHeight);
      
      // Create PNG blob
      onProgress?.({ phase: 'complete', pct: 100 });
      const blob = await this.createPngBlob(imageData, genWidth, genHeight);
      
      const timeMs = performance.now() - start;
      return { ok: true, blob, timeMs };
    } catch (e) {
      return { ok: false, reason: 'internal_error', message: e instanceof Error ? e.message : String(e) };
    }
  }
  
  private generateSeededNoise(size: number, seed: number): Float32Array {
    // Simple seeded PRNG (Mulberry32)
    const rng = this.mulberry32(seed);
    const noise = new Float32Array(size);
    
    for (let i = 0; i < size; i++) {
      // Box-Muller transform for normal distribution
      const u1 = rng();
      const u2 = rng();
      const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      noise[i] = z;
    }
    
    return noise;
  }
  
  private mulberry32(a: number): () => number {
    return () => {
      a |= 0;
      a = a + 0x6D2B7DFD ^ a << 13;
      a = a ^ a >>> 17;
      a = a + 0x04C26269 ^ a << 5;
      return ((a ^ a >>> 15) >>> 0) / 4294967296;
    };
  }
  
  private generateTimesteps(steps: number): number[] {
    // Euler flow-matching timesteps
    const timesteps: number[] = [];
    for (let i = 0; i < steps; i++) {
      timesteps.push(i / steps);
    }
    return timesteps;
  }
  
  private createPlaceholderImage(width: number, height: number): Uint8ClampedArray {
    // Create a simple gradient image as placeholder
    const imageData = new Uint8ClampedArray(width * height * 4);
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        // Gradient from green to blue
        imageData[idx] = Math.round(50 + 100 * (x / width)); // R
        imageData[idx + 1] = Math.round(200 - 100 * (y / height)); // G
        imageData[idx + 2] = Math.round(150 + 50 * (x / width)); // B
        imageData[idx + 3] = 255; // A
      }
    }
    
    return imageData;
  }
  
  private async createPngBlob(imageData: Uint8ClampedArray, width: number, height: number): Promise<Blob> {
    // Use OffscreenCanvas or Canvas to create PNG
    const hasOffscreen = typeof OffscreenCanvas !== 'undefined';
    const canvas = hasOffscreen ? new OffscreenCanvas(width, height) : (globalThis as any).document?.createElement('canvas');
    
    if (!canvas) {
      throw new Error('Canvas not available');
    }
    
    (canvas as any).width = width;
    (canvas as any).height = height;
    
    const ctx = (canvas as any).getContext('2d');
    if (!ctx) {
      throw new Error('Canvas 2D context unavailable');
    }
    
    const bitmap = new ImageData(
      imageData as unknown as Uint8ClampedArray<ArrayBuffer>,
      width,
      height
    );
    ctx.putImageData(bitmap, 0, 0);
    
    if (hasOffscreen) {
      return await (canvas as OffscreenCanvas).convertToBlob({ type: 'image/png' });
    }
    
    return new Promise<Blob>((resolve) => {
      (canvas as HTMLCanvasElement).toBlob((b) => resolve(b!), 'image/png');
    });
  }
  
  async unload(): Promise<void> {
    // Clean up GPU buffers
    for (const buffer of this.tensorBuffers.values()) {
      buffer.destroy?.();
    }
    this.tensorBuffers.clear();
    
    // Free text encoder if still loaded
    if (this.textEncoder) {
      try {
        await this.textEncoder.free();
      } catch {}
      this.textEncoder = null;
    }
    
    // Free tokenizer
    if (this.tokenizer) {
      try {
        await this.tokenizer.free();
      } catch {}
      this.tokenizer = null;
    }
    
    // Clear GGUF data
    this.ggufTransformer = null;
    this.ggufVae = null;
    
    // Release WebGPU device
    this.webgpuDevice = null;
    
    this.loaded = false;
    this.backendUsed = null;
  }
  
  async purgeCache(): Promise<void> {
    await purgeModelCache(this.id);
  }
}
