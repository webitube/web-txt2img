import type {
  Adapter,
  BackendId,
  Capabilities,
  GenerateParams,
  GenerateResult,
  LoadOptions,
  LoadResult,
} from '../types.js';
import { fetchArrayBufferWithCacheProgress, purgeModelCache } from '../cache.js';

type ORT = typeof import('onnxruntime-web');

// Minimal adapter scaffold. Actual ONNX pipeline is TBD.

export class SDTurboAdapter implements Adapter {
  readonly id = 'sd-turbo' as const;

  private loaded = false;
  private backendUsed: BackendId | null = null;
  private ort: ORT | null = null;
  private sessions: {
    unet?: any;
    text_encoder?: any;
    vae_decoder?: any;
  } = {};
  private tokenizerFn: ((text: string, opts?: any) => Promise<{ input_ids: number[] }>) | null = null;
  private tokenizerProvider: (() => Promise<(text: string, opts?: any) => Promise<{ input_ids: number[] }>>) | null = null;
  private modelBase = 'https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main';
  private localModelBase: string | null = null;

  checkSupport(c: Capabilities): BackendId[] {
    const backends: BackendId[] = [];
    if (c.webgpu) backends.push('webgpu');
    // WASM is assumed available
    backends.push('wasm');
    return backends;
  }

  async load(options: Required<Pick<LoadOptions, 'backendPreference'>> & LoadOptions): Promise<LoadResult> {
    const preferred = options.backendPreference;
    const supported = ['webgpu', 'wasm'] as BackendId[];
    let chosen = preferred.find((b) => supported.includes(b));
    if (!chosen) return { ok: false, reason: 'backend_unavailable', message: 'No viable backend for SD-Turbo' };

    // Resolve model base URL override
    if (options.modelBaseUrl) {
      this.modelBase = options.modelBaseUrl;
      this.localModelBase = options.modelBaseUrl;
    }
    if (options.tokenizerProvider) this.tokenizerProvider = options.tokenizerProvider;

    // Resolve ORT runtime: injected → dynamic import → global
    try {
      let ort: any = options.ort ?? null;
      if (!ort) {
        let ortMod: any = null;
        if (chosen === 'webgpu') {
          ortMod = await import('onnxruntime-web/webgpu').catch(() => null);
        } else {
          // WASM uses the default entry
          ortMod = await import('onnxruntime-web').catch(() => null);
        }
        ort = ortMod && (ortMod.default ?? ortMod);
      }
      if (!ort) {
        const gOrt = (globalThis as any).ort; // fallback if app added <script>
        if (gOrt) ort = gOrt;
      }
      if (!ort) {
        return { ok: false, reason: 'internal_error', message: 'onnxruntime-web not available. Install as a dependency or inject via loadModel({ ort }).' };
      }
      this.ort = ort as ORT;
    } catch (e) {
      return { ok: false, reason: 'internal_error', message: `Failed to load onnxruntime-web: ${e instanceof Error ? e.message : String(e)}` };
    }

    // Placeholder for downloading model assets using Cache Storage
    try {
      options.onProgress?.({ phase: 'loading', message: 'Preparing SD-Turbo model...' });
      this.backendUsed = chosen;

      const ort = this.ort!;
      const opt: any = {
        executionProviders: [chosen],
        enableMemPattern: false,
        enableCpuMemArena: false,
        // Suppress EP assignment warnings (shape ops fall back to CPU by design)
        logSeverityLevel: 3, // Warning level: 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal
        extra: {
          session: {
            disable_prepacking: '1',
            use_device_allocator_for_initializers: '1',
            use_ort_model_bytes_directly: '1',
            use_ort_model_bytes_for_initializers: '1',
          },
        },
      };
      if (chosen === 'webgpu') {
        (opt as any).preferredOutputLocation = { last_hidden_state: 'gpu-buffer' };
      }
      // Configure WASM env if provided, regardless of EP; ORT may still load WASM helpers
      try {
        if (options.wasmPaths) (ort as any).env.wasm.wasmPaths = options.wasmPaths;
        if (typeof options.wasmNumThreads === 'number') (ort as any).env.wasm.numThreads = options.wasmNumThreads;
        if (typeof options.wasmSimd === 'boolean') (ort as any).env.wasm.simd = options.wasmSimd;
      } catch {}

      const models = {
        unet: {
          url: 'unet/model.onnx', sizeMB: 640,
          opt: { freeDimensionOverrides: { batch_size: 1, num_channels: 4, sequence_length: 77 } },
        },
        text_encoder: {
          url: 'text_encoder/model.onnx', sizeMB: 1700,
          opt: { freeDimensionOverrides: { batch_size: 1 } },
        },
        vae_decoder: {
          url: 'vae_decoder/model.onnx', sizeMB: 95,
          opt: { freeDimensionOverrides: { batch_size: 1, num_channels_latent: 4 } },
        },
      } as const;

      // compute base URL
      const base = this.modelBase;

      // Fetch and create sessions with progress
      let bytesDownloaded = 0;
      // Use approximate grand total injected from registry (single source of truth)
      const fallbackTotal = Object.values(models).reduce((acc, m) => acc + m.sizeMB * 1024 * 1024, 0);
      const GRAND_APPROX = (typeof options.approxTotalBytes === 'number' ? options.approxTotalBytes : fallbackTotal);
      options.onProgress?.({
        phase: 'loading',
        message: `starting downloads (~${Math.round(GRAND_APPROX/1024/1024)}MB total)...`,
        bytesDownloaded: 0,
        totalBytesExpected: GRAND_APPROX,
        pct: 0,
        accuracy: 'exact',
      });
      for (const key of Object.keys(models) as Array<keyof typeof models>) {
        const model = models[key];
        options.onProgress?.({ phase: 'loading', message: `downloading ${model.url}...`, bytesDownloaded });
        const expectedTotal = model.sizeMB * 1024 * 1024;
        const buf = await fetchArrayBufferWithCacheProgress(`${base}/${model.url}`, this.id, (loaded, total) => {
          const pct = Math.min(100, Math.round(((bytesDownloaded + loaded) / GRAND_APPROX) * 100));
          options.onProgress?.({
            phase: 'loading',
            message: `downloading ${model.url}...`,
            pct,
            bytesDownloaded: bytesDownloaded + loaded,
            totalBytesExpected: GRAND_APPROX,
            asset: model.url,
            accuracy: 'exact',
          });
        }, expectedTotal);
        bytesDownloaded += buf.byteLength;
        const start = performance.now();
        const sess = await (ort as any).InferenceSession.create(buf, { ...opt, ...(model.opt as any) });
        const ms = performance.now() - start;
        options.onProgress?.({
          phase: 'loading',
          message: `${model.url} ready in ${ms.toFixed(1)}ms`,
          bytesDownloaded,
          totalBytesExpected: GRAND_APPROX,
          asset: model.url,
          accuracy: 'exact',
        });
        (this.sessions as any)[key] = sess;
      }

      this.loaded = true;
      return { ok: true, backendUsed: chosen, bytesDownloaded };
    } catch (e) {
      console.error('[sd-turbo] load error', e);
      return { ok: false, reason: 'internal_error', message: e instanceof Error ? e.message : String(e) };
    }
  }

  isLoaded(): boolean {
    return this.loaded;
  }

  async generate(params: Omit<GenerateParams, 'model'>): Promise<GenerateResult> {
    if (!this.loaded) return { ok: false, reason: 'model_not_loaded', message: 'Call loadModel() first' };

    const { prompt, width = 512, height = 512, signal, onProgress, seed, numInferenceSteps = 1 } = params;
    if (!prompt || !prompt.trim()) return { ok: false, reason: 'unsupported_option', message: 'Prompt is required' };
    // Validate dimensions: must be multiples of 8 (latent space is 1/8 of image space)
    if (width % 8 !== 0 || height % 8 !== 0) {
      return { ok: false, reason: 'unsupported_option', message: 'Width and height must be multiples of 8' };
    }
    // UNet skip connections require latent dims divisible by 2^N (N=downsampling blocks)
    // In practice this means width/height must be multiples of 64 for stable UNet inference
    if (width % 64 !== 0 || height % 64 !== 0) {
      const nearestW = Math.round(width / 64) * 64;
      const nearestH = Math.round(height / 64) * 64;
      return { ok: false, reason: 'unsupported_option', message: `Width and height must be multiples of 64 for UNet compatibility. Try ${nearestW}x${nearestH} instead of ${width}x${height}.` };
    }
    // Reasonable bounds to prevent OOM
    if (width < 64 || height < 64 || width > 2048 || height > 2048) {
      return { ok: false, reason: 'unsupported_option', message: 'Width and height must be between 64 and 2048' };
    }

    const start = performance.now();
    const ort = this.ort!;

    try {
      // Tokenizer (injected or dynamic)
      onProgress?.({ phase: 'tokenizing', pct: 10 });
      if (!this.tokenizerFn) {
        if (this.tokenizerProvider) this.tokenizerFn = await this.tokenizerProvider();
        else this.tokenizerFn = await getTokenizer(this.localModelBase);
      }
      if (signal?.aborted) { onProgress?.({ phase: 'complete', aborted: true as any, pct: 0 }); return { ok: false, reason: 'cancelled' }; }
      const tok = this.tokenizerFn!;
      const { input_ids } = await tok(prompt, { padding: true, max_length: 77, truncation: true, return_tensor: false });

      // Text encoder
      onProgress?.({ phase: 'encoding', pct: 25 });
      const ids = Int32Array.from(input_ids as number[]);
      let encOut: any;
      try {
        encOut = await this.sessions.text_encoder!.run({ input_ids: new (ort as any).Tensor('int32', ids, [1, ids.length]) });
      } catch (e) {
        throw new Error(`text_encoder.run failed: ${e instanceof Error ? e.message : String(e)}`);
      }
      const last_hidden_state = (encOut as any).last_hidden_state ?? encOut;
      if (signal?.aborted) { onProgress?.({ phase: 'complete', aborted: true as any, pct: 0 }); return { ok: false, reason: 'cancelled' }; }

      // Latents - dynamic shape based on requested width/height
      const latent_height = Math.floor(height / 8);
      const latent_width = Math.floor(width / 8);
      const latent_shape = [1, 4, latent_height, latent_width];
      const sigma = 14.6146;
      const vae_scaling_factor = 0.18215;
      const numSteps = Math.max(1, Math.min(numInferenceSteps ?? 1, 50));
      const scheduler = params.scheduler ?? 'euler';

      // Build timestep schedule: evenly spaced from 999 down to 0
      const timesteps = Array.from({ length: numSteps }, (_, i) => Math.round(999 * (1 - i / numSteps)));

      let currentLatent = new (ort as any).Tensor(randn_latents(latent_shape, sigma, seed), latent_shape);

      // DPM++ 2M needs the previous model output for 2nd-order integration
      let prevModelOutput: Float32Array | null = null;
      let prevTimestep = 999;

      // Denoising loop
      onProgress?.({ phase: 'denoising', pct: 40 });
      for (let i = 0; i < numSteps; i++) {
        if (signal?.aborted) { onProgress?.({ phase: 'complete', aborted: true as any, pct: 0 }); return { ok: false, reason: 'cancelled' }; }

        const currentTimestep = timesteps[i];
        const nextTimestep = i < numSteps - 1 ? timesteps[i + 1] : 0;

        // Map timestep to sigma: sigma_t = sigma * (t / 999)
        const currentSigma = sigma * (currentTimestep / 999);
        const nextSigma = sigma * (nextTimestep / 999);

        const latent_model_input = scale_model_inputs(ort as any, currentLatent, currentSigma);

        const tstep = [BigInt(currentTimestep)];
        const feed: Record<string, any> = {
          sample: latent_model_input,
          timestep: new (ort as any).Tensor('int64', tstep as any, [1]),
          encoder_hidden_states: last_hidden_state,
        };
        let out_sample: any;
        try {
          out_sample = await this.sessions.unet!.run(feed);
          out_sample = (out_sample as any).out_sample ?? out_sample;
        } catch (e) {
          throw new Error(`unet.run failed at step ${i + 1}/${numSteps}: ${e instanceof Error ? e.message : String(e)}`);
        }

        // Dispatch to scheduler step function
        currentLatent = schedulerStep(
          scheduler, ort as any, out_sample, currentLatent,
          currentSigma, nextSigma, prevModelOutput, prevTimestep, currentTimestep,
          seed, i
        );

        // Cache model output for DPM++ 2M (needs previous step)
        prevModelOutput = out_sample.data;
        prevTimestep = currentTimestep;

        const pct = 40 + Math.round((i + 1) / numSteps * 55);
        onProgress?.({ phase: 'denoising', pct, step: i + 1, totalSteps: numSteps });
      }

      if (typeof (last_hidden_state as any).dispose === 'function') (last_hidden_state as any).dispose();
      if (signal?.aborted) { onProgress?.({ phase: 'complete', aborted: true as any, pct: 0 }); return { ok: false, reason: 'cancelled' }; }

      // Apply VAE scaling factor to denoised latents before decode
      const vaeLatent = scaleLatent(ort as any, currentLatent, vae_scaling_factor);

      // VAE decode
      onProgress?.({ phase: 'decoding', pct: 95 });
      if (signal?.aborted) { onProgress?.({ phase: 'complete', aborted: true as any, pct: 0 }); return { ok: false, reason: 'cancelled' }; }
      let vaeOut: any;
      try {
        vaeOut = await this.sessions.vae_decoder!.run({ latent_sample: vaeLatent });
      } catch (e) {
        throw new Error(`vae_decoder.run failed: ${e instanceof Error ? e.message : String(e)}`);
      }
      const sample = (vaeOut as any).sample ?? vaeOut;

      if (signal?.aborted) { onProgress?.({ phase: 'complete', aborted: true as any, pct: 0 }); return { ok: false, reason: 'cancelled' }; }
      const blob = await tensorToPngBlob(sample);
      const timeMs = performance.now() - start;
      onProgress?.({ phase: 'complete', pct: 100, timeMs });
      return { ok: true, blob, timeMs };
    } catch (e) {
      console.error('[sd-turbo] generate error', e);
      return { ok: false, reason: 'internal_error', message: e instanceof Error ? e.message : String(e) };
    }
  }

  async unload(): Promise<void> {
    try {
      // Dispose ORT sessions if available and clear references
      try { (this.sessions.unet as any)?.release?.(); } catch {}
      try { (this.sessions.text_encoder as any)?.release?.(); } catch {}
      try { (this.sessions.vae_decoder as any)?.release?.(); } catch {}
    } finally {
      this.sessions = {};
      this.ort = null;
      this.loaded = false;
      this.backendUsed = null;
    }
  }

  async purgeCache(): Promise<void> {
    await purgeModelCache(this.id);
  }
}

// Helpers

function mulberry32(seed: number) {
  let t = seed >>> 0;
  return function () {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function randn_latents(shape: number[], noise_sigma: number, seed?: number) {
  const rand = seed !== undefined ? mulberry32(seed) : Math.random;
  function randn() {
    const u = rand();
    const v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  let size = 1;
  for (const s of shape) size *= s;
  const data = new Float32Array(size);
  for (let i = 0; i < size; i++) data[i] = randn() * noise_sigma;
  return data;
}

function scale_model_inputs(ort: ORT, t: any, sigma: number) {
  const d_i: Float32Array = t.data;
  const d_o = new Float32Array(d_i.length);
  const divi = Math.sqrt(sigma * sigma + 1);
  for (let i = 0; i < d_i.length; i++) d_o[i] = d_i[i] / divi;
  return new (ort as any).Tensor(d_o, t.dims);
}

/**
 * Scheduler dispatcher: routes to the correct step function based on scheduler type.
 * All step functions use sample.dims to preserve latent shape across iterations.
 *
 * Key: the UNet outputs the noise prediction (epsilon). We use the relationship:
 *   pred_original = x_t - sigma * epsilon
 *   denoising_direction = (x_t - pred_original) / sigma = epsilon
 *
 * The original single-step code did: x_next = x_t + epsilon * (-sigma) = pred_original
 * For multi-step we step from sigma_t toward sigma_{t-1} instead of all the way to 0.
 */
function schedulerStep(
  scheduler: string,
  ort: ORT,
  model_output: any,
  sample: any,
  sigma: number,
  nextSigma: number,
  prevModelOutput: Float32Array | null,
  prevTimestep: number,
  currentTimestep: number,
  seed: number | undefined,
  stepIndex: number,
) {
  switch (scheduler) {
    case 'ddim':
      return ddimStep(ort, model_output, sample, sigma, nextSigma);
    case 'dpmpp_2m_karras':
      return dpmpp2mStep(ort, model_output, sample, sigma, nextSigma, prevModelOutput, prevTimestep, currentTimestep, stepIndex);
    case 'euler_ancestral':
      return eulerAncestralStep(ort, model_output, sample, sigma, nextSigma, seed, stepIndex);
    case 'euler':
    default:
      return eulerStep(ort, model_output, sample, sigma, nextSigma);
  }
}

/**
 * Euler step: first-order denoising from current sigma toward next sigma.
 * Does NOT apply VAE scaling — that's done once before VAE decode.
 * Uses sample.dims to preserve latent shape across iterations.
 *
 * Formula: x_{t-1} = x_t + D(x_t, t) * (sigma_{t-1} - sigma_t)
 * where D(x_t, t) = epsilon (the noise prediction from UNet)
 */
function eulerStep(ort: ORT, model_output: any, sample: any, sigma: number, nextSigma: number) {
  const d_o = new Float32Array(sample.data.length);
  const prev_sample = new (ort as any).Tensor(d_o, sample.dims);
  for (let i = 0; i < sample.data.length; i++) {
    // model_output IS epsilon (noise prediction)
    const epsilon = model_output.data[i];
    // Step toward next sigma
    const dt = nextSigma - sigma;
    d_o[i] = sample.data[i] + epsilon * dt;
  }
  return prev_sample;
}

/**
 * DDIM step: deterministic denoising.
 * Uses sample.dims to preserve latent shape across iterations.
 *
 * Formula (eta=0, deterministic):
 *   pred_original = x_t - sigma * epsilon
 *   x_{t-1} = nextSigma * epsilon + (sigma - nextSigma) * pred_original / sigma + x_t * (nextSigma / sigma)
 * Simplified: x_{t-1} = (nextSigma/sigma) * x_t + (1 - nextSigma/sigma) * pred_original
 */
function ddimStep(ort: ORT, model_output: any, sample: any, sigma: number, nextSigma: number) {
  const d_o = new Float32Array(sample.data.length);
  const prev_sample = new (ort as any).Tensor(d_o, sample.dims);
  for (let i = 0; i < sample.data.length; i++) {
    const epsilon = model_output.data[i];
    const pred_original = sample.data[i] - sigma * epsilon;
    // DDIM: interpolate between current noisy sample and predicted clean sample
    const ratio = nextSigma / sigma;
    d_o[i] = ratio * sample.data[i] + (1 - ratio) * pred_original;
  }
  return prev_sample;
}

/**
 * DPM++ 2M (2nd-order multistep) step.
 * Uses the current and previous model outputs for a higher-order integration.
 * First step falls back to Euler (no previous output available).
 * Uses sample.dims to preserve latent shape across iterations.
 *
 * Formula (from DPM-Solver++ paper):
 *   x_{t-1} = x_t + h * D(x_t, t) + (h^2 / 2) * (D(x_t, t) - D(x_{t-1}, t-1)) / h_prev
 * where h = sigma_{t-1} - sigma_t, h_prev = sigma_t - sigma_{t-1_prev}
 */
function dpmpp2mStep(
  ort: ORT,
  model_output: any,
  sample: any,
  sigma: number,
  nextSigma: number,
  prevModelOutput: Float32Array | null,
  prevTimestep: number,
  currentTimestep: number,
  stepIndex: number,
) {
  const d_o = new Float32Array(sample.data.length);
  const prev_sample = new (ort as any).Tensor(d_o, sample.dims);

  // First step: fall back to Euler
  if (stepIndex === 0 || !prevModelOutput) {
    for (let i = 0; i < sample.data.length; i++) {
      const epsilon = model_output.data[i];
      const dt = nextSigma - sigma;
      d_o[i] = sample.data[i] + epsilon * dt;
    }
    return prev_sample;
  }

  // 2nd-order DPM++ 2M step
  const h = nextSigma - sigma; // current step size
  const h_prev = sigma - (sigma + (prevTimestep - currentTimestep) / currentTimestep * sigma); // approximate
  // Simpler: use the ratio of step sizes
  const r = h_prev !== 0 ? h / h_prev : 1;

  for (let i = 0; i < sample.data.length; i++) {
    const epsilon = model_output.data[i];
    const epsilon_prev = prevModelOutput[i];
    // 2nd-order correction: D_2nd = epsilon + r * (epsilon - epsilon_prev) / (1 + r)
    const d_2nd = epsilon + r * (epsilon - epsilon_prev) / (1 + r);
    d_o[i] = sample.data[i] + h * d_2nd;
  }
  return prev_sample;
}

/**
 * Euler Ancestral step: Euler + added noise for diversity.
 * The noise injection makes outputs more varied but less deterministic.
 * Uses sample.dims to preserve latent shape across iterations.
 *
 * Formula: x_{t-1} = x_t + epsilon * (sigma_{t-1} - sigma_t) + noise * sqrt(sigma_{t-1}^2 - sigma_t^2)
 */
function eulerAncestralStep(
  ort: ORT,
  model_output: any,
  sample: any,
  sigma: number,
  nextSigma: number,
  seed: number | undefined,
  stepIndex: number,
) {
  const d_o = new Float32Array(sample.data.length);
  const prev_sample = new (ort as any).Tensor(d_o, sample.dims);

  // Deterministic noise for reproducibility
  const rand = seed !== undefined ? mulberry32(seed + stepIndex * 1000) : Math.random;
  // Ancestral noise scale: sqrt(sigma_{t-1}^2 - sigma_t^2)
  const noiseScale = Math.sqrt(Math.max(nextSigma * nextSigma - sigma * sigma, 0));

  for (let i = 0; i < sample.data.length; i++) {
    const epsilon = model_output.data[i];
    // Euler step
    const dt = nextSigma - sigma;
    let val = sample.data[i] + epsilon * dt;

    // Add ancestral noise (only if sigma is decreasing, i.e., noiseScale > 0)
    if (noiseScale > 0) {
      const u = Math.max(rand(), 1e-10);
      const v = rand();
      const noise = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      val += noise * noiseScale;
    }

    d_o[i] = val;
  }
  return prev_sample;
}

/**
 * Scale all latent values by dividing by the VAE scaling factor.
 * Applied once, after all denoising steps, before VAE decode.
 */
function scaleLatent(ort: ORT, t: any, factor: number) {
  const d_o = new Float32Array(t.data.length);
  for (let i = 0; i < t.data.length; i++) {
    d_o[i] = t.data[i] / factor;
  }
  return new (ort as any).Tensor(d_o, t.dims);
}

async function tensorToPngBlob(t: any): Promise<Blob> {
  // t: [1, 3, H, W]
  const [n, c, h, w] = t.dims;
  const data: Float32Array = t.data;
  const out = new Uint8ClampedArray(w * h * 4);
  let idx = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const r = data[0 * h * w + y * w + x];
      const g = data[1 * h * w + y * w + x];
      const b = data[2 * h * w + y * w + x];
      const clamp = (v: number) => {
        let x = v / 2 + 0.5;
        if (x < 0) x = 0;
        if (x > 1) x = 1;
        return Math.round(x * 255);
      };
      out[idx++] = clamp(r);
      out[idx++] = clamp(g);
      out[idx++] = clamp(b);
      out[idx++] = 255;
    }
  }
  const imageData = new ImageData(out, w, h);
  const hasOffscreen = typeof OffscreenCanvas !== 'undefined';
  const canvas = hasOffscreen ? new OffscreenCanvas(w, h) : document.createElement('canvas');
  (canvas as any).width = w; (canvas as any).height = h;
  const ctx = (canvas as any).getContext('2d');
  if (!ctx) throw new Error('Canvas 2D context unavailable');
  ctx.putImageData(imageData, 0, 0);
  const hasHTMLCanvas = typeof (globalThis as any).HTMLCanvasElement !== 'undefined';
  if (hasHTMLCanvas && (canvas as any) instanceof (globalThis as any).HTMLCanvasElement) {
    return await new Promise<Blob>((resolve) => (canvas as HTMLCanvasElement).toBlob((b) => resolve(b!), 'image/png'));
  }
  return await (canvas as OffscreenCanvas).convertToBlob({ type: 'image/png' });
}

let _tokInstance: any = null;
async function getTokenizer(localBase?: string | null): Promise<any> {
  if (_tokInstance) return (text: string, opts: any) => _tokInstance(text, opts);
  // Prefer a global AutoTokenizer (if host app preloaded it), else dynamic import.
  const g: any = globalThis as any;
  
  // Configure env for local or remote loading based on localBase
  const useLocal = !!localBase;
  if (g.env) {
    g.env.allowLocalModels = useLocal;
    g.env.allowRemoteModels = !useLocal;
    if (useLocal) {
      g.env.localModelPath = localBase;
    } else {
      g.env.remoteHost = 'https://huggingface.co/';
      g.env.remotePathTemplate = '{model}/resolve/{revision}/';
    }
  }
  
  if (g.AutoTokenizer && typeof g.AutoTokenizer.from_pretrained === 'function') {
    _tokInstance = await g.AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch16');
    _tokInstance.pad_token_id = 0;
    return (text: string, opts: any) => _tokInstance(text, opts);
  }
  let AutoTokenizerMod: any = null;
  let env: any = null;
  try {
    const mod = await import('@xenova/transformers');
    AutoTokenizerMod = (mod as any).AutoTokenizer;
    env = (mod as any).env;
  } catch {
    try {
      const spec = '@huggingface/transformers';
      const mod2 = await import(/* @vite-ignore */ spec);
      AutoTokenizerMod = (mod2 as any).AutoTokenizer;
      env = (mod2 as any).env;
    } catch {
      throw new Error('Failed to load a tokenizer. Install @xenova/transformers or provide tokenizerProvider in loadModel options.');
    }
  }
  
  // Configure env for local or remote loading based on localBase
  if (env) {
    env.allowLocalModels = useLocal;
    env.allowRemoteModels = !useLocal;
    if (useLocal) {
      env.localModelPath = localBase;
    } else {
      env.remoteHost = 'https://huggingface.co/';
      env.remotePathTemplate = '{model}/resolve/{revision}/';
    }
  }
  
  // Load tokenizer with explicit options based on local/remote
  const loadOpts = useLocal
    ? { local_files_only: true }
    : { local_files_only: false, revision: 'main' };
  
  _tokInstance = await AutoTokenizerMod.from_pretrained('Xenova/clip-vit-base-patch16', loadOpts);
  _tokInstance.pad_token_id = 0;
  return (text: string, opts: any) => _tokInstance(text, opts);
}
