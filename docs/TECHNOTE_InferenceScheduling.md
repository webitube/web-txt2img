# Tech Note: Inference Scheduling for Diffusion Models

> **Purpose:** Document the mathematical foundations, implementation patterns, and adaptation notes for inference schedulers in the WebBonsai text-to-image pipeline.
>
> **Reference Implementation:** SD.Next (`G:\Dev\sdnext`) — a production-grade, battle-tested scheduler system supporting 100+ schedulers across classical, flow-matching, and Res4Lyf families.
>
> **Target:** `web-txt2img` — a browser-based ONNX Runtime Web inference pipeline for SD-Turbo and related models.

---

## Table of Contents

1. [Background: Diffusion as an ODE/SDE](#1-background-diffusion-as-an-odesde)
2. [Core Mathematical Formulas](#2-core-mathematical-formulas)
3. [Sigma Schedules](#3-sigma-schedules)
4. [Solver Families](#4-solver-families)
5. [Flow Matching](#5-flow-matching)
6. [Noise Sampling Strategies](#6-noise-sampling-strategies)
7. [Timestep Shifting](#7-timestep-shifting)
8. [Adaptation Notes for WebBonsai](#8-adaptation-notes-for-webbonsai)
9. [Relevant Files & Notable Sections](#9-relevant-files--notable-sections)

---

## 1. Background: Diffusion as an ODE/SDE

Diffusion models learn to reverse a gradual noising process. At inference time, a **scheduler** (also called a sampler) determines how to step from pure noise to a clean sample.

### Forward Process (Training)

The forward diffusion process adds Gaussian noise over `T` timesteps:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \cdot x_{t-1}, \beta_t \cdot I)$$

Where:
- $\beta_t$ is the variance schedule (beta schedule)
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ (cumulative product)

This gives the closed-form:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \cdot x_0, (1 - \bar{\alpha}_t) \cdot I)$$

### Reverse Process (Inference)

The reverse process can be formulated as:

**ODE (deterministic):**
$$\frac{dx_t}{dt} = f(x_t, t) + g(t) \cdot u_\theta(x_t, t)$$

**SDE (stochastic):**
$$dx_t = [f(x_t, t) + g(t) \cdot u_\theta(x_t, t)] dt + |g(t)| dw_t$$

Where $u_\theta(x_t, t)$ is the model's velocity/score prediction.

---

## 2. Core Mathematical Formulas

### 2.1 Sigma Parameterization

The noise level at timestep $t$ is parameterized by $\sigma_t$:

$$\sigma_t = \sqrt{\frac{1 - \bar{\alpha}_t}{\bar{\alpha}_t}}$$

In practice, schedulers work with a discrete sequence of sigmas $\{\sigma_0, \sigma_1, ..., \sigma_T\}$ where $\sigma_0 \gg 1$ (noisy) and $\sigma_T \approx 0$ (clean).

### 2.2 Prediction Types

Models are trained to predict one of three targets:

| Prediction Type | Formula | Description |
|----------------|---------|-------------|
| **epsilon** | $\epsilon_\theta(x_t, t)$ | Predict the added noise |
| **v-prediction** | $v_\theta(x_t, t) = \alpha_t \cdot \epsilon - \sigma_t \cdot x_0$ | Velocity prediction (SDXL default) |
| **sample (x-prediction)** | $\hat{x}_0$ | Predict the clean sample directly |
| **flow_prediction** | $v_\theta(x_t, t) = \hat{x}_0 - x_t$ | Flow matching velocity |

Converting between prediction types:

$$\hat{x}_0 = \frac{\alpha_t \cdot x_t - \sigma_t \cdot \epsilon_\theta(x_t, t)}{\sqrt{\alpha_t^2 + \sigma_t^2}}$$

$$\epsilon_\theta = \alpha_t \cdot v_\theta + \sigma_t \cdot x_t$$

### 2.3 Model Input Scaling

Before feeding to the UNet, the input is scaled:

$$x_{\text{scaled}} = \frac{x_t}{\sqrt{\sigma_t^2 + 1}}$$

This normalizes the input regardless of the noise level.

---

## 3. Sigma Schedules

### 3.1 Linear / Scaled Linear (Classic)

```python
# Linear
betas = linspace(beta_start, beta_end, T)

# Scaled Linear (SD latent diffusion)
betas = linspace(sqrt(beta_start), sqrt(beta_end), T) ** 2
```

Then derive sigmas from alphas_cumprod:

$$\sigma_t = \sqrt{\frac{1 - \bar{\alpha}_t}{\bar{\alpha}_t}}$$

### 3.2 Karras Schedule

From the EDM paper (https://arxiv.org/abs/2206.00364):

$$\sigma_i = \frac{\sigma_{\min}^{1/\rho} + \frac{i}{N-1}(\sigma_{\max}^{1/\rho} - \sigma_{\min}^{1/\rho})}{(\sigma_{\max}^{1/\rho} + \frac{i}{N-1}(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho}))}^\rho$$

Default $\rho = 7.0$. Places more steps at higher noise levels.

### 3.3 Exponential Schedule

$$\sigma_i = \exp\left(\log(\sigma_{\max}) + \frac{i}{N-1}(\log(\sigma_{\min}) - \log(\sigma_{\max}))\right)$$

Uniform spacing in log-space.

### 3.4 Beta Schedule

From "Beta Sampling is All You Need" (Lee et al., 2024):

$$\sigma_i = \sigma_{\min} + \text{Beta}^{-1}\left(1 - \frac{i}{N-1}; \alpha, \beta\right) \cdot (\sigma_{\max} - \sigma_{\min})$$

Uses the inverse CDF (PPF) of a Beta distribution for non-uniform sampling.

### 3.5 Lu's Lambda Schedule

From DPM-Solver (Lu et al., 2022):

$$\lambda_i = \left(\lambda_{\max}^{1/\rho} + \frac{i}{N-1}(\lambda_{\min}^{1/\rho} - \lambda_{\max}^{1/\rho})\right)^\rho$$

Where $\lambda = \log(\sigma)$, then $\sigma = \exp(\lambda)$.

---

## 4. Solver Families

### 4.1 Euler (First-Order)

The simplest denoising step:

$$x_{t-1} = x_t + \epsilon_\theta(x_t, t) \cdot (\sigma_{t-1} - \sigma_t)$$

Or equivalently:

$$x_{t-1} = x_t + D(x_t, t) \cdot \Delta\sigma$$

Where $D(x_t, t) = \epsilon_\theta$ is the denoising direction.

**Reference:** `sd-turbo.ts` → `eulerStep()`

### 4.2 Euler Ancestral (Euler + Noise)

Adds stochastic noise for diversity:

$$x_{t-1} = x_t + \epsilon_\theta \cdot (\sigma_{t-1} - \sigma_t) + \mathcal{N}(0, I) \cdot \sqrt{\sigma_{t-1}^2 - \sigma_t^2}$$

**Reference:** `sd-turbo.ts` → `eulerAncestralStep()`

### 4.3 DDIM (Denoising Diffusion Implicit Models)

Deterministic interpolation:

$$\text{pred\_original} = x_t - \sigma_t \cdot \epsilon_\theta$$

$$x_{t-1} = \frac{\sigma_{t-1}}{\sigma_t} \cdot x_t + \left(1 - \frac{\sigma_{t-1}}{\sigma_t}\right) \cdot \text{pred\_original}$$

**Reference:** `sd-turbo.ts` → `ddimStep()`

### 4.4 DPM-Solver (High-Order Multistep)

#### DPM-Solver-2 (Midpoint)

Two-stage midpoint method:

$$\sigma_{\text{mid}} = \exp(\log(\sigma_t) \cdot 0.5 + \log(\sigma_{t-1}) \cdot 0.5)$$

$$d = \frac{x_t - \hat{x}_0}{\sigma_t}$$

$$x_{\text{intermediate}} = x_t + d \cdot (\sigma_{\text{mid}} - \sigma_t)$$

$$x_{t-1} = x_{\text{intermediate}} + d_{\text{mid}} \cdot (\sigma_{t-1} - \sigma_{\text{mid}})$$

#### DPM-Solver-2A (Ancestral)

Similar to DPM-Solver-2 but with noise injection:

$$s_u = \min\left(\sigma_{t-1}, \sqrt{\frac{\sigma_{t-1}^2(\sigma_t^2 - \sigma_{t-1}^2)}{\sigma_t^2}}\right)$$

$$s_d = \sqrt{\sigma_{t-1}^2 - s_u^2}$$

Noise is added scaled by $s_u \cdot s_{\text{noise}}$.

#### DPM-Solver++(2M) (2nd-Order Multistep)

Uses the previous model output for higher-order accuracy:

Define:
- $t = -\log(\sigma)$, $t_{\text{next}} = -\log(\sigma_{\text{next}})$
- $h = t_{\text{next}} - t$
- $\sigma_{\text{fn}}(\tau) = \exp(-\tau)$

First step (no previous output):
$$x_{t-1} = \frac{\sigma_{\text{fn}}(t_{\text{next}})}{\sigma_{\text{fn}}(t)} \cdot x_t - \text{expm1}(-h) \cdot \hat{x}_0$$

Subsequent steps:
$$h_{\text{last}} = t - (-\log(\sigma_{\text{prev}}))$$

$$r = \frac{h_{\text{last}}}{h}$$

$$\text{denoised}_d = \left(1 + \frac{1}{2r}\right) \cdot \hat{x}_0 - \frac{1}{2r} \cdot \hat{x}_0^{\text{prev}}$$

$$x_{t-1} = \frac{\sigma_{\text{fn}}(t_{\text{next}})}{\sigma_{\text{fn}}(t)} \cdot x_t - \text{expm1}(-h) \cdot \text{denoised}_d$$

**Reference:** `sd-turbo.ts` → `dpmpp2mStep()`

#### DPM-Solver++(SDE)

Combines deterministic stepping with stochastic noise at each step. Two-phase approach:
1. Euler step to intermediate point
2. Add controlled noise based on sigma gap

### 4.5 UniPC (Unified Predictor-Corrector)

Uses a unified formulation that works for both ODE and SDE. Maintains a buffer of previous model outputs and uses polynomial extrapolation.

**Key difference from DPM-Solver:** Uses a different coefficient computation based on the principle of uniform prediction.

### 4.6 DEIS (Denosing Exponential Integrator Solver)

Exponential integrator approach, particularly effective for stiff ODEs. Uses log-rho parameterization for step sizes.

---

## 5. Flow Matching

Flow matching is an alternative formulation to traditional diffusion, where the model learns a vector field that transports noise to data.

### 5.1 Core Concept

Instead of learning to denoise, the model learns a velocity field:

$$v_\theta(x_t, t) = \hat{x}_0 - x_t$$

The ODE becomes:

$$\frac{dx_t}{dt} = v_\theta(x_t, t)$$

### 5.2 Flow Sigma Computation

For flow matching models, sigmas are computed differently:

$$\sigma_t = \frac{t}{T}$$

Where $t \in [0, T]$ and $T$ is the number of training timesteps (typically 1000).

### 5.3 Timestep Shifting

Flow matching uses a shifting function to concentrate steps where they matter most:

$$\sigma_{\text{shifted}} = \frac{\text{shift} \cdot \sigma}{1 + (\text{shift} - 1) \cdot \sigma}$$

For SD3, typical shift = 3.0–10.0. For FLUX, dynamic shifting is used.

### 5.4 Dynamic Shifting (FLUX)

From the FLUX architecture, shift is computed based on image resolution:

$$\text{mu} = \log\left(\frac{\text{image\_seq\_len}}{\text{base\_image\_seq\_len}}\right)$$

$$\sigma_{\text{shifted}} = \frac{\exp(\text{mu})}{\exp(\text{mu}) + (1/\sigma - 1)^\sigma}$$

Where:
- `base_image_seq_len` = 36864 (512×512 / 16²)
- `max_image_seq_len` = 4096 (for higher resolutions)

### 5.5 FlowMatch DPM-Solver

The `FlowMatchDPMSolverMultistepScheduler` in sdnext adapts DPM-Solver for flow matching by:

1. Computing flow sigmas instead of beta-derived sigmas
2. Converting model output from flow prediction to x0:
   $$\hat{x}_0 = x_t - \sigma_t \cdot v_\theta(x_t, t)$$
3. Using the same multistep integration formulas with flow-adapted sigmas

**Reference:** `modules/schedulers/scheduler_dpm_flowmatch.py`

---

## 6. Noise Sampling Strategies

### 6.1 Random Noise (Standard)

$$\epsilon \sim \mathcal{N}(0, I)$$

Simple, but can cause instability in SDE methods.

### 6.2 Brownian Bridge Noise

Uses a BrownianTree to generate temporally correlated noise:

```python
class BrownianTreeNoiseSampler:
    def __init__(self, x, sigma_min, sigma_max, seed=None):
        self.tree = BatchedBrownianTree(x, t0=sigma_min, t1=sigma_max, seed=seed)
    
    def __call__(self, sigma, sigma_next):
        return self.tree(sigma, sigma_next) / sqrt(|sigma_next - sigma|)
```

**Benefits:**
- More stable convergence for SDE methods
- Reduces "jumping around" in latent space
- Uses the generation seed for reproducibility

**Reference:** `scheduler_dpm_flowmatch.py` → `BrownianTreeNoiseSampler`

### 6.3 Seeded Deterministic Noise

For browser-based inference (no GPU random), use a seeded PRNG:

```javascript
function mulberry32(seed) {
  let t = seed >>> 0;
  return function() {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (t >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
```

Then use Box-Muller transform for normal distribution:

$$z_0 = \sqrt{-2\ln(u)} \cdot \cos(2\pi v)$$

**Reference:** `sd-turbo.ts` → `mulberry32()`, `randn_latents()`

---

## 7. Timestep Shifting

### 7.1 Static Shifting

$$\sigma' = \frac{\text{shift} \cdot \sigma}{1 + (\text{shift} - 1) \cdot \sigma}$$

Higher shift values concentrate more steps at lower noise levels (finer details).

### 7.2 Dynamic Shifting

Used by FLUX and some SD3 variants. The shift is computed based on the image resolution:

$$\text{shift} = \text{clamp}\left(\text{base\_shift} + \frac{\text{max\_shift} - \text{base\_shift}}{\text{base\_image\_seq\_len}} \cdot \text{image\_seq\_len}, \text{base\_shift}, \text{max\_shift}\right)$$

Then applied via the `time_shift()` function:

$$\sigma' = \frac{\exp(\text{mu})}{\exp(\text{mu}) + (1/\sigma - 1)^{\text{sigma}}}$$

### 7.3 When to Use Shifting

| Model Type | Shifting | Typical Shift |
|-----------|----------|--------------|
| SD 1.5 / SD-Turbo | None | N/A |
| SDXL | Optional | 1.0–3.0 |
| SD3 | Static | 3.0–10.0 |
| FLUX | Dynamic | base=0.5, max=1.15 |

---

## 8. Adaptation Notes for WebBonsai

### 8.1 Current State

The current `sd-turbo.ts` implements:
- ✅ Euler (first-order)
- ✅ Euler Ancestral (stochastic)
- ✅ DDIM (deterministic interpolation)
- ✅ DPM++ 2M (2nd-order multistep)
- ✅ Sigma-based timestep scheduling
- ✅ Seeded deterministic noise (Mulberry32 + Box-Muller)
- ✅ Model input scaling
- ✅ VAE latent scaling

### 8.2 Recommended Enhancements (Priority Order)

#### P0: Sigma Schedule Options

Add Karras and exponential sigma schedules for improved quality:

```typescript
function computeSigmas(numSteps: number, schedule: string, sigma: number): number[] {
  switch (schedule) {
    case 'karras':
      return computeKarrasSigmas(sigma, 1e-3, numSteps);
    case 'exponential':
      return computeExponentialSigmas(sigma, 1e-3, numSteps);
    default:
      return computeLinearSigmas(sigma, numSteps);
  }
}
```

**Reference:** `scheduler_dpm_flowmatch.py` → `_convert_to_karras()`, `_convert_to_exponential()`

#### P1: Flow Matching Support

For SD3/FLUX-style models, add flow sigma computation:

```typescript
function computeFlowSigmas(numSteps: number, shift: number, numTrainTimesteps: number = 1000): number[] {
  const sigmas = Array.from({ length: numSteps }, (_, i) => 1 - i / numSteps);
  return sigmas.map(s => (shift * s) / (1 + (shift - 1) * s));
}
```

**Reference:** `scheduler_dpm_flowmatch.py` → `set_timesteps()`, lines 200-250

#### P2: Improved DPM++ 2M

The current DPM++ 2M implementation uses an approximate `h_prev` computation. Replace with the proper formulation:

```typescript
// Current (approximate):
const h_prev = sigma - (sigma + (prevTimestep - currentTimestep) / currentTimestep * sigma);

// Correct (from sdnext):
const t = -Math.log(sigma);
const t_next = -Math.log(nextSigma);
const h = t_next - t;
const sigma_fn = (tau: number) => Math.exp(-tau);
// First step:
x_next = (sigma_fn(t_next) / sigma_fn(t)) * x_t - (-h).expm1() * x0_hat;
// Subsequent steps:
const h_last = t - (-Math.log(prevSigma));
const r = h_last / h;
const denoised_d = (1 + 1 / (2 * r)) * x0_hat - (1 / (2 * r)) * x0_hat_prev;
x_next = (sigma_fn(t_next) / sigma_fn(t)) * x_t - (-h).expm1() * denoised_d;
```

**Reference:** `scheduler_dpm_flowmatch.py` → `step()`, lines 450-500 (dpmsolver++2M branch)

#### P3: Scheduler Hijacking Pattern

Adopt sdnext's pattern of patching `set_timesteps` to support custom timestep arrays:

```typescript
interface SchedulerConfig {
  setTimesteps(numSteps: number, customTimesteps?: number[], customSigmas?: number[]): void;
  step(modelOutput: Float32Array, sample: Float32Array, sigma: number, nextSigma: number): Float32Array;
}
```

**Reference:** `sd_hijack_schedulers.py` → `_prepare_custom_schedule_from_sigmas()`, `_prepare_custom_schedule_from_timesteps()`

#### P4: Brownian Bridge Noise (for SDE methods)

For SDE-based schedulers, replace random noise with temporally correlated noise. A simplified Brownian bridge can be implemented in JavaScript:

```typescript
class SimpleBrownianBridge {
  private cache: Map<string, number> = new Map();
  
  constructor(seed: number) {
    this.rng = mulberry32(seed);
  }
  
  getNoise(sigma: number, sigmaNext: number): number {
    const key = `${sigma}-${sigmaNext}`;
    if (this.cache.has(key)) return this.cache.get(key)!;
    const noise = this.generateBrownianStep(sigma, sigmaNext);
    this.cache.set(key, noise);
    return noise;
  }
}
```

**Reference:** `scheduler_dpm_flowmatch.py` → `BatchedBrownianTree`, `BrownianTreeNoiseSampler`

#### P5: Heun and DPM-Solver-2

Add Heun (2nd-order Runge-Kutta) and DPM-Solver-2 for models that benefit from predictor-corrector methods:

```typescript
function heunStep(ort: ORT, modelOutput: any, sample: any, sigma: number, nextSigma: number) {
  // Predictor (Euler step)
  const epsilon = modelOutput.data;
  const dt = nextSigma - sigma;
  const predicted = sample.data.map((s, i) => s + epsilon[i] * dt);
  
  // Corrector (trapezoidal rule)
  // Would require a second model evaluation at the predicted point
  // For ONNX Web, this doubles the UNet calls — use with caution
}
```

### 8.3 Architecture Recommendations

#### Scheduler Registry Pattern

Mirror sdnext's `SamplerData` namedtuple approach:

```typescript
interface SchedulerInfo {
  name: string;
  aliases: string[];
  createStepFn: (config: SchedulerConfig) => SchedulerStepFunction;
  config: SchedulerConfig;
}

const SCHEDULER_REGISTRY: SchedulerInfo[] = [
  { name: 'euler', aliases: ['euler_discrete'], createStepFn: createEulerStep, config: { /* ... */ } },
  { name: 'dpmpp_2m', aliases: ['dpm++_2m'], createStepFn: createDpmpp2mStep, config: { /* ... */ } },
  // ...
];
```

#### Config-Driven Scheduler Creation

Like sdnext's `config` dictionary in `sd_samplers_diffusers.py`, define scheduler parameters as data:

```typescript
const SCHEDULER_CONFIGS: Record<string, Partial<SchedulerConfig>> = {
  'euler': { stepsOffset: 0, rescaleBetasZeroSNR: false, timestepSpacing: 'linspace' },
  'dpmpp_2m': { thresholding: false, sampleMaxValue: 1.0, algorithmType: 'dpmsolver++', 
                solverType: 'midpoint', lowerOrderFinal: true, useKarrasSigmas: false,
                finalSigmasType: 'zero', timestepSpacing: 'linspace', solverOrder: 2 },
  // ...
};
```

### 8.4 Model Compatibility Matrix

| Model | Prediction Type | Sigma Schedule | Recommended Scheduler |
|-------|---------------|---------------|---------------------|
| SD-Turbo (1-step) | epsilon | Linear | Euler (1 step) |
| SD 1.5 | epsilon | Linear / Karras | DPM++ 2M, Euler |
| SDXL | v-prediction | Linear | DPM++ 2M, Euler |
| SD3 | flow_prediction | Flow (shift=3-10) | FlowMatch DPM++ 2M |
| FLUX | flow_prediction | Flow (dynamic shift) | FlowMatch Euler, FlowMatch DPM++ 2M |

---

## 9. Relevant Files & Notable Sections

### SD.Next Reference (`G:\Dev\sdnext`)

| File | Description | Key Sections |
|------|-------------|-------------|
| `modules/schedulers/scheduler_dpm_flowmatch.py` | Flow-matching DPM solver with Brownian noise | Lines 1-100: BrownianTree noise sampler; Lines 100-200: `__init__` and config; Lines 200-400: `set_timesteps()` with all sigma schedule conversions; Lines 400-700: `step()` with all algorithm branches |
| `modules/sd_samplers_diffusers.py` | Scheduler registry, config definitions, `DiffusionSampler` wrapper | Lines 1-100: imports and config dicts; Lines 100-200: sampler registration with `SamplerData`; Lines 200+: `DiffusionSampler` class |
| `modules/sd_samplers.py` | Scheduler selection, flow/discrete detection, fallback logic | Lines 1-50: `find_sampler`, `list_samplers`; Lines 50-150: `create_sampler()` with flow/discrete validation; Lines 150+: `restore_default()` |
| `modules/sd_hijack_schedulers.py` | Runtime `set_timesteps` patching, custom schedule support | Lines 1-50: `_patch_scheduler_set_timesteps`; Lines 50-150: `_prepare_custom_schedule_from_sigmas`; Lines 150-200: `_prepare_custom_schedule_from_timesteps`; Lines 200+: `_invert_unipc_timesteps` |
| `modules/sd_samplers_common.py` | Shared utilities, flow model detection, VAE approximation | Lines 1-30: `SamplerData` namedtuple, `flow_models` list; Lines 30+: `single_sample_to_image()` |
| `modules/schedulers/scheduler_unipc_flowmatch.py` | Flow-matching UniPC variant | Full file: flow-adapted UniPC implementation |
| `modules/schedulers/scheduler_flashflow.py` | Flash FlowMatch Euler | Full file: optimized flow Euler for speed |
| `modules/schedulers/scheduler_ersde.py` | ER-SDE scheduler | Full file: exponential integrator SDE |
| `modules/res4lyf/` | Res4Lyf scheduler family | Multiple files: RK variants, RES solvers, sigma profiles |

### WebBonsai Target (`G:\Dev\WebBonsai\web-txt2img`)

| File | Description | Key Sections |
|------|-------------|-------------|
| `packages/web-txt2img/src/adapters/sd-turbo.ts` | Current ONNX Web inference pipeline | Lines 1-100: class definition, `load()`; Lines 100-200: `generate()` with denoising loop; Lines 200-350: scheduler step functions (Euler, DDIM, DPM++ 2M, Euler Ancestral); Lines 350+: helper functions |
| `packages/web-txt2img/src/types.ts` | TypeScript type definitions | `GenerateParams`, `SchedulerConfig` interfaces |
| `packages/web-txt2img/src/adapters/` | Adapter directory | All model adapter implementations |

---

## Appendix A: Quick Reference — Sigma Conversion Table

| Schedule | Formula | Parameters |
|----------|---------|-----------|
| Linear | $\sigma_t = \sqrt{(1-\bar{\alpha}_t)/\bar{\alpha}_t}$ | $\beta_{\text{start}}$, $\beta_{\text{end}}$ |
| Karras | $\sigma_i = (\sigma_{\min}^{1/\rho} + \frac{i}{N-1}(\sigma_{\max}^{1/\rho} - \sigma_{\min}^{1/\rho}))^\rho$ | $\rho=7.0$ |
| Exponential | $\sigma_i = \exp(\log(\sigma_{\max}) + \frac{i}{N-1}(\log(\sigma_{\min}) - \log(\sigma_{\max})))$ | — |
| Flow | $\sigma_t = t/T$ | $T=1000$ |
| Flow (shifted) | $\sigma' = \frac{s \cdot \sigma}{1 + (s-1) \cdot \sigma}$ | shift $s$ |
| Flow (dynamic) | $\sigma' = \frac{e^\mu}{e^\mu + (1/\sigma - 1)^\sigma}$ | $\mu$, $\sigma$ |

## Appendix B: Quick Reference — Solver Order & Quality

| Solver | Order | Steps for Quality | Speed | Stability |
|--------|-------|-------------------|-------|-----------|
| Euler | 1st | 20-50 | ⚡⚡⚡ | Good |
| DDIM | 1st | 20-50 | ⚡⚡⚡ | Good |
| DPM-Solver-2 | 2nd | 10-20 | ⚡⚡ | Very Good |
| DPM++ 2M | 2nd | 10-20 | ⚡⚡ | Excellent |
| DPM++ 3M | 3rd | 5-15 | ⚡⚡ | Excellent |
| Heun | 2nd | 10-20 | ⚡ | Good (2× UNet calls) |
| UniPC | Variable | 5-15 | ⚡⚡⚡ | Excellent |
| DEIS | 2nd | 10-20 | ⚡⚡ | Very Good |

---

*Generated: 2026-07-06*
*Reference: SD.Next (vladmandic/sdnext), Diffusers library, DPM-Solver paper (Lu et al., 2022), EDM paper (Karras et al., 2022)*
