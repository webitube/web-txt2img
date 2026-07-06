# Plan: Upgrade Inference Scheduler

> **Goal:** Modernize the inference scheduler system in `web-txt2img` by adopting proven patterns from SD.Next, enabling support for advanced sigma schedules, flow matching models, and a pluggable scheduler architecture.
>
> **Reference:** SD.Next (`G:\Dev\sdnext`) — production-grade scheduler system with 100+ schedulers
> **Target:** `packages/web-txt2img/src/adapters/sd-turbo.ts` and related files
> **Created:** 2026-07-06

---

## Table of Contents

- [Plan: Upgrade Inference Scheduler](#plan-upgrade-inference-scheduler)
  - [Table of Contents](#table-of-contents)
  - [Phase 1: Foundation — Scheduler Types \& Registry](#phase-1-foundation--scheduler-types--registry)
    - [TODO List](#todo-list)
  - [Phase 2: Sigma Schedule System](#phase-2-sigma-schedule-system)
    - [TODO List](#todo-list-1)
  - [Phase 3: Core Solver Improvements](#phase-3-core-solver-improvements)
    - [TODO List](#todo-list-2)
  - [Phase 4: Flow Matching Support](#phase-4-flow-matching-support)
    - [TODO List](#todo-list-3)
  - [Phase 5: Advanced Features](#phase-5-advanced-features)
    - [TODO List](#todo-list-4)
  - [Phase 6: Testing \& Validation](#phase-6-testing--validation)
    - [TODO List](#todo-list-5)
  - [File Structure After Implementation](#file-structure-after-implementation)
  - [Dependencies \& References](#dependencies--references)
  - [Risk Assessment](#risk-assessment)
  - [Success Criteria](#success-criteria)

---

## Phase 1: Foundation — Scheduler Types & Registry

**Objective:** Create a type-safe, extensible scheduler registry that mirrors SD.Next's `SamplerData` pattern.

**Duration:** ~2-3 hours

### TODO List

- [ ] **1.1** Create `packages/web-txt2img/src/scheduler/types.ts`
  - [ ] Define `SchedulerConfig` interface with all configurable parameters
    ```typescript
    export interface SchedulerConfig {
      // Beta schedule
      betaStart?: number;
      betaEnd?: number;
      betaSchedule?: 'linear' | 'scaled_linear';
      numTrainTimesteps?: number;
      
      // Sigma schedule
      useKarrasSigmas?: boolean;
      useExponentialSigmas?: boolean;
      sigmaSchedule?: 'karras' | 'exponential' | 'beta' | 'lambdas' | null;
      
      // Solver settings
      solverOrder?: number;
      solverType?: 'midpoint' | 'heun';
      algorithmType?: string;
      lowerOrderFinal?: boolean;
      
      // Flow matching
      useFlowSigmas?: boolean;
      shift?: number;
      useDynamicShifting?: boolean;
      baseShift?: number;
      maxShift?: number;
      
      // Noise settings
      sNoise?: number;
      useNoiseSampler?: boolean;
      
      // Other
      predictionType?: 'epsilon' | 'v_prediction' | 'flow_prediction';
      timestepSpacing?: 'linspace' | 'leading' | 'trailing';
      finalSigmasType?: 'zero' | 'sigma_min';
    }
    ```
  - [ ] Define `SchedulerStepFunction` type signature
    ```typescript
    export type SchedulerStepFunction = (
      modelOutput: Float32Array,
      sample: Float32Array,
      sigma: number,
      nextSigma: number,
      state: SchedulerState
    ) => Float32Array;
    ```
  - [ ] Define `SchedulerState` interface for multistep solvers
    ```typescript
    export interface SchedulerState {
      stepIndex: number;
      prevModelOutput: Float32Array | null;
      prevSigma: number | null;
      modelOutputs: Float32Array[]; // buffer for multistep
    }
    ```
  - [ ] Define `SchedulerInfo` interface (mirrors SD.Next's `SamplerData`)
    ```typescript
    export interface SchedulerInfo {
      name: string;
      aliases: string[];
      config: SchedulerConfig;
      createStepFn: (config: SchedulerConfig) => SchedulerStepFunction;
    }
    ```

- [ ] **1.2** Create `packages/web-txt2img/src/scheduler/registry.ts`
  - [ ] Create `SCHEDULER_REGISTRY` array with initial schedulers
    - [ ] Euler (existing, refactored)
    - [ ] DDIM (existing, refactored)
    - [ ] DPM++ 2M (existing, refactored)
    - [ ] Euler Ancestral (existing, refactored)
  - [ ] Implement `findScheduler(name: string): SchedulerInfo | null`
  - [ ] Implement `listSchedulers(): string[]`
  - [ ] Add validation for scheduler configuration

- [ ] **1.3** Update `packages/web-txt2img/src/types.ts`
  - [ ] Expand `SchedulerId` type to include new schedulers
    ```typescript
    export type SchedulerId = 
      | 'euler' 
      | 'ddim' 
      | 'dpmpp_2m_karras' 
      | 'euler_ancestral'
      | 'dpmpp_2m'
      | 'dpmpp_sde'
      | 'heun'
      | 'flow_euler'
      | 'flow_dpmpp_2m';
    ```
  - [ ] Add `schedulerConfig` optional field to `GenerateParams`

- [ ] **1.4** Refactor existing step functions in `sd-turbo.ts`
  - [ ] Extract `eulerStep` into `scheduler/steps/euler.ts`
  - [ ] Extract `ddimStep` into `scheduler/steps/ddim.ts`
  - [ ] Extract `dpmpp2mStep` into `scheduler/steps/dpmpp2m.ts`
  - [ ] Extract `eulerAncestralStep` into `scheduler/steps/eulerAncestral.ts`
  - [ ] Update imports in `sd-turbo.ts`

- [ ] **1.5** Create barrel export `packages/web-txt2img/src/scheduler/index.ts`
  - [ ] Re-export all types from `types.ts`
  - [ ] Re-export registry functions
  - [ ] Re-export step functions

---

## Phase 2: Sigma Schedule System

**Objective:** Implement configurable sigma schedules (Karras, Exponential, Beta) as documented in the tech note.

**Duration:** ~3-4 hours

### TODO List

- [ ] **2.1** Create `packages/web-txt2img/src/scheduler/sigmas.ts`
  - [ ] Implement `computeLinearSigmas(sigma: number, numSteps: number): number[]`
    - [ ] Use beta schedule: `betas = linspace(beta_start, beta_end, T)`
    - [ ] Compute alphas_cumprod
    - [ ] Derive sigmas: `sigma_t = sqrt((1 - alpha_cumprod) / alpha_cumprod)`
  - [ ] Implement `computeKarrasSigmas(sigmaMin: number, sigmaMax: number, numSteps: number, rho: number = 7.0): number[]`
    - [ ] Formula: `sigma_i = (sigma_min^(1/rho) + i/(N-1) * (sigma_max^(1/rho) - sigma_min^(1/rho)))^rho`
    - [ ] Add validation for sigma_min < sigma_max
  - [ ] Implement `computeExponentialSigmas(sigmaMin: number, sigmaMax: number, numSteps: number): number[]`
    - [ ] Formula: `sigma_i = exp(log(sigma_max) + i/(N-1) * (log(sigma_min) - log(sigma_max)))`
  - [ ] Implement `computeBetaSigmas(sigmaMin: number, sigmaMax: number, numSteps: number, alpha: number = 0.6, beta: number = 0.6): number[]`
    - [ ] Use Beta distribution PPF for non-uniform sampling
    - [ ] Note: May require a lightweight Beta PPF approximation for browser

- [ ] **2.2** Create `packages/web-txt2img/src/scheduler/schedule.ts`
  - [ ] Implement `SigmaSchedule` class
    - [ ] Constructor takes `SchedulerConfig`
    - [ ] `setTimesteps(numSteps: number): void` — computes sigma array
    - [ ] `getSigma(stepIndex: number): number` — returns sigma at step
    - [ ] `getNextSigma(stepIndex: number): number` — returns next sigma
    - [ ] `getTimesteps(): number[]` — returns timestep array
  - [ ] Support different timestep spacing modes
    - [ ] `linspace` — uniform spacing
    - [ ] `leading` — include t=0
    - [ ] `trailing` — include t=T

- [ ] **2.3** Update `sd-turbo.ts` denoising loop
  - [ ] Replace hardcoded sigma computation with `SigmaSchedule` instance
  - [ ] Pass `schedulerConfig` from `GenerateParams` if provided
  - [ ] Add progress reporting for sigma schedule selection

- [ ] **2.4** Add Karras schedule to existing DPM++ 2M scheduler
  - [ ] Update `dpmpp_2m_karras` to use proper Karras sigma computation
  - [ ] Validate against SD.Next's `_convert_to_karras()` implementation

---

## Phase 3: Core Solver Improvements

**Objective:** Fix the approximate DPM++ 2M implementation and add Heun solver.

**Duration:** ~3-4 hours

### TODO List

- [ ] **3.1** Fix DPM++ 2M implementation in `scheduler/steps/dpmpp2m.ts`
  - [ ] Replace approximate `h_prev` computation with proper log-sigma formulation
    ```typescript
    // Current (WRONG):
    const h_prev = sigma - (sigma + (prevTimestep - currentTimestep) / currentTimestep * sigma);
    
    // Correct (from SD.Next):
    const t = -Math.log(sigma);
    const t_next = -Math.log(nextSigma);
    const h = t_next - t;
    const sigma_fn = (tau: number) => Math.exp(-tau);
    ```
  - [ ] Implement proper first-step fallback (Euler)
  - [ ] Implement proper multistep formula with `denoised_d` computation
  - [ ] Add `expm1` helper for numerical stability: `Math.expm1(x) = e^x - 1`

- [ ] **3.2** Add Heun solver (2nd-order Runge-Kutta)
  - [ ] Create `scheduler/steps/heun.ts`
  - [ ] Implement predictor-corrector step
    - [ ] Predictor: Euler step to intermediate point
    - [ ] Corrector: Trapezoidal rule using model output at intermediate point
    - [ ] Note: Requires 2 UNet evaluations per step — document this tradeoff
  - [ ] Add to registry with appropriate config

- [ ] **3.3** Add DPM-Solver-2 (midpoint method)
  - [ ] Create `scheduler/steps/dpmSolver2.ts`
  - [ ] Implement two-stage midpoint method
    - [ ] Compute `sigma_mid = exp(log(sigma_t) * 0.5 + log(sigma_next) * 0.5)`
    - [ ] First derivative: `d = (x_t - x0_hat) / sigma_t`
    - [ ] Intermediate step: `x_mid = x_t + d * (sigma_mid - sigma_t)`
    - [ ] Second derivative at midpoint
    - [ ] Final step to `sigma_next`

- [ ] **3.4** Add DPM++ SDE (stochastic)
  - [ ] Create `scheduler/steps/dpmppSde.ts`
  - [ ] Implement two-phase SDE step
    - [ ] Phase 1: Euler step to intermediate point
    - [ ] Phase 2: Add controlled noise based on sigma gap
  - [ ] Use seeded noise from `mulberry32` for reproducibility

- [ ] **3.5** Update `SchedulerState` to support multistep solvers
  - [ ] Add `modelOutputs` buffer (size = solver_order)
  - [ ] Add `prevSigma` tracking
  - [ ] Add step index management
  - [ ] Implement state reset for new generation

---

## Phase 4: Flow Matching Support

**Objective:** Add support for flow matching models (SD3, FLUX-style) with proper sigma computation and timestep shifting.

**Duration:** ~4-5 hours

### TODO List

- [ ] **4.1** Create `packages/web-txt2img/src/scheduler/flow.ts`
  - [ ] Implement `computeFlowSigmas(numSteps: number, shift: number, numTrainTimesteps: number = 1000): number[]`
    - [ ] Base formula: `sigma_t = t / T`
    - [ ] Apply shifting: `sigma_shifted = (shift * sigma) / (1 + (shift - 1) * sigma)`
  - [ ] Implement `computeDynamicShift(imageSeqLen: number, baseSeqLen: number = 36864, baseShift: number = 0.5, maxShift: number = 1.15): number`
    - [ ] Formula: `shift = clamp(base_shift + (max_shift - base_shift) / base_seq_len * image_seq_len, base_shift, max_shift)`
  - [ ] Implement `timeShift(mu: number, sigma: number, t: number): number`
    - [ ] Formula: `exp(mu) / (exp(mu) + (1/t - 1)^sigma)`

- [ ] **4.2** Create flow-adapted step functions
  - [ ] Create `scheduler/steps/flowEuler.ts`
    - [ ] Same as Euler but with flow sigma computation
    - [ ] Convert model output from flow prediction to x0: `x0_hat = x_t - sigma * v_theta`
  - [ ] Create `scheduler/steps/flowDpmpp2m.ts`
    - [ ] Same as DPM++ 2M but with flow sigma computation
    - [ ] Handle flow prediction type conversion

- [ ] **4.3** Update `SigmaSchedule` class
  - [ ] Add `useFlowSigmas` flag
  - [ ] Add `shift` parameter
  - [ ] Add `useDynamicShifting` flag
  - [ ] Route to flow sigma computation when enabled

- [ ] **4.4** Add flow schedulers to registry
  - [ ] `flow_euler` — Euler with flow sigmas
  - [ ] `flow_dpmpp_2m` — DPM++ 2M with flow sigmas
  - [ ] Configure appropriate shift values for SD3 (shift=3.0) and FLUX (dynamic)

- [ ] **4.5** Update model compatibility detection
  - [ ] Add `isFlowModel(modelId: string): boolean` helper
  - [ ] Auto-select flow schedulers for flow models
  - [ ] Validate scheduler compatibility with model prediction type

---

## Phase 5: Advanced Features

**Objective:** Implement advanced noise sampling and custom schedule support.

**Duration:** ~3-4 hours

### TODO List

- [ ] **5.1** Implement Brownian Bridge noise sampling
  - [ ] Create `packages/web-txt2img/src/scheduler/noise.ts`
  - [ ] Implement `SimpleBrownianBridge` class
    - [ ] Constructor takes seed
    - [ ] `getNoise(sigma: number, sigmaNext: number): Float32Array` — returns correlated noise
    - [ ] Cache noise values for (sigma, sigma_next) pairs
    - [ ] Use Mulberry32 for seeded randomness
  - [ ] Note: Full BrownianTree is complex; simplified version sufficient for browser

- [ ] **5.2** Add custom timestep support
  - [ ] Update `GenerateParams` to accept `customTimesteps?: number[]`
  - [ ] Update `SigmaSchedule` to accept custom timesteps
  - [ ] Validate custom timesteps are within valid range
  - [ ] Convert custom timesteps to sigmas

- [ ] **5.3** Add sigma interpolation modes
  - [ ] Implement `interpolateSigmas(sigmas: number[], mode: 'linear' | 'logarithmic'): number[]`
  - [ ] Support non-uniform step distribution
  - [ ] Allow more steps at critical denoising phases

- [ ] **5.4** Add scheduler configuration presets
  - [ ] Create `packages/web-txt2img/src/scheduler/presets.ts`
  - [ ] Define presets for common use cases:
    - [ ] `fast` — Euler, 1-4 steps, linear sigmas
    - [ ] `balanced` — DPM++ 2M, 10-20 steps, Karras sigmas
    - [ ] `quality` — DPM++ 2M SDE, 20-50 steps, exponential sigmas
    - [ ] `flow_fast` — Flow Euler, 1-4 steps, flow sigmas
    - [ ] `flow_quality` — Flow DPM++ 2M, 10-20 steps, flow sigmas with shift

- [ ] **5.5** Add scheduler metadata
  - [ ] Add `description` field to `SchedulerInfo`
  - [ ] Add `recommendedSteps` range
  - [ ] Add `qualityRating` (1-5 stars)
  - [ ] Add `speedRating` (1-5 stars)

---

## Phase 6: Testing & Validation

**Objective:** Ensure all schedulers produce correct results and are numerically stable.

**Duration:** ~2-3 hours

### TODO List

- [ ] **6.1** Create unit tests for sigma schedule computations
  - [ ] Test `computeLinearSigmas` produces expected range
  - [ ] Test `computeKarrasSigmas` matches reference values
  - [ ] Test `computeExponentialSigmas` produces monotonic sequence
  - [ ] Test `computeFlowSigmas` with various shift values
  - [ ] Test edge cases: 1 step, 100 steps, extreme shift values

- [ ] **6.2** Create unit tests for step functions
  - [ ] Test Euler step produces expected output for known inputs
  - [ ] Test DDIM step preserves norm
  - [ ] Test DPM++ 2M first step falls back to Euler
  - [ ] Test DPM++ 2M multistep uses previous output correctly
  - [ ] Test Heun step produces smoother results than Euler

- [ ] **6.3** Create integration tests
  - [ ] Test full denoising loop with each scheduler
  - [ ] Verify output dimensions are preserved
  - [ ] Verify seed reproducibility
  - [ ] Test abort signal handling during denoising

- [ ] **6.4** Create numerical stability tests
  - [ ] Test with very small sigmas (near 0)
  - [ ] Test with very large sigmas (near 14.6146)
  - [ ] Test with extreme aspect ratios
  - [ ] Test with 1-step generation (SD-Turbo mode)

- [ ] **6.5** Performance benchmarks
  - [ ] Measure step function execution time
  - [ ] Compare scheduler overhead (should be < 5% of total time)
  - [ ] Profile memory usage for multistep buffers

---

## File Structure After Implementation

```
packages/web-txt2img/src/
├── scheduler/
│   ├── index.ts              # Barrel exports
│   ├── types.ts              # SchedulerConfig, SchedulerState, etc.
│   ├── registry.ts           # SCHEDULER_REGISTRY, findScheduler()
│   ├── sigmas.ts             # Sigma schedule computations
│   ├── schedule.ts           # SigmaSchedule class
│   ├── flow.ts               # Flow matching utilities
│   ├── noise.ts              # Brownian bridge noise
│   ├── presets.ts            # Configuration presets
│   └── steps/
│       ├── euler.ts          # Euler step
│       ├── eulerAncestral.ts # Euler ancestral step
│       ├── ddim.ts           # DDIM step
│       ├── dpmpp2m.ts        # DPM++ 2M step (fixed)
│       ├── dpmppSde.ts       # DPM++ SDE step
│       ├── dpmSolver2.ts     # DPM-Solver-2 step
│       ├── heun.ts           # Heun step
│       ├── flowEuler.ts      # Flow Euler step
│       └── flowDpmpp2m.ts    # Flow DPM++ 2M step
├── adapters/
│   └── sd-turbo.ts          # Updated to use new scheduler system
└── types.ts                 # Updated SchedulerId type
```

---

## Dependencies & References

| Phase | SD.Next Reference | Notes |
|-------|------------------|-------|
| 1 | `sd_samplers_diffusers.py` lines 1-100 | Config patterns |
| 1 | `sd_samplers_common.py` lines 1-30 | SamplerData namedtuple |
| 2 | `scheduler_dpm_flowmatch.py` lines 200-400 | Sigma schedule conversions |
| 3 | `scheduler_dpm_flowmatch.py` lines 400-700 | Step function implementations |
| 4 | `scheduler_dpm_flowmatch.py` lines 100-200 | Flow sigma computation |
| 4 | `sd_hijack_schedulers.py` lines 50-150 | Custom schedule support |
| 5 | `scheduler_dpm_flowmatch.py` lines 1-100 | BrownianTree noise |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Browser precision issues with expm1 | Medium | High | Use `Math.expm1()` which is available in all modern browsers |
| Multistep buffer memory overhead | Low | Medium | Limit buffer size to solver_order (typically 2-3) |
| Flow model compatibility | Medium | High | Add model detection and fallback to default scheduler |
| Performance regression | Low | Medium | Benchmark each phase; scheduler overhead should be minimal |
| Beta PPF not available in browser | High | Low | Use approximation or skip Beta schedule initially |

---

## Success Criteria

- [ ] All existing schedulers (Euler, DDIM, DPM++ 2M, Euler Ancestral) continue to work
- [ ] Karras sigma schedule produces visibly improved results
- [ ] DPM++ 2M produces identical results to SD.Next (within floating point tolerance)
- [ ] Flow matching schedulers work with SD3-style models
- [ ] Scheduler selection is type-safe and validated
- [ ] No performance regression (> 5% overhead is unacceptable)
- [ ] All schedulers support seeded reproducibility

---

*Plan created: 2026-07-06*
*Based on: SD.Next scheduler analysis, TECHNOTE_InferenceScheduling.md*
