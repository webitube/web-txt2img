/**
 * Euler Ancestral step: Euler + added noise for diversity.
 * The noise injection makes outputs more varied but less deterministic.
 *
 * Formula: x_{t-1} = x_t + epsilon * (sigma_{t-1} - sigma_t) + noise * sqrt(sigma_{t-1}^2 - sigma_t^2)
 */
import type { SchedulerConfig, SchedulerState, SchedulerStepFunction } from '../types';

/** Mulberry32 PRNG for seeded reproducibility */
function mulberry32(seed: number) {
  let t = seed >>> 0;
  return function () {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export function createEulerAncestralStepFn(_config: SchedulerConfig): SchedulerStepFunction {
  return function eulerAncestralStep(
    modelOutput: Float32Array,
    sample: Float32Array,
    sigma: number,
    nextSigma: number,
    state: SchedulerState,
  ) {
    const d_o = new Float32Array(sample.length);

    // Deterministic noise for reproducibility - use step index for unique seed per step
    const seed = (state.stepIndex >>> 0) || 0;
    const rand = mulberry32(seed);

    // Ancestral noise scale: sqrt(sigma_{t-1}^2 - sigma_t^2)
    // Note: sigma decreases during denoising, so nextSigma < sigma
    // We need sqrt(sigma^2 - nextSigma^2) for the noise scale
    const noiseScale = Math.sqrt(Math.max(sigma * sigma - nextSigma * nextSigma, 0));

    for (let i = 0; i < sample.length; i++) {
      const epsilon = modelOutput[i];
      // Euler step
      const dt = nextSigma - sigma;
      let val = sample[i] + epsilon * dt;

      // Add ancestral noise
      if (noiseScale > 0) {
        const u = Math.max(rand(), 1e-10);
        const v = rand();
        const noise = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
        val += noise * noiseScale;
      }

      d_o[i] = val;
    }
    return d_o;
  };
}
