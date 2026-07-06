/**
 * DPM++ SDE (stochastic) step.
 * Two-phase SDE step with controlled noise injection.
 *
 * Formula:
 *   Phase 1: Euler step to intermediate point
 *   Phase 2: Add controlled noise based on sigma gap
 *
 * Uses seeded noise from mulberry32 for reproducibility.
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

export function createDpmppSdeStepFn(config: SchedulerConfig): SchedulerStepFunction {
  const sNoise = config.sNoise ?? 0.1;

  return function dpmppSdeStep(
    modelOutput: Float32Array,
    sample: Float32Array,
    sigma: number,
    nextSigma: number,
    state: SchedulerState,
  ) {
    const d_o = new Float32Array(sample.length);

    // Deterministic noise for reproducibility
    const seed = (state.stepIndex >>> 0) || 0;
    const rand = mulberry32(seed + 12345); // Offset seed for SDE noise

    // Phase 1: Euler step to intermediate point
    const lambda = -Math.log(sigma) + Math.log(nextSigma);
    const dt = nextSigma - sigma;

    for (let i = 0; i < sample.length; i++) {
      const epsilon = modelOutput[i];
      // Euler step
      let val = sample[i] + epsilon * dt;

      // Phase 2: Add controlled noise based on sigma gap
      const noiseScale = Math.sqrt(Math.max(sigma * sigma - nextSigma * nextSigma, 0)) * sNoise;
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
