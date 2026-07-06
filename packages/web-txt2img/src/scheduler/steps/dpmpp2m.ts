/**
 * DPM++ 2M (2nd-order multistep) step.
 * Uses the current and previous model outputs for a higher-order integration.
 * First step falls back to Euler (no previous output available).
 *
 * Correct formula (from SD.Next / DPM-Solver++ paper):
 *   t = -log(sigma), t_next = -log(sigma_next)
 *   h = t_next - t
 *   sigma_fn(tau) = exp(-tau)
 *   denoised_d = (denoised_d - prev_denoised_d) / h_prev
 *   x_{t-1} = exp(-h) * x_t + sigma_fn(t_next) * (expm1(h) * D(x_t, t) + 0.5 * h * denoised_d)
 *
 * This is the proper log-sigma formulation used by SD.Next.
 */
import type { SchedulerConfig, SchedulerState, SchedulerStepFunction } from '../types';

export function createDpmpp2mStepFn(_config: SchedulerConfig): SchedulerStepFunction {
  return function dpmpp2mStep(
    modelOutput: Float32Array,
    sample: Float32Array,
    sigma: number,
    nextSigma: number,
    state: SchedulerState,
  ) {
    const d_o = new Float32Array(sample.length);

    // First step: fall back to Euler
    if (state.stepIndex === 0 || !state.prevModelOutput) {
      for (let i = 0; i < sample.length; i++) {
        const epsilon = modelOutput[i];
        const dt = nextSigma - sigma;
        d_o[i] = sample[i] + epsilon * dt;
      }
      return d_o;
    }

    // Proper log-sigma formulation (from SD.Next)
    const t = -Math.log(sigma);
    const t_next = -Math.log(nextSigma);
    const h = t_next - t;
    const sigma_fn = (tau: number) => Math.exp(-tau);

    // Previous step size
    const prevT = state.prevSigma ? -Math.log(state.prevSigma) : t;
    const h_prev = t - prevT;

    for (let i = 0; i < sample.length; i++) {
      // modelOutput is the denoised prediction (x0_hat)
      const denoised_d = modelOutput[i];
      const prev_denoised_d = state.prevModelOutput![i];

      // Compute derivative of denoised direction
      const denoised_d_d = h_prev !== 0 ? (denoised_d - prev_denoised_d) / h_prev : 0;

      // DPM++ 2M formula
      d_o[i] = Math.exp(-h) * sample[i] + sigma_fn(t_next) * (Math.expm1(h) * denoised_d + 0.5 * h * denoised_d_d);
    }
    return d_o;
  };
}
