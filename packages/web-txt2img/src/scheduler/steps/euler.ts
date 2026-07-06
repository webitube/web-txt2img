/**
 * Euler step: first-order denoising from current sigma toward next sigma.
 *
 * Formula: x_{t-1} = x_t + D(x_t, t) * (sigma_{t-1} - sigma_t)
 * where D(x_t, t) = epsilon (the noise prediction from UNet)
 */
import type { SchedulerConfig, SchedulerState, SchedulerStepFunction } from '../types';

export function createEulerStepFn(_config: SchedulerConfig): SchedulerStepFunction {
  return function eulerStep(
    modelOutput: Float32Array,
    sample: Float32Array,
    sigma: number,
    nextSigma: number,
    _state: SchedulerState,
  ) {
    const d_o = new Float32Array(sample.length);
    for (let i = 0; i < sample.length; i++) {
      const epsilon = modelOutput[i];
      const dt = nextSigma - sigma;
      d_o[i] = sample[i] + epsilon * dt;
    }
    return d_o;
  };
}
