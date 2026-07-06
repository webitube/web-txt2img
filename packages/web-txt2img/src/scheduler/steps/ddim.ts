/**
 * DDIM step: deterministic denoising.
 *
 * Formula (eta=0, deterministic):
 *   pred_original = x_t - sigma * epsilon
 *   x_{t-1} = (nextSigma/sigma) * x_t + (1 - nextSigma/sigma) * pred_original
 */
import type { SchedulerConfig, SchedulerState, SchedulerStepFunction } from '../types';

export function createDdimStepFn(_config: SchedulerConfig): SchedulerStepFunction {
  return function ddimStep(
    modelOutput: Float32Array,
    sample: Float32Array,
    sigma: number,
    nextSigma: number,
    _state: SchedulerState,
  ) {
    const d_o = new Float32Array(sample.length);
    for (let i = 0; i < sample.length; i++) {
      const epsilon = modelOutput[i];
      const predOriginal = sample[i] - sigma * epsilon;
      const ratio = nextSigma / sigma;
      d_o[i] = ratio * sample[i] + (1 - ratio) * predOriginal;
    }
    return d_o;
  };
}
