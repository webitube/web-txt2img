/**
 * Flow Euler step - Euler with flow sigma computation.
 * Same as Euler but with flow sigma computation.
 *
 * Convert model output from flow prediction to x0: x0_hat = x_t - sigma * v_theta
 */
import type { SchedulerConfig, SchedulerState, SchedulerStepFunction } from '../types';

export function createFlowEulerStepFn(_config: SchedulerConfig): SchedulerStepFunction {
  return function flowEulerStep(
    modelOutput: Float32Array,
    sample: Float32Array,
    sigma: number,
    nextSigma: number,
    _state: SchedulerState,
  ) {
    const d_o = new Float32Array(sample.length);

    for (let i = 0; i < sample.length; i++) {
      // modelOutput is v_theta (flow prediction)
      // Convert to x0: x0_hat = x_t - sigma * v_theta
      const vTheta = modelOutput[i];
      const x0Hat = sample[i] - sigma * vTheta;
      // Step toward next sigma
      const dt = nextSigma - sigma;
      d_o[i] = sample[i] + vTheta * dt;
    }
    return d_o;
  };
}
