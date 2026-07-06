/**
 * Flow DPM++ 2M step - DPM++ 2M with flow sigma computation.
 * Same as DPM++ 2M but with flow sigma computation.
 *
 * Handle flow prediction type conversion.
 */
import type { SchedulerConfig, SchedulerState, SchedulerStepFunction } from '../types';

export function createFlowDpmpp2mStepFn(_config: SchedulerConfig): SchedulerStepFunction {
  return function flowDpmpp2mStep(
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
        const vTheta = modelOutput[i];
        const dt = nextSigma - sigma;
        d_o[i] = sample[i] + vTheta * dt;
      }
      return d_o;
    }

    // Proper log-sigma formulation for flow matching
    const t = -Math.log(sigma);
    const t_next = -Math.log(nextSigma);
    const h = t_next - t;
    const sigma_fn = (tau: number) => Math.exp(-tau);

    // Previous step size
    const prevT = state.prevSigma ? -Math.log(state.prevSigma) : t;
    const h_prev = t - prevT;

    for (let i = 0; i < sample.length; i++) {
      // modelOutput is v_theta (flow prediction)
      const vTheta = modelOutput[i];
      const prevVTheta = state.prevModelOutput![i];

      // Compute derivative of flow direction
      const vTheta_d = h_prev !== 0 ? (vTheta - prevVTheta) / h_prev : 0;

      // DPM++ 2M formula for flow matching
      d_o[i] = Math.exp(-h) * sample[i] + sigma_fn(t_next) * (Math.expm1(h) * vTheta + 0.5 * h * vTheta_d);
    }
    return d_o;
  };
}
