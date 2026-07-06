/**
 * Barrel exports for the scheduler module.
 * Import from 'scheduler' to get all types, registry, and step functions.
 */

// Types
export type { SchedulerConfig, SchedulerState, SchedulerStepFunction, SchedulerInfo } from './types';

// Registry
export { SCHEDULER_REGISTRY, findScheduler, listSchedulers, validateSchedulerConfig } from './registry';

// Sigma schedule
export { SigmaSchedule } from './schedule';
export { computeLinearSigmas, computeKarrasSigmas, computeExponentialSigmas } from './sigmas';

// Flow matching
export { computeFlowSigmas, computeDynamicShift, timeShift, isFlowModel } from './flow';

// Noise
export { SimpleBrownianBridge } from './noise';

// Presets
export { SCHEDULER_PRESETS, getPreset, listPresets } from './presets';

// Step functions
export { createEulerStepFn } from './steps/euler';
export { createDdimStepFn } from './steps/ddim';
export { createDpmpp2mStepFn } from './steps/dpmpp2m';
export { createEulerAncestralStepFn } from './steps/eulerAncestral';
export { createHeunStepFn } from './steps/heun';
export { createDpmSolver2StepFn } from './steps/dpmSolver2';
export { createDpmppSdeStepFn } from './steps/dpmppSde';
export { createFlowEulerStepFn } from './steps/flowEuler';
export { createFlowDpmpp2mStepFn } from './steps/flowDpmpp2m';

/**
 * Create a fresh SchedulerState for a new generation run.
 */
export function createSchedulerState(solverOrder: number = 2): import('./types').SchedulerState {
  return {
    stepIndex: 0,
    prevModelOutput: null,
    prevSigma: null,
    modelOutputs: new Array(solverOrder),
  };
}
