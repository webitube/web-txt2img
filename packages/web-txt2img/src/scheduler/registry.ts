/**
 * Scheduler registry - provides type-safe lookup and listing of available schedulers.
 * Mirrors SD.Next's scheduler registration pattern.
 */
import type { SchedulerConfig, SchedulerInfo } from './types';
import { createEulerStepFn } from './steps/euler';
import { createDdimStepFn } from './steps/ddim';
import { createDpmpp2mStepFn } from './steps/dpmpp2m';
import { createEulerAncestralStepFn } from './steps/eulerAncestral';
import { createHeunStepFn } from './steps/heun';
import { createDpmSolver2StepFn } from './steps/dpmSolver2';
import { createDpmppSdeStepFn } from './steps/dpmppSde';
import { createFlowEulerStepFn } from './steps/flowEuler';
import { createFlowDpmpp2mStepFn } from './steps/flowDpmpp2m';

/** Default scheduler configuration */
const DEFAULT_CONFIG: SchedulerConfig = {
  betaStart: 0.00085,
  betaEnd: 0.012,
  betaSchedule: 'scaled_linear',
  numTrainTimesteps: 1000,
  sigmaSchedule: null,
  solverOrder: 1,
  lowerOrderFinal: true,
  predictionType: 'epsilon',
  timestepSpacing: 'linspace',
  finalSigmasType: 'zero',
};

/** Default flow scheduler configuration */
const DEFAULT_FLOW_CONFIG: SchedulerConfig = {
  ...DEFAULT_CONFIG,
  useFlowSigmas: true,
  shift: 3.0, // SD3 default
  predictionType: 'flow_prediction',
};

/**
 * The scheduler registry - array of all available schedulers.
 * Add new schedulers here to make them available.
 */
export const SCHEDULER_REGISTRY: SchedulerInfo[] = [
  {
    name: 'euler',
    aliases: ['euler-ddim', 'Euler'],
    config: { ...DEFAULT_CONFIG, solverOrder: 1 },
    createStepFn: createEulerStepFn,
    description: 'First-order Euler method - fast and simple',
    recommendedSteps: [1, 50],
    qualityRating: 3,
    speedRating: 5,
  },
  {
    name: 'ddim',
    aliases: ['DDIM', 'ddim-dim'],
    config: { ...DEFAULT_CONFIG, solverOrder: 1 },
    createStepFn: createDdimStepFn,
    description: 'Denoising Diffusion Implicit Models - deterministic sampling',
    recommendedSteps: [10, 100],
    qualityRating: 3,
    speedRating: 4,
  },
  {
    name: 'dpmpp_2m_karras',
    aliases: ['dpmpp_2m', 'DPM++ 2M', 'dpm++2m'],
    config: { ...DEFAULT_CONFIG, sigmaSchedule: 'karras', solverOrder: 2 },
    createStepFn: createDpmpp2mStepFn,
    description: 'DPM-Solver++ 2M with Karras sigmas - good quality/speed balance',
    recommendedSteps: [15, 50],
    qualityRating: 4,
    speedRating: 4,
  },
  {
    name: 'euler_ancestral',
    aliases: ['euler_a', 'Euler Ancestral', 'euler-ancestral'],
    config: { ...DEFAULT_CONFIG, solverOrder: 1, sNoise: 0.1 },
    createStepFn: createEulerAncestralStepFn,
    description: 'Euler with ancestral noise sampling - more diverse outputs',
    recommendedSteps: [10, 100],
    qualityRating: 3,
    speedRating: 4,
  },
  {
    name: 'dpmpp_2m',
    aliases: ['DPM++ 2M', 'dpm++2m', 'dpmpp2m'],
    config: { ...DEFAULT_CONFIG, solverOrder: 2 },
    createStepFn: createDpmpp2mStepFn,
    description: 'DPM-Solver++ 2M - 2nd-order multistep solver',
    recommendedSteps: [15, 50],
    qualityRating: 4,
    speedRating: 4,
  },
  {
    name: 'dpmpp_sde',
    aliases: ['DPM++ SDE', 'dpm++sde'],
    config: { ...DEFAULT_CONFIG, solverOrder: 2, sNoise: 0.1 },
    createStepFn: createDpmppSdeStepFn,
    description: 'DPM-Solver++ SDE - stochastic solver with noise injection',
    recommendedSteps: [20, 50],
    qualityRating: 4,
    speedRating: 3,
  },
  {
    name: 'heun',
    aliases: ['Heun', 'heun2'],
    config: { ...DEFAULT_CONFIG, solverOrder: 2 },
    createStepFn: createHeunStepFn,
    description: 'Heun solver - 2nd-order Runge-Kutta method',
    recommendedSteps: [10, 50],
    qualityRating: 4,
    speedRating: 3,
  },
  {
    name: 'flow_euler',
    aliases: ['flow-euler', 'Flow Euler'],
    config: { ...DEFAULT_FLOW_CONFIG, solverOrder: 1 },
    createStepFn: createFlowEulerStepFn,
    description: 'Euler with flow matching sigmas - for SD3/FLUX models',
    recommendedSteps: [1, 20],
    qualityRating: 3,
    speedRating: 5,
  },
  {
    name: 'flow_dpmpp_2m',
    aliases: ['flow-dpmpp-2m', 'Flow DPM++ 2M'],
    config: { ...DEFAULT_FLOW_CONFIG, solverOrder: 2 },
    createStepFn: createFlowDpmpp2mStepFn,
    description: 'DPM++ 2M with flow matching sigmas - for SD3/FLUX models',
    recommendedSteps: [10, 50],
    qualityRating: 5,
    speedRating: 4,
  },
];

/**
 * Find a scheduler by name or alias.
 * Returns null if not found.
 */
export function findScheduler(name: string): SchedulerInfo | null {
  const normalized = name.toLowerCase().replace(/[\s_-]+/g, '');
  return SCHEDULER_REGISTRY.find(
    (s) =>
      s.name.replace(/[\s_-]+/g, '') === normalized ||
      s.aliases.some((a) => a.replace(/[\s_-]+/g, '') === normalized),
  ) ?? null;
}

/**
 * List all available scheduler names.
 */
export function listSchedulers(): string[] {
  return SCHEDULER_REGISTRY.map((s) => s.name);
}

/**
 * Validate scheduler configuration.
 * Returns true if valid, false otherwise.
 */
export function validateSchedulerConfig(config: SchedulerConfig): boolean {
  // Beta schedule validation
  if (config.betaStart !== undefined && config.betaEnd !== undefined) {
    if (config.betaStart <= 0 || config.betaEnd <= 0 || config.betaStart >= config.betaEnd) {
      return false;
    }
  }

  // Solver order validation
  if (config.solverOrder !== undefined && (config.solverOrder < 1 || config.solverOrder > 4)) {
    return false;
  }

  // Flow matching validation
  if (config.useFlowSigmas && config.shift !== undefined) {
    if (config.shift <= 0) {
      return false;
    }
  }

  return true;
}
