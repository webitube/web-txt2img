/**
 * Scheduler types and interfaces for the inference scheduler system.
 * Mirrors SD.Next's SamplerData pattern for type-safe, extensible scheduler management.
 */

/**
 * Configuration for a scheduler instance.
 * All parameters are optional with sensible defaults.
 */
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

/**
 * State carried between scheduler steps for multistep solvers.
 */
export interface SchedulerState {
  stepIndex: number;
  prevModelOutput: Float32Array | null;
  prevSigma: number | null;
  modelOutputs: Float32Array[]; // buffer for multistep solvers
}

/**
 * Signature of a scheduler step function.
 * Takes model output, current sample, and sigma values, returns next sample.
 */
export type SchedulerStepFunction = (
  modelOutput: Float32Array,
  sample: Float32Array,
  sigma: number,
  nextSigma: number,
  state: SchedulerState
) => Float32Array;

/**
 * Scheduler information - mirrors SD.Next's SamplerData namedtuple.
 * Each scheduler is registered with a name, aliases, config, and step function factory.
 */
export interface SchedulerInfo {
  name: string;
  aliases: string[];
  config: SchedulerConfig;
  createStepFn: (config: SchedulerConfig) => SchedulerStepFunction;
  description?: string;
  recommendedSteps?: [min: number, max: number];
  qualityRating?: number; // 1-5 stars
  speedRating?: number; // 1-5 stars
}
