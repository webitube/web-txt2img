/**
 * Scheduler configuration presets for common use cases.
 */
import type { SchedulerConfig } from './types';

/**
 * Preset configurations for different quality/speed tradeoffs.
 */
export const SCHEDULER_PRESETS: Record<string, SchedulerConfig> = {
  /** Fast preset - Euler, 1-4 steps, linear sigmas */
  fast: {
    sigmaSchedule: null,
    solverOrder: 1,
    lowerOrderFinal: true,
    predictionType: 'epsilon',
  },

  /** Balanced preset - DPM++ 2M, 10-20 steps, Karras sigmas */
  balanced: {
    sigmaSchedule: 'karras',
    solverOrder: 2,
    lowerOrderFinal: true,
    predictionType: 'epsilon',
  },

  /** Quality preset - DPM++ 2M SDE, 20-50 steps, exponential sigmas */
  quality: {
    sigmaSchedule: 'exponential',
    solverOrder: 2,
    sNoise: 0.1,
    lowerOrderFinal: true,
    predictionType: 'epsilon',
  },

  /** Flow fast preset - Flow Euler, 1-4 steps, flow sigmas */
  flow_fast: {
    useFlowSigmas: true,
    shift: 3.0,
    solverOrder: 1,
    lowerOrderFinal: true,
    predictionType: 'flow_prediction',
  },

  /** Flow quality preset - Flow DPM++ 2M, 10-20 steps, flow sigmas with shift */
  flow_quality: {
    useFlowSigmas: true,
    shift: 3.0,
    solverOrder: 2,
    lowerOrderFinal: true,
    predictionType: 'flow_prediction',
  },
};

/**
 * Get a preset configuration by name.
 *
 * @param presetName - Name of the preset
 * @returns Preset configuration or undefined if not found
 */
export function getPreset(presetName: string): SchedulerConfig | undefined {
  return SCHEDULER_PRESETS[presetName];
}

/**
 * List all available preset names.
 *
 * @returns Array of preset names
 */
export function listPresets(): string[] {
  return Object.keys(SCHEDULER_PRESETS);
}
