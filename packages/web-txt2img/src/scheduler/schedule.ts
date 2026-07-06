/**
 * SigmaSchedule class - manages sigma computation and timestep management.
 * Supports different sigma schedules (linear, Karras, exponential, flow) and timestep spacing modes.
 */
import type { SchedulerConfig } from './types';
import { computeLinearSigmas, computeKarrasSigmas, computeExponentialSigmas } from './sigmas';

/**
 * SigmaSchedule manages the sigma array and provides access to sigma values at each step.
 * Supports multiple sigma schedules and timestep spacing modes.
 */
export class SigmaSchedule {
  private sigmas: number[] = [];
  private timesteps: number[] = [];
  private config: SchedulerConfig;
  private numSteps: number = 0;

  constructor(config: SchedulerConfig = {}) {
    this.config = config;
  }

  /**
   * Set the number of timesteps and compute the sigma array.
   * This must be called before accessing sigma values.
   *
   * @param numSteps - Number of denoising steps
   * @param customTimesteps - Optional custom timestep array
   */
  setTimesteps(numSteps: number, customTimesteps?: number[]): void {
    this.numSteps = numSteps;

    if (customTimesteps && customTimesteps.length > 0) {
      this.timesteps = customTimesteps;
      this.sigmas = this.timesteps.map((t) => this.timestepToSigma(t));
    } else {
      // Build timesteps based on spacing mode
      this.timesteps = this.buildTimesteps();
      // Build sigmas based on schedule type
      this.sigmas = this.buildSigmas();
    }
  }

  /**
   * Get sigma value at a given step index.
   *
   * @param stepIndex - Index of the step (0 to numSteps-1)
   * @returns Sigma value at that step
   */
  getSigma(stepIndex: number): number {
    if (stepIndex < 0 || stepIndex >= this.sigmas.length) {
      throw new Error(`Step index ${stepIndex} out of range [0, ${this.sigmas.length - 1}]`);
    }
    return this.sigmas[stepIndex];
  }

  /**
   * Get the next sigma value (for stepping toward).
   *
   * @param stepIndex - Current step index
   * @returns Next sigma value (or 0 if at last step)
   */
  getNextSigma(stepIndex: number): number {
    if (stepIndex < 0 || stepIndex >= this.sigmas.length) {
      throw new Error(`Step index ${stepIndex} out of range [0, ${this.sigmas.length - 1}]`);
    }
    if (stepIndex === this.sigmas.length - 1) {
      // Final step - return 0 or sigma_min based on config
      return this.config.finalSigmasType === 'sigma_min' ? this.sigmas[this.sigmas.length - 1] : 0;
    }
    return this.sigmas[stepIndex + 1];
  }

  /**
   * Get all timesteps.
   *
   * @returns Array of timestep values
   */
  getTimesteps(): number[] {
    return [...this.timesteps];
  }

  /**
   * Get all sigmas.
   *
   * @returns Array of sigma values
   */
  getSigmas(): number[] {
    return [...this.sigmas];
  }

  /**
   * Get the number of steps.
   */
  getNumSteps(): number {
    return this.numSteps;
  }

  /**
   * Build timesteps based on spacing mode.
   */
  private buildTimesteps(): number[] {
    const numTrainTimesteps = this.config.numTrainTimesteps ?? 1000;
    const spacing = this.config.timestepSpacing ?? 'linspace';

    switch (spacing) {
      case 'leading':
        // Include t=0
        return Array.from({ length: this.numSteps }, (_, i) =>
          Math.round((i * numTrainTimesteps) / (this.numSteps - 1)),
        );
      case 'trailing':
        // Include t=T
        return Array.from({ length: this.numSteps }, (_, i) =>
          Math.round(((this.numSteps - 1 - i) * numTrainTimesteps) / (this.numSteps - 1)),
        );
      case 'linspace':
      default:
        // Uniform spacing from numTrainTimesteps-1 down to 0
        return Array.from({ length: this.numSteps }, (_, i) =>
          Math.round(numTrainTimesteps * (1 - i / this.numSteps)),
        );
    }
  }

  /**
   * Build sigmas based on schedule type.
   */
  private buildSigmas(): number[] {
    const sigmaSchedule = this.config.sigmaSchedule;
    const numTrainTimesteps = this.config.numTrainTimesteps ?? 1000;

    // Flow sigmas (for flow matching models)
    if (this.config.useFlowSigmas) {
      return this.buildFlowSigmas();
    }

    // Get base sigmas from beta schedule
    const baseSigmas = computeLinearSigmas(
      this.config.betaStart ?? 0.00085,
      this.config.betaEnd ?? 0.012,
      numTrainTimesteps,
      this.config.betaSchedule ?? 'linear',
    );

    // Sample sigmas at timestep positions
    const sampledSigmas = this.timesteps.map((t) => {
      const idx = Math.min(t, baseSigmas.length - 1);
      return baseSigmas[idx];
    });

    // Apply sigma schedule transformation if specified
    if (sigmaSchedule === 'karras' || this.config.useKarrasSigmas) {
      return this.convertToKarras(sampledSigmas);
    }

    if (sigmaSchedule === 'exponential' || this.config.useExponentialSigmas) {
      return this.convertToExponential(sampledSigmas);
    }

    return sampledSigmas;
  }

  /**
   * Build flow matching sigmas.
   */
  private buildFlowSigmas(): number[] {
    const shift = this.config.shift ?? 1.0;
    const numTrainTimesteps = this.config.numTrainTimesteps ?? 1000;

    const sigmas = new Array(this.numSteps);
    for (let i = 0; i < this.numSteps; i++) {
      // Base formula: sigma_t = t / T
      const t = i / this.numSteps;
      let sigma = t * numTrainTimesteps;

      // Apply shifting: sigma_shifted = (shift * sigma) / (1 + (shift - 1) * sigma)
      if (shift !== 1.0) {
        sigma = (shift * sigma) / (1 + (shift - 1) * sigma);
      }

      sigmas[i] = sigma;
    }

    // Flow sigmas go from high to low (noise to clean)
    return sigmas.reverse();
  }

  /**
   * Convert sigmas to Karras schedule.
   * Based on SD.Next's _convert_to_karras() implementation.
   */
  private convertToKarras(sigmas: number[]): number[] {
    const sigmaMin = sigmas[sigmas.length - 1];
    const sigmaMax = sigmas[0];
    const rho = 7.0;

    return computeKarrasSigmas(sigmaMin, sigmaMax, sigmas.length, rho);
  }

  /**
   * Convert sigmas to exponential schedule.
   */
  private convertToExponential(sigmas: number[]): number[] {
    const sigmaMin = sigmas[sigmas.length - 1];
    const sigmaMax = sigmas[0];

    return computeExponentialSigmas(sigmaMin, sigmaMax, sigmas.length);
  }

  /**
   * Convert timestep to sigma value.
   * Used for custom timestep support.
   */
  private timestepToSigma(t: number): number {
    const numTrainTimesteps = this.config.numTrainTimesteps ?? 1000;
    const betaStart = this.config.betaStart ?? 0.00085;
    const betaEnd = this.config.betaEnd ?? 0.012;

    // Compute alpha_cumprod at timestep t
    const betas = computeLinearSigmas(betaStart, betaEnd, numTrainTimesteps + 1, this.config.betaSchedule ?? 'linear');
    const idx = Math.min(t, betas.length - 1);
    return betas[idx];
  }
}
