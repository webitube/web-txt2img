/**
 * Flow matching utilities for SD3, FLUX-style models.
 * Implements flow sigma computation and timestep shifting.
 */

/**
 * Compute flow sigmas for flow matching models.
 *
 * @param numSteps - Number of denoising steps
 * @param shift - Shift parameter for timestep shifting
 * @param numTrainTimesteps - Number of training timesteps (default: 1000)
 * @returns Array of sigma values (descending)
 */
export function computeFlowSigmas(
  numSteps: number,
  shift: number = 1.0,
  numTrainTimesteps: number = 1000,
): number[] {
  const sigmas = new Array(numSteps);

  for (let i = 0; i < numSteps; i++) {
    // Base formula: sigma_t = t / T
    const t = i / numSteps;
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
 * Compute dynamic shift based on image sequence length.
 * Used for FLUX-style models with adaptive shifting.
 *
 * @param imageSeqLen - Image sequence length
 * @param baseSeqLen - Base sequence length (default: 36864)
 * @param baseShift - Base shift value (default: 0.5)
 * @param maxShift - Maximum shift value (default: 1.15)
 * @returns Computed shift value
 */
export function computeDynamicShift(
  imageSeqLen: number,
  baseSeqLen: number = 36864,
  baseShift: number = 0.5,
  maxShift: number = 1.15,
): number {
  // Formula: shift = clamp(base_shift + (max_shift - base_shift) / base_seq_len * image_seq_len, base_shift, max_shift)
  const shift = baseShift + (maxShift - baseShift) / baseSeqLen * imageSeqLen;
  return Math.max(baseShift, Math.min(maxShift, shift));
}

/**
 * Time shift function for flow matching.
 *
 * @param mu - Mu parameter
 * @param sigma - Sigma parameter
 * @param t - Time value
 * @returns Shifted time value
 */
export function timeShift(mu: number, sigma: number, t: number): number {
  // Formula: exp(mu) / (exp(mu) + (1/t - 1)^sigma)
  const expMu = Math.exp(mu);
  const invT = 1 / t - 1;
  return expMu / (expMu + Math.pow(invT, sigma));
}

/**
 * Check if a model is a flow matching model.
 *
 * @param modelId - Model identifier
 * @returns True if the model uses flow matching
 */
export function isFlowModel(modelId: string): boolean {
  const flowModels = ['sd3', 'sd3-medium', 'sd3-large', 'flux', 'flux-schnell', 'flux-dev'];
  const normalized = modelId.toLowerCase().replace(/[\s_-]+/g, '');
  return flowModels.some((fm) => normalized.includes(fm.replace(/[\s_-]+/g, '')));
}
