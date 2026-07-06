/**
 * Sigma schedule computations for various scheduling strategies.
 * Implements Karras, Exponential, Beta, and linear sigma schedules.
 */

/**
 * Compute linear sigmas from beta schedule.
 * Uses beta schedule: betas = linspace(beta_start, beta_end, T)
 * Computes alphas_cumprod then derives sigmas.
 *
 * @param betaStart - Start of beta schedule (default: 0.00085)
 * @param betaEnd - End of beta schedule (default: 0.012)
 * @param numSteps - Number of timesteps (default: 1000)
 * @param betaSchedule - Beta schedule type (default: 'linear')
 * @returns Array of sigma values (descending)
 */
export function computeLinearSigmas(
  betaStart: number = 0.00085,
  betaEnd: number = 0.012,
  numSteps: number = 1000,
  betaSchedule: 'linear' | 'scaled_linear' = 'linear',
): number[] {
  let betas: number[];

  if (betaSchedule === 'scaled_linear') {
    // Scaled linear: betas = linspace(sqrt(beta_start), sqrt(beta_end), T) ** 2
    // This is the formula used by SD 2.x / SD-Turbo
    const sqrtBetas = linspace(Math.sqrt(betaStart), Math.sqrt(betaEnd), numSteps);
    betas = sqrtBetas.map((b) => b * b);
  } else {
    betas = linspace(betaStart, betaEnd, numSteps);
  }

  // Compute alphas_cumprod = cumprod(1 - betas)
  const alphasCumprod: number[] = [];
  let acc = 1;
  for (const b of betas) {
    acc *= 1 - b;
    alphasCumprod.push(acc);
  }

  // sigma_t = sqrt((1 - alpha_cumprod) / alpha_cumprod)
  const sigmas = alphasCumprod.map((alpha) => Math.sqrt((1 - alpha) / alpha));
  return sigmas;
}

/**
 * Compute Karras sigmas - non-uniform schedule that concentrates steps at lower noise levels.
 * Formula: sigma_i = (sigma_min^(1/rho) + i/(N-1) * (sigma_max^(1/rho) - sigma_min^(1/rho)))^rho
 *
 * @param sigmaMin - Minimum sigma value
 * @param sigmaMax - Maximum sigma value
 * @param numSteps - Number of steps
 * @param rho - Shape parameter (default: 7.0)
 * @returns Array of sigma values (descending)
 */
export function computeKarrasSigmas(
  sigmaMin: number,
  sigmaMax: number,
  numSteps: number,
  rho: number = 7.0,
): number[] {
  if (sigmaMin >= sigmaMax) {
    throw new Error('sigmaMin must be less than sigmaMax for Karras schedule');
  }

  const sigmas = new Array(numSteps);
  const minInvRho = Math.pow(sigmaMin, 1 / rho);
  const maxInvRho = Math.pow(sigmaMax, 1 / rho);

  for (let i = 0; i < numSteps; i++) {
    const t = i / (numSteps - 1);
    const val = minInvRho + t * (maxInvRho - minInvRho);
    sigmas[i] = Math.pow(val, rho);
  }

  // Return in descending order (high noise to low noise)
  return sigmas.reverse();
}

/**
 * Compute exponential sigmas - geometric spacing between sigma values.
 * Formula: sigma_i = exp(log(sigma_max) + i/(N-1) * (log(sigma_min) - log(sigma_max)))
 *
 * @param sigmaMin - Minimum sigma value
 * @param sigmaMax - Maximum sigma value
 * @param numSteps - Number of steps
 * @returns Array of sigma values (descending)
 */
export function computeExponentialSigmas(
  sigmaMin: number,
  sigmaMax: number,
  numSteps: number,
): number[] {
  if (sigmaMin >= sigmaMax) {
    throw new Error('sigmaMin must be less than sigmaMax for exponential schedule');
  }

  const sigmas = new Array(numSteps);
  const logMin = Math.log(sigmaMin);
  const logMax = Math.log(sigmaMax);

  for (let i = 0; i < numSteps; i++) {
    const t = i / (numSteps - 1);
    sigmas[i] = Math.exp(logMax + t * (logMin - logMax));
  }

  return sigmas;
}

/**
 * Compute Beta distribution sigmas using a lightweight approximation.
 * Uses a simplified Beta PPF approximation for browser compatibility.
 *
 * @param sigmaMin - Minimum sigma value
 * @param sigmaMax - Maximum sigma value
 * @param numSteps - Number of steps
 * @param alpha - Beta distribution alpha parameter (default: 0.6)
 * @param beta - Beta distribution beta parameter (default: 0.6)
 * @returns Array of sigma values (descending)
 */
export function computeBetaSigmas(
  sigmaMin: number,
  sigmaMax: number,
  numSteps: number,
  alpha: number = 0.6,
  beta: number = 0.6,
): number[] {
  // Simple Beta PPF approximation using inverse CDF
  // For browser compatibility, we use a simplified approach
  const sigmas = new Array(numSteps);

  for (let i = 0; i < numSteps; i++) {
    // Uniform quantile
    const q = (i + 1) / numSteps;
    // Approximate Beta PPF using normal approximation for alpha, beta < 1
    // This is a simplified approximation - full Beta PPF is complex
    const ppf = approximateBetaPpf(q, alpha, beta);
    sigmas[i] = sigmaMin + ppf * (sigmaMax - sigmaMin);
  }

  // Return in descending order
  return sigmas.reverse();
}

/**
 * Approximate Beta PPF (Percent Point Function / Inverse CDF).
 * Uses a simplified approximation suitable for browser environments.
 *
 * @param q - Quantile (0 to 1)
 * @param alpha - Alpha parameter
 * @param beta - Beta parameter
 * @returns Approximate PPF value (0 to 1)
 */
function approximateBetaPpf(q: number, alpha: number, beta: number): number {
  // For alpha = beta = 0.6, the distribution is U-shaped
  // Use a simple transformation that approximates this behavior
  // This is NOT exact but provides reasonable non-uniform sampling

  // Mean and variance of Beta distribution
  const mean = alpha / (alpha + beta);
  const variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1));
  const stdDev = Math.sqrt(variance);

  // Approximate using normal distribution inverse CDF
  // This is a rough approximation but works for scheduling purposes
  const z = approximateNormalPpf(q);
  let result = mean + z * stdDev;

  // Clamp to [0, 1]
  return Math.max(0, Math.min(1, result));
}

/**
 * Approximate inverse normal CDF (probit function).
 * Uses Beasley-Springer-Moro algorithm approximation.
 *
 * @param p - Probability (0 to 1, exclusive)
 * @returns Approximate z-score
 */
function approximateNormalPpf(p: number): number {
  // Rational approximation for inverse normal CDF
  // aken from https://www.codeproject.com/Articles/43758/Approximating+Inverse+CDF-of-Standard-Normal-Distrib
  if (p <= 0 || p >= 1) {
    throw new Error('Probability must be strictly between 0 and 1');
  }

  const a1 = -3.969683028665376e1;
  const a2 = 2.209460984245205e2;
  const a3 = -2.759285104469687e2;
  const a4 = 1.383577518672690e2;
  const a5 = -3.066479806614716e1;
  const a6 = 2.506628277459239e0;

  const b1 = -5.447609879822406e1;
  const b2 = 1.615858368580409e2;
  const b3 = -1.556989798598866e2;
  const b4 = 6.680131188771972e1;
  const b5 = -1.328068155288572e1;

  const c1 = -7.784894002430293e-3;
  const c2 = -3.223964580491304e-2;
  const c3 = -2.400758277161838e-2;
  const c4 = -3.587409286709193e-1;
  const c5 = -3.359256673909838e0;
  const c6 = -1.374901641734137e0;

  const d1 = 4.851654746071869e-3;
  const d2 = 4.138585482528203e-2;
  const d3 = 3.766326670436303e-2;
  const d4 = 9.661797689252693e-1;
  const d5 = 4.484136212796216e0;
  const d6 = 1.374901641734137e0;

  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  let q: number;
  let r: number;

  if (p < pLow) {
    // Lower tail
    q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
      ((((d1 * q + d2) * q + d3) * q + d4) * q + d5) * q + d6
    );
  } else if (p <= pHigh) {
    // Central region
    q = p - 0.5;
    r = q * q;
    return (
      (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
      (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
    );
  } else {
    // Upper tail
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(
      (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
      ((((d1 * q + d2) * q + d3) * q + d4) * q + d5) * q + d6
    );
  }
}

/**
 * Create a linearly spaced array from start to end.
 *
 * @param start - Start value
 * @param end - End value
 * @param length - Number of elements
 * @returns Array of linearly spaced values
 */
function linspace(start: number, end: number, length: number): number[] {
  const result = new Array(length);
  const step = (end - start) / (length - 1);
  for (let i = 0; i < length; i++) {
    result[i] = start + step * i;
  }
  return result;
}
