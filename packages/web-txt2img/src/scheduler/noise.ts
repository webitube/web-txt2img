/**
 * Brownian Bridge noise sampling for SDE schedulers.
 * Provides correlated noise for stochastic solvers.
 */

/** Mulberry32 PRNG for seeded reproducibility */
function mulberry32(seed: number) {
  let t = seed >>> 0;
  return function () {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Simple Brownian Bridge noise sampler.
 * Returns correlated noise values for (sigma, sigma_next) pairs.
 */
export class SimpleBrownianBridge {
  private seed: number;
  private noiseCache: Map<string, Float32Array> = new Map();

  constructor(seed: number) {
    this.seed = seed;
  }

  /**
   * Get correlated noise for a (sigma, sigma_next) pair.
   *
   * @param sigma - Current sigma
   * @param sigmaNext - Next sigma
   * @param size - Size of noise array
   * @returns Correlated noise values
   */
  getNoise(sigma: number, sigmaNext: number, size: number): Float32Array {
    // Create cache key from sigma pair
    const key = `${sigma.toFixed(6)}-${sigmaNext.toFixed(6)}`;

    if (this.noiseCache.has(key)) {
      return new Float32Array(this.noiseCache.get(key)!);
    }

    // Generate new noise
    const rand = mulberry32(this.seed + Math.round(sigma * 1000) + Math.round(sigmaNext * 1000));
    const noise = new Float32Array(size);

    for (let i = 0; i < size; i++) {
      const u = Math.max(rand(), 1e-10);
      const v = rand();
      noise[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    this.noiseCache.set(key, noise);
    return noise;
  }

  /**
   * Clear the noise cache.
   */
  clearCache(): void {
    this.noiseCache.clear();
  }
}
