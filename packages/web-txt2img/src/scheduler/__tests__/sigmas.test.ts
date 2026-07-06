/**
 * Unit tests for sigma schedule computations.
 */
import { computeLinearSigmas, computeKarrasSigmas, computeExponentialSigmas } from '../sigmas';
import { SigmaSchedule } from '../schedule';

describe('Sigma Schedule Computations', () => {
  describe('computeLinearSigmas', () => {
    it('produces expected range of sigmas', () => {
      const sigmas = computeLinearSigmas(0.00085, 0.012, 1000);
      expect(sigmas.length).toBe(1000);
      // Sigmas are ascending (low noise to high noise) - caller reverses for denoising
      expect(sigmas[0]).toBeGreaterThan(0);
      expect(sigmas[sigmas.length - 1]).toBeGreaterThan(sigmas[0]);
    });

    it('produces ascending sigmas (low to high noise)', () => {
      const sigmas = computeLinearSigmas(0.00085, 0.012, 100);
      for (let i = 1; i < sigmas.length; i++) {
        expect(sigmas[i]).toBeGreaterThan(sigmas[i - 1]);
      }
    });

    it('works with scaled_linear beta schedule', () => {
      const sigmas = computeLinearSigmas(0.00085, 0.012, 100, 'scaled_linear');
      expect(sigmas.length).toBe(100);
      // Last sigma should be larger than first (ascending)
      expect(sigmas[sigmas.length - 1]).toBeGreaterThan(sigmas[0]);
    });
  });

  describe('computeKarrasSigmas', () => {
    it('produces expected number of sigmas', () => {
      const sigmas = computeKarrasSigmas(0.002, 14.6146, 50);
      expect(sigmas.length).toBe(50);
    });

    it('produces descending sigmas', () => {
      const sigmas = computeKarrasSigmas(0.002, 14.6146, 50);
      expect(sigmas[0]).toBeGreaterThan(sigmas[sigmas.length - 1]);
    });

    it('throws when sigmaMin >= sigmaMax', () => {
      expect(() => computeKarrasSigmas(10, 5, 50)).toThrow();
    });

    it('concentrates steps at lower noise levels', () => {
      const sigmas = computeKarrasSigmas(0.002, 14.6146, 100, 7.0);
      // Check that the distribution is non-uniform (Karras property)
      const firstHalf = sigmas.slice(0, 50);
      const secondHalf = sigmas.slice(50);
      const firstHalfRange = firstHalf[0] - firstHalf[firstHalf.length - 1];
      const secondHalfRange = secondHalf[0] - secondHalf[secondHalf.length - 1];
      // Karras should have more variation in the first half (high noise region)
      expect(firstHalfRange).toBeGreaterThan(secondHalfRange);
    });
  });

  describe('computeExponentialSigmas', () => {
    it('produces expected number of sigmas', () => {
      const sigmas = computeExponentialSigmas(0.002, 14.6146, 50);
      expect(sigmas.length).toBe(50);
    });

    it('produces monotonically descending sequence', () => {
      const sigmas = computeExponentialSigmas(0.002, 14.6146, 50);
      for (let i = 1; i < sigmas.length; i++) {
        expect(sigmas[i]).toBeLessThan(sigmas[i - 1]);
      }
    });

    it('throws when sigmaMin >= sigmaMax', () => {
      expect(() => computeExponentialSigmas(10, 5, 50)).toThrow();
    });
  });

  describe('SigmaSchedule edge cases', () => {
    it('handles Karras schedule with 1 step without throwing', () => {
      // Regression test: with 1 step, sigmaMin === sigmaMax which would fail Karras validation
      const schedule = new SigmaSchedule({ sigmaSchedule: 'karras' });
      expect(() => schedule.setTimesteps(1)).not.toThrow();
    });

    it('handles exponential schedule with 1 step without throwing', () => {
      const schedule = new SigmaSchedule({ sigmaSchedule: 'exponential' });
      expect(() => schedule.setTimesteps(1)).not.toThrow();
    });

    it('produces correct sigmas with 1 step', () => {
      const schedule = new SigmaSchedule({ sigmaSchedule: 'karras' });
      schedule.setTimesteps(1);
      const sigmas = schedule.getSigmas();
      expect(sigmas.length).toBe(1);
      expect(sigmas[0]).toBeGreaterThan(0);
    });
  });
});
