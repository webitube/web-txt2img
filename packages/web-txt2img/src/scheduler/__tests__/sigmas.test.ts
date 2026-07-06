/**
 * Unit tests for sigma schedule computations.
 */
import { computeLinearSigmas, computeKarrasSigmas, computeExponentialSigmas } from '../sigmas';
import { SigmaSchedule } from '../schedule';
import { findScheduler, listSchedulers } from '../registry';

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

    it('scaled_linear max sigma matches SD-Turbo initial noise (~14.6146)', () => {
      const sigmas = computeLinearSigmas(0.00085, 0.012, 1000, 'scaled_linear');
      // The max sigma (at t=999) should be close to 14.6146 (SD-Turbo's initial noise sigma)
      const maxSigma = sigmas[sigmas.length - 1];
      expect(maxSigma).toBeGreaterThan(10);
      expect(maxSigma).toBeLessThan(20);
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

  describe('SD-Turbo inference params validation', () => {
    const SD_TURBO_INITIAL_NOISE_SIGMA = 14.6146;
    const SD_TURBO_NUM_TRAIN_TIMESTEPS = 1000;

    it('euler scheduler 1-step: sigma matches initial noise, timestep in range', () => {
      const info = findScheduler('euler');
      expect(info).not.toBeNull();

      const schedule = new SigmaSchedule(info!.config);
      schedule.setTimesteps(1);

      const sigmas = schedule.getSigmas();
      const timesteps = schedule.getTimesteps();

      // Sigma should be close to SD-Turbo's initial noise sigma
      expect(sigmas[0]).toBeGreaterThan(10);
      expect(sigmas[0]).toBeLessThan(20);
      expect(Math.abs(sigmas[0] - SD_TURBO_INITIAL_NOISE_SIGMA)).toBeLessThan(2);

      // Timestep should be 999 (max noise)
      expect(timesteps[0]).toBe(999);
    });

    it('euler scheduler 10-steps: sigmas descending, all finite, timesteps in [0, 999]', () => {
      const info = findScheduler('euler');
      expect(info).not.toBeNull();

      const schedule = new SigmaSchedule(info!.config);
      schedule.setTimesteps(10);

      const sigmas = schedule.getSigmas();
      const timesteps = schedule.getTimesteps();

      // All sigmas must be finite and positive
      for (const s of sigmas) {
        expect(Number.isFinite(s)).toBe(true);
        expect(s).toBeGreaterThan(0);
      }

      // First sigma should match initial noise
      expect(Math.abs(sigmas[0] - SD_TURBO_INITIAL_NOISE_SIGMA)).toBeLessThan(2);

      // Sigmas should be descending (high noise → low noise)
      for (let i = 1; i < sigmas.length; i++) {
        expect(sigmas[i]).toBeLessThan(sigmas[i - 1]);
      }

      // All timesteps in valid range
      for (const t of timesteps) {
        expect(t).toBeGreaterThanOrEqual(0);
        expect(t).toBeLessThanOrEqual(SD_TURBO_NUM_TRAIN_TIMESTEPS - 1);
      }

      // Timesteps should be descending
      for (let i = 1; i < timesteps.length; i++) {
        expect(timesteps[i]).toBeLessThan(timesteps[i - 1]);
      }
    });

    it('dpmpp_2m_karras 10-steps: sigmas descending, all finite, timesteps valid', () => {
      const info = findScheduler('dpmpp_2m_karras');
      expect(info).not.toBeNull();

      const schedule = new SigmaSchedule(info!.config);
      schedule.setTimesteps(10);

      const sigmas = schedule.getSigmas();
      const timesteps = schedule.getTimesteps();

      // All sigmas must be finite and positive
      for (const s of sigmas) {
        expect(Number.isFinite(s)).toBe(true);
        expect(s).toBeGreaterThan(0);
      }

      // First sigma should be reasonable (not astronomical)
      expect(sigmas[0]).toBeLessThan(100);
      expect(sigmas[0]).toBeGreaterThan(5);

      // Sigmas should be descending
      for (let i = 1; i < sigmas.length; i++) {
        expect(sigmas[i]).toBeLessThan(sigmas[i - 1]);
      }

      // All timesteps in valid range
      for (const t of timesteps) {
        expect(t).toBeGreaterThanOrEqual(0);
        expect(t).toBeLessThanOrEqual(SD_TURBO_NUM_TRAIN_TIMESTEPS - 1);
      }
    });

    it('all non-flow schedulers produce reasonable sigmas for 10 steps', () => {
      const schedulers = listSchedulers().filter(
        (name) => !name.startsWith('flow_')
      );

      for (const name of schedulers) {
        const info = findScheduler(name);
        expect(info).not.toBeNull();

        const schedule = new SigmaSchedule(info!.config);
        schedule.setTimesteps(10);

        const sigmas = schedule.getSigmas();
        const timesteps = schedule.getTimesteps();

        // All sigmas finite and positive
        for (const s of sigmas) {
          expect(Number.isFinite(s)).toBe(true);
          expect(s).toBeGreaterThan(0);
        }

        // First sigma in reasonable range (5-50 covers all standard SD schedulers)
        expect(sigmas[0]).toBeGreaterThan(5);
        expect(sigmas[0]).toBeLessThan(50);

        // Timesteps in valid range
        for (const t of timesteps) {
          expect(t).toBeGreaterThanOrEqual(0);
          expect(t).toBeLessThanOrEqual(999);
        }
      }
    });
  });
});
