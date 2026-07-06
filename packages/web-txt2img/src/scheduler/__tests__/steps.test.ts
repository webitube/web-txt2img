/**
 * Unit tests for scheduler step functions.
 */
import { createEulerStepFn } from '../steps/euler';
import { createDdimStepFn } from '../steps/ddim';
import { createDpmpp2mStepFn } from '../steps/dpmpp2m';
import { createEulerAncestralStepFn } from '../steps/eulerAncestral';
import type { SchedulerState, SchedulerConfig } from '../types';

describe('Scheduler Step Functions', () => {
  const defaultConfig: SchedulerConfig = {};
  const defaultState: SchedulerState = {
    stepIndex: 0,
    prevModelOutput: null,
    prevSigma: null,
    modelOutputs: [],
  };

  describe('Euler Step', () => {
    const stepFn = createEulerStepFn(defaultConfig);

    it('produces expected output for known inputs', () => {
      const modelOutput = new Float32Array([1, 2, 3]);
      const sample = new Float32Array([10, 20, 30]);
      const sigma = 10;
      const nextSigma = 5;

      const result = stepFn(modelOutput, sample, sigma, nextSigma, defaultState);
      // x_{t-1} = x_t + epsilon * (sigma_{t-1} - sigma_t)
      // = [10, 20, 30] + [1, 2, 3] * (5 - 10) = [10, 20, 30] + [-5, -10, -15] = [5, 10, 15]
      expect(result[0]).toBeCloseTo(5);
      expect(result[1]).toBeCloseTo(10);
      expect(result[2]).toBeCloseTo(15);
    });

    it('preserves array length', () => {
      const modelOutput = new Float32Array(100);
      const sample = new Float32Array(100);
      const result = stepFn(modelOutput, sample, 10, 5, defaultState);
      expect(result.length).toBe(100);
    });
  });

  describe('DDIM Step', () => {
    const stepFn = createDdimStepFn(defaultConfig);

    it('produces expected output', () => {
      const modelOutput = new Float32Array([1, 2, 3]);
      const sample = new Float32Array([10, 20, 30]);
      const sigma = 10;
      const nextSigma = 5;

      const result = stepFn(modelOutput, sample, sigma, nextSigma, defaultState);
      // pred_original = x_t - sigma * epsilon = [10, 20, 30] - 10 * [1, 2, 3] = [0, 0, 0]
      // ratio = nextSigma / sigma = 0.5
      // x_{t-1} = ratio * x_t + (1 - ratio) * pred_original
      // = 0.5 * [10, 20, 30] + 0.5 * [0, 0, 0] = [5, 10, 15]
      expect(result[0]).toBeCloseTo(5);
      expect(result[1]).toBeCloseTo(10);
      expect(result[2]).toBeCloseTo(15);
    });
  });

  describe('DPM++ 2M Step', () => {
    const stepFn = createDpmpp2mStepFn(defaultConfig);

    it('first step falls back to Euler', () => {
      const modelOutput = new Float32Array([1, 2, 3]);
      const sample = new Float32Array([10, 20, 30]);
      const sigma = 10;
      const nextSigma = 5;
      const state: SchedulerState = { ...defaultState, stepIndex: 0 };

      const result = stepFn(modelOutput, sample, sigma, nextSigma, state);
      // Same as Euler: [10, 20, 30] + [1, 2, 3] * (5 - 10) = [5, 10, 15]
      expect(result[0]).toBeCloseTo(5);
      expect(result[1]).toBeCloseTo(10);
      expect(result[2]).toBeCloseTo(15);
    });

    it('uses previous output for multistep', () => {
      const modelOutput = new Float32Array([1, 2, 3]);
      const sample = new Float32Array([10, 20, 30]);
      const sigma = 10;
      const nextSigma = 5;
      const state: SchedulerState = {
        stepIndex: 1,
        prevModelOutput: new Float32Array([0.5, 1, 1.5]),
        prevSigma: 15,
        modelOutputs: [],
      };

      const result = stepFn(modelOutput, sample, sigma, nextSigma, state);
      expect(result.length).toBe(3);
    });
  });

  describe('Euler Ancestral Step', () => {
    const stepFn = createEulerAncestralStepFn(defaultConfig);

    it('produces output with noise', () => {
      const modelOutput = new Float32Array([1, 2, 3]);
      const sample = new Float32Array([10, 20, 30]);
      const sigma = 10;
      const nextSigma = 5;
      const state: SchedulerState = { ...defaultState, stepIndex: 0 };

      const result = stepFn(modelOutput, sample, sigma, nextSigma, state);
      expect(result.length).toBe(3);
    });
  });
});
