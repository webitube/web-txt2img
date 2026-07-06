# Future Work: Diffusion Scheduler Enhancements

## Status: Planned

This document tracks additional diffusion schedulers and related features that are candidates for future implementation in the SD-Turbo adapter.

---

## Currently Implemented

| Scheduler | Status | Notes |
|-----------|--------|-------|
| Euler | ✅ Implemented | 1st-order baseline, default |
| DDIM | ✅ Implemented | Deterministic, stable for low step counts |
| DPM++ 2M Karras | ✅ Implemented | 2nd-order multistep with Karras sigma schedule |
| Euler Ancestral | ✅ Implemented | Stochastic variant for diverse outputs |

---

## Candidate Schedulers (Priority Ordered)

### High Priority

#### 1. DPM++ 2S Karras
- **Type:** 2nd-order single-step (simplified DPM++)
- **Why:** Fewer UNet evaluations than 2M while maintaining quality
- **Complexity:** Low — similar to DPM++ 2M but with simpler state tracking
- **Use Case:** Good middle ground between speed and quality

#### 2. DEIS (Denoisier-Euler Implicit Solver)
- **Type:** Implicit multistep
- **Why:** State-of-the-art for low step counts (1-4 steps)
- **Complexity:** Medium — requires implicit integration
- **Use Case:** SD-Turbo's sweet spot (few steps, high quality)

#### 3. UniPC (Unified Predictor-Corrector)
- **Type:** Predictor-corrector multistep
- **Why:** Excellent quality at very low step counts
- **Complexity:** Medium-High — predictor + corrector phases
- **Use Case:** Maximum quality per step

### Medium Priority

#### 4. Heun (Euler + Correction)
- **Type:** 2nd-order with correction step
- **Why:** Simple improvement over Euler
- **Complexity:** Low — adds one correction pass per step
- **Trade-off:** ~2x UNet calls per step for better quality

#### 5. LMS (Linear Multi-Step / Adams-Bashforth)
- **Type:** Higher-order multistep
- **Why:** Smooth denoising trajectories
- **Complexity:** Medium — needs N previous outputs cached
- **Use Case:** High step counts (10+)

#### 6. DPM++ SDE Karras
- **Type:** Stochastic DPM++ variant
- **Why:** Adds noise injection for diversity while maintaining quality
- **Complexity:** Medium — similar to DPM++ 2M but with SDE noise
- **Use Case:** When diversity matters more than reproducibility

### Lower Priority

#### 7. Ancestral Sampling (DDIM Ancestral)
- **Type:** DDIM + noise injection
- **Why:** Alternative to Euler Ancestral
- **Complexity:** Low
- **Note:** May be redundant with Euler Ancestral

#### 8. KDPM2 (2nd-order DPM with 2 evaluations)
- **Type:** 2nd-order with double evaluation
- **Why:** High quality per step
- **Complexity:** Medium
- **Trade-off:** 2 UNet calls per step

#### 9. LCMScheduler (Latent Consistency Model)
- **Type:** Consistency model scheduler
- **Why:** 1-4 step generation with high quality
- **Complexity:** High — requires distillation or pre-trained consistency model
- **Note:** May require model retraining, not just scheduler change

---

## Sigma Schedule Variants

Beyond schedulers, different sigma schedules can be mixed and matched:

| Schedule | Status | Notes |
|----------|--------|-------|
| Linear | ✅ Current default | Simple, works with all schedulers |
| Karras | ✅ Used by DPM++ 2M Karras | Better noise distribution |
| Exponential | ❌ Not implemented | Alternative for smooth transitions |
| Cosine | ❌ Not implemented | Used by some diffusion models |
| Polyexponential | ❌ Not implemented | Generalization of Karras |

---

## Feature Enhancements

### Guided Generation
- **CFG (Classifier-Free Guidance):** Add guidance scale parameter
- **Negative prompts:** Support anti-prompts for exclusion
- **Regional guidance:** Per-region prompt weighting

### Advanced Sampling
- **Skip-Step sampling:** Skip intermediate steps for speed
- **Resuming generation:** Save/load latent state mid-generation
- **Multi-pass refinement:** Run multiple denoising passes

### Performance
- **Batch generation:** Generate multiple images in parallel
- **Progressive refinement:** Start low-res, upscale iteratively
- **Early exit:** Stop denoising when convergence detected

---

## Implementation Notes

### Adding a New Scheduler

1. Add scheduler ID to `SchedulerId` type in `types.ts`
2. Implement step function in `sd-turbo.ts` (follow `eulerStep` pattern)
3. Add case to `schedulerStep()` dispatcher
4. If needed, add sigma schedule to `buildSigmaSchedule()`
5. Add option to UI dropdown in `index.html`
6. Update this document

### Testing Considerations

- Verify latent dimensions remain stable across all steps
- Test with step counts: 1, 3, 10, 25, 50
- Compare output quality vs. step count for each scheduler
- Ensure seed reproducibility (except for stochastic schedulers)
