# Plan: Set Resolution and Local Assets for web-txt2img

This plan outlines the steps to enable custom image dimensions and transition from remote Hugging Face assets to local asset serving for the SD-Turbo pipeline.

## Phase 1: Local Asset Serving
**Goal:** Eliminate dependency on Hugging Face CDN by serving models and tokenizer assets from the local workspace.

### TODO List
- [ ] **Update `SDTurboAdapter` Base URL Logic**
    - **Files:** `packages/web-txt2img/src/adapters/sd-turbo.ts`
    - **Change:** Modify `load()` to prioritize `options.modelBaseUrl` and ensure local paths are handled correctly.
- [ ] **Configure Local Asset Path**
    - **Files:** `examples/vanilla-worker/main.js` (or app config)
    - **Change:** Set `modelBaseUrl` to point to the local assets directory.
- [ ] **Verify Local Loading**
    - **Files:** Browser Network Tab
    - **Change:** Confirm `.onnx` and tokenizer files are fetched from local origin.
- [ ] **Update Tokenizer Loading**
    - **Files:** `packages/web-txt2img/src/adapters/sd-turbo.ts`
    - **Change:** Ensure the tokenizer provider is configured to load `vocab.json` and `merges.txt` from the local path.

## Phase 2: Dynamic Resolution Support
**Goal:** Allow users to specify width and height for generated images, moving beyond the fixed 512x512 limit.

### TODO List
- [ ] **Remove Resolution Guards**
    - **Files:** `packages/web-txt2img/src/adapters/sd-turbo.ts`
    - **Change:** Remove the `if (width !== 512 || height !== 512)` check in `generate()`.
- [ ] **Implement Dynamic Latent Calculation**
    - **Files:** `packages/web-txt2img/src/adapters/sd-turbo.ts`
    - **Change:** Calculate `latent_shape` as `[1, 4, height / 8, width / 8]` and add multiple-of-8 validation.
- [ ] **Adjust ONNX Session Overrides**
    - **Files:** `packages/web-txt2img/src/adapters/sd-turbo.ts`
    - **Change:** Remove fixed `freeDimensionOverrides` for height/width in `load()` to enable dynamic shapes.
- [ ] **Update UI for Dimension Inputs**
    - **Files:** `examples/vanilla-worker/index.html`, `examples/vanilla-worker/styles.css`
    - **Change:** Add range sliders (512-1024, step 8) for width and height.
- [ ] **Connect UI to Inference Pipeline**
    - **Files:** `examples/vanilla-worker/main.js`
    - **Change:** Read slider values and pass them into the `client.generate()` params.
- [ ] **End-to-End Validation**
    - **Files:** Browser UI
    - **Change:** Verify that different resolutions produce correctly sized images.
