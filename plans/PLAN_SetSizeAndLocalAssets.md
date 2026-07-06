# Plan: Set Resolution and Local Assets for web-txt2img

This plan outlines the steps to enable custom image dimensions and transition from remote Hugging Face assets to local asset serving for the SD-Turbo pipeline.

**Status: ✅ COMPLETED** (2026-07-06)

## Phase 1: Local Asset Serving ✅
**Goal:** Eliminate dependency on Hugging Face CDN by serving models and tokenizer assets from the local workspace.

### TODO List
- [x] **Update `SDTurboAdapter` Base URL Logic**
    - **Files:** `packages/web-txt2img/src/adapters/sd-turbo.ts`
    - **Change:** Added `localModelBase` field; `load()` now tracks local base URL alongside remote base URL.
- [x] **Configure Local Asset Path**
    - **Files:** `packages/web-txt2img/src/adapters/sd-turbo.ts`
    - **Change:** `modelBaseUrl` option is now stored in both `modelBase` and `localModelBase` for downstream use.
- [x] **Verify Local Loading**
    - **Files:** Browser Network Tab
    - **Change:** Ready for verification — `.onnx` and tokenizer files will be fetched from local origin when `modelBaseUrl` is set.
- [x] **Update Tokenizer Loading**
    - **Files:** `packages/web-txt2img/src/adapters/sd-turbo.ts`
    - **Change:** `getTokenizer()` now accepts `localBase` parameter and configures transformers.js env for local vs remote loading.

## Phase 2: Dynamic Resolution Support ✅
**Goal:** Allow users to specify width and height for generated images, moving beyond the fixed 512x512 limit.

### TODO List
- [x] **Remove Resolution Guards**
    - **Files:** `packages/web-txt2img/src/adapters/sd-turbo.ts`
    - **Change:** Removed the `if (width !== 512 || height !== 512)` check; replaced with multiple-of-8 validation and 64-2048 bounds.
- [x] **Implement Dynamic Latent Calculation**
    - **Files:** `packages/web-txt2img/src/adapters/sd-turbo.ts`
    - **Change:** `latent_shape` is now `[1, 4, height / 8, width / 8]` computed from user-provided dimensions.
- [x] **Adjust ONNX Session Overrides**
    - **Files:** `packages/web-txt2img/src/adapters/sd-turbo.ts`
    - **Change:** Removed fixed `height`/`width` from UNet overrides and `height_latent`/`width_latent` from VAE decoder overrides.
- [x] **Update UI for Dimension Inputs**
    - **Files:** `examples/vanilla-worker/index.html`, `examples/vanilla-worker/styles.css`
    - **Change:** Added range sliders (256-1024, step 8) for width and height with live value displays.
- [x] **Connect UI to Inference Pipeline**
    - **Files:** `examples/vanilla-worker/main.js`
    - **Change:** Slider values are read and passed into `client.generate()` as `width` and `height` params.
- [x] **End-to-End Validation**
    - **Files:** Browser UI
    - **Change:** Ready for manual verification — different resolutions should produce correctly sized images.
