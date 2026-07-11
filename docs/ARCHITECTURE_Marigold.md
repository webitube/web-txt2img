# Marigold Architecture

## Overview

Marigold is a diffusion-based framework for monocular depth estimation, surface normal prediction, and intrinsic image decomposition (IID). It is built on top of the Hugging Face `diffusers` library and uses a conditional U-Net architecture where an input RGB image conditions the denoising of a random latent into a structured output (depth map, normals, or intrinsic targets).

**Key papers:**
- Marigold: https://marigoldmonodepth.github.io
- Marigold Computer Vision: https://marigoldcomputervision.github.io

**Repository:** https://github.com/prs-eth/Marigold

---

## Core Components

### 1. Submodules (Shared Across All Pipelines)

Marigold pipelines share these core submodules, inherited from `diffusers.DiffusionPipeline`:

| Submodule | Type | Role |
|-----------|------|------|
| **UNet** | `UNet2DConditionModel` | Conditional U-Net that denoises a latent, conditioned on an image latent |
| **VAE** | `AutoencoderKL` | Encodes input RGB images and decodes output latents |
| **Scheduler** | `DDIMScheduler` or `LCMScheduler` | Controls the denoising schedule (timesteps, noise prediction) |
| **Text Encoder** | `CLIPTextModel` | Encodes an **empty** text prompt (used as a fixed conditioning vector) |
| **Tokenizer** | `CLIPTokenizer` | Tokenizes the empty prompt for the text encoder |

#### Empty Text Embedding

A design quirk of Marigold: the text encoder is used to encode an **empty string** (`""`), producing a fixed embedding vector that is batched and passed to the U-Net at every denoising step. This effectively turns the text conditioning into a "null" condition, making the pipeline purely image-conditioned.

```
empty_text_embed = text_encoder(tokenizer("", ...))  # shape: [1, 1, 1024]
batch_empty_text_embed = empty_text_embed.repeat([batch_size, 1, 1])  # [B, 2, 1024]
```

### 2. Three Pipeline Variants

#### MarigoldDepthPipeline (`marigold_depth_pipeline.py`)

- **Task:** Monocular depth estimation
- **Output:** Single-channel depth map in `[0, 1]`
- **Config flags:** `scale_invariant`, `shift_invariant`, `default_denoising_steps`, `default_processing_resolution`
- **Decode path:** VAE decoder → mean across output channels → depth map

#### MarigoldNormalsPipeline (`marigold_normals_pipeline.py`)

- **Task:** Surface normal estimation
- **Output:** 3-channel normals map in `[-1, 1]`
- **Decode path:** VAE decoder → raw 3-channel output → clipped to `[-1, 1]`

#### MarigoldIIDPipeline (`marigold_iid_pipeline.py`)

- **Task:** Intrinsic image decomposition (appearance, lighting, etc.)
- **Output:** Multiple target maps (configurable via `target_properties`)
- **Decode path:** VAE decoder → `3 × n_targets` channels → split per target

### 3. Latent Space

All pipelines operate in a shared latent space:

- **Latent scale factor:** `0.18215`
- **Encoding:** `rgb_latent = vae.encoder(rgb) → quant_conv → mean × latent_scale_factor`
- **Decoding:** `depth_latent / latent_scale_factor → post_quant_conv → vae.decoder → output`
- **Latent shape:** `[B, 4, H/8, W/8]` (8× downsampling by VAE)

### 4. Inference Flow

The inference pipeline for all three variants follows the same high-level pattern:

```
┌─────────────────────────────────────────────────────────────┐
│                    IMAGE PREPROCESSING                       │
│  PIL/Tensor → RGB tensor [1,3,H,W] → resize → normalize    │
│  to [-1, 1] range                                           │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              ENSEMBLE PREPARATION                            │
│  Expand input to ensemble_size copies: [B,3,H,W]            │
│  Auto-detect batch size from VRAM/resolution table          │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              DENOISING INFERENCE (per batch)                 │
│  for each batch in loader:                                  │
│    1. Encode RGB → rgb_latent [B,4,h,w]                     │
│    2. Generate random noise → target_latent [B,4,h,w]       │
│    3. For T timesteps:                                     │
│       a. Concat [rgb_latent, target_latent] → U-Net input   │
│       b. U-Net predicts noise                              │
│       c. Scheduler steps: target_latent ← denoise           │
│    4. Decode target_latent → raw prediction                 │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              TEST-TIME ENSEMBLING                            │
│  If ensemble_size > 1:                                      │
│    - Depth:    Align via scale/shift optimization (BFGS)    │
│    - Normals:  Align via cosine similarity + closest pick   │
│    - IID:      Simple median/mean reduction                 │
│  Reduce via median (default) or mean                        │
│  Compute uncertainty (MAD / std / angular)                  │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              POST-PROCESSING                                │
│  Resize to original resolution                              │
│  Clip, colorize (depth), or convert to PIL (normals)        │
└─────────────────────────────────────────────────────────────┘
```

### 5. Test-Time Ensembling

Marigold achieves high-quality results through **test-time ensembling**:

1. **Multiple random seeds:** Run inference `ensemble_size` times with different random noise initializations
2. **Alignment:** For scale/shift-invariant models, optimize per-sample scale (`s`) and shift (`t`) parameters using BFGS to minimize pairwise differences
3. **Reduction:** Combine aligned predictions via median (default) or mean
4. **Uncertainty:** Compute MAD (Median Absolute Deviation) for depth, angular deviation for normals, or std for IID

#### Depth Ensembling Details

```python
# Cost function for BFGS optimization:
cost = Σᵢⱼ ||aligned[i] - aligned[j]||₂ + regularizer × (err_near + err_far)

# Where alignment is: aligned = depth × s + t  (affine transform)
# Regularizer pulls predictions toward [0, 1] range
```

#### Normals Ensembling Details

```python
# Compute mean normals, normalize to unit length
mean_normals = normalize(normals.mean(dim=0))

# Cosine similarity to each member
sim_cos = (mean_normals * normals).sum(dim=1).clamp(-1, 1)

# "closest" reduction: pick the member closest to mean per-pixel
# "mean" reduction: use normalized mean directly
```

### 6. Batch Size Heuristics

Marigold auto-selects batch size based on a lookup table (`bs_search_table`) that maps:

- **Resolution** (512, 768, 1024)
- **Available VRAM** (10, 23, 39, 79 GB)
- **Data type** (float32, float16)

to a recommended batch size. This ensures stable inference across hardware configurations.

### 7. Processing Resolution

- Input images are resized to a `processing_resolution` (default from model config) before inference
- After inference, predictions are upscaled back to the original input resolution
- This tradeoff balances quality (higher res = crisper) vs. context (lower res = better global structure)

---

## Model Variants (HuggingFace)

| Model | Task | Steps | Notes |
|-------|------|-------|-------|
| `prs-eth/marigold-depth-v1-1` | Depth | ~4-10 | Scale+shift invariant, DDIM |
| `prs-eth/marigold-depth-v1-0` | Depth | ~4-10 | Original, LCM scheduler |
| `prs-eth/marigold-normals-v1-1` | Normals | ~4-10 | DDIM with trailing timesteps |
| `prs-eth/marigold-iid-appearance-v1-1` | IID | ~4-10 | Appearance decomposition |
| `prs-eth/marigold-iid-lighting-v1-1` | IID | ~4-10 | Lighting decomposition |

---

## Dependencies

```
diffusers >= 0.25.0
torch == 2.4.1
torchvision == 0.19.1
transformers >= 4.32.1
accelerate >= 0.22.0
scipy           # BFGS optimizer for ensembling
matplotlib      # Color maps for depth visualization
```

---

## Key Design Decisions

1. **Diffusion for deterministic tasks:** Uses a diffusion prior (trained on depth/normals data) rather than direct regression, leveraging the generative prior for robustness
2. **Image conditioning via concatenation:** The U-Net takes `[rgb_latent, target_latent]` concatenated along the channel dimension, not cross-attention
3. **Empty text embedding:** Repurposes the text-conditional U-Net architecture by feeding a null text embedding
4. **Ensembling over single prediction:** Multiple noisy initializations + alignment significantly outperforms a single forward pass
5. **Resolution decoupling:** Process at lower resolution, upscale after — balances quality and VRAM
