// GGUF (GGML Universal File) format types for browser-side parsing
// Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

// ─── GGUF Tensor Types ───────────────────────────────────────────────────────
// Matches the GGUF spec tensor type enum.
// We include standard types (F32, F16) plus quantized types relevant to
// Bonsai's ternary/binary weight scheme (TQ1_0, TQ2_0, IQ1_S, IQ1_M).

export enum GgufTensorType {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q4_1_F16 = 4, // legacy, should be removed
  Q4_2 = 5, // support removed
  Q4_3 = 6, // support removed
  Q5_0 = 7, // support removed
  Q5_1 = 8, // support removed
  Q8_0 = 9,
  Q8_1 = 10,
  Q2_K = 11,
  Q3_K = 12,
  Q4_K = 13,
  Q5_K = 14,
  Q6_K = 15,
  IQ2_XXS = 16,
  IQ2_XS = 17,
  IQ3_XXS = 18,
  IQ3_XS = 19,
  IQ1_S = 20,
  IQ4_XXS = 21,
  IQ3_S = 22,
  IQ2_S = 23,
  I8 = 24,
  I16 = 25,
  I32 = 26,
  I64 = 27,
  F64 = 28,
  IQ1_M = 29,
  BF16 = 30,
  Q4_0_4_4 = 31,
  Q4_0_4_8 = 32,
  Q4_0_8_8 = 33,
  TQ1_0 = 34,  // Ternary quantization {-1, 0, 1} — Bonsai 1.58-bit
  TQ2_0 = 35,  // 2-bit ternary quantization — Bonsai 1.58-bit variant
  // Additional types may be added as GGUF spec evolves
}

// ─── GGUF Metadata Types ─────────────────────────────────────────────────────
// Matches the GGUF spec metadata value type enum.

export enum GgufMetadataType {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
}

// ─── GGUF Header ─────────────────────────────────────────────────────────────

export interface GgufHeader {
  magic: string;       // "GGUF"
  version: number;     // GGUF spec version (currently 2 or 3)
  tensorCount: bigint;
  metadataCount: bigint;
}

// ─── GGUF Tensor Info ────────────────────────────────────────────────────────
// Parsed from the tensor info section (before tensor data).

export interface GgufTensorInfo {
  name: string;
  dimensions: bigint[]; // number of dimensions (1-4)
  type: GgufTensorType;
  offset: bigint; // byte offset from start of file
  // Computed after parsing:
  shape: number[]; // dimensions as numbers
  elementCount: number;
}

// ─── GGUF Metadata Entry ─────────────────────────────────────────────────────

export interface GgufMetadataEntry {
  key: string;
  value: GgufMetadataValue;
}

export type GgufMetadataValue =
  | number
  | boolean
  | string
  | GgufMetadataValue[];

// ─── GGUF File (parsed result) ───────────────────────────────────────────────

export interface GgufFile {
  header: GgufHeader;
  metadata: Map<string, GgufMetadataValue>;
  tensorInfos: GgufTensorInfo[];
  buffer: ArrayBuffer;
}

// ─── GGUF Tensor Data (extracted tensor) ──────────────────────────────────────

export interface GgufTensorData {
  name: string;
  shape: number[];
  type: GgufTensorType;
  raw: Uint8Array; // raw quantized bytes
  dequantized?: Float32Array; // dequantized to FP32
}

// ─── Bonsai-DiT Architecture Metadata ────────────────────────────────────────
// Custom GGUF metadata keys following the "bonsai-dit.*" namespace.
// These are written by the export script and read by the browser parser.

export interface BonsaiDitMetadata {
  // Architecture dimensions
  hiddenSize: number;
  intermediateSize: number;
  numLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;

  // Latent configuration
  latentChannels: number;
  patchSize: number;
  timestepEmbeddingDim: number;

  // Quantization profile
  quantizationType: 'ternary' | 'binary';
  blockSize: number;
  scaleType: GgufTensorType;

  // Pipeline interface shapes
  textEncoderHiddenSize: number;
  vaeLatentChannels: number;

  // Generation defaults
  defaultSteps: number;
  defaultGuidanceScale: number;
  defaultSize: [number, number];

  // Scheduler
  scheduler: 'euler';
}

// ─── GGUF Metadata Key Constants ─────────────────────────────────────────────

export const GGUF_KEYS = {
  // Standard GGUF keys
  ARCHITECTURE: 'general.architecture',
  QUANTIZATION_VERSION: 'general.quantization_version',
  FILE_TYPE: 'general.file_type',

  // Bonsai-DiT custom keys
  BONSAI_PREFIX: 'bonsai-dit',
  HIDDEN_SIZE: 'bonsai-dit.hidden_size',
  INTERMEDIATE_SIZE: 'bonsai-dit.intermediate_size',
  NUM_LAYERS: 'bonsai-dit.num_layers',
  NUM_ATTENTION_HEADS: 'bonsai-dit.num_attention_heads',
  NUM_KEY_VALUE_HEADS: 'bonsai-dit.num_key_value_heads',
  LATENT_CHANNELS: 'bonsai-dit.latent_channels',
  PATCH_SIZE: 'bonsai-dit.patch_size',
  TIMESTEP_EMBEDDING_DIM: 'bonsai-dit.timestep_embedding_dim',
  QUANTIZATION_TYPE: 'bonsai-dit.quantization_type',
  BLOCK_SIZE: 'bonsai-dit.block_size',
  SCALE_TYPE: 'bonsai-dit.scale_type',
  TEXT_ENCODER_HIDDEN_SIZE: 'bonsai-dit.text_encoder_hidden_size',
  VAE_LATENT_CHANNELS: 'bonsai-dit.vae_latent_channels',
  DEFAULT_STEPS: 'bonsai-dit.default_steps',
  DEFAULT_GUIDANCE_SCALE: 'bonsai-dit.default_guidance_scale',
  SCHEDULER: 'bonsai-dit.scheduler',
} as const;

// ─── GGUF Magic & Version ────────────────────────────────────────────────────

export const GGUF_MAGIC = 'GGUF';
export const GGUF_MAGIC_BYTES = new Uint8Array([0x47, 0x47, 0x55, 0x46]); // "GGUF"
export const GGUF_VERSION = 3; // Current spec version

// ─── Tensor Type Size Helpers ────────────────────────────────────────────────

export function getTensorTypeSize(type: GgufTensorType): number | null {
  // For standard types, return bytes per element.
  // For quantized types, return null (size depends on block structure).
  switch (type) {
    case GgufTensorType.F32: return 4;
    case GgufTensorType.F16: return 2;
    case GgufTensorType.BF16: return 2;
    case GgufTensorType.I8: return 1;
    case GgufTensorType.I16: return 2;
    case GgufTensorType.I32: return 4;
    case GgufTensorType.I64: return 8;
    case GgufTensorType.F64: return 8;
    case GgufTensorType.Q8_0: return 1;
    case GgufTensorType.Q4_0: return 1; // packed
    case GgufTensorType.Q4_1: return 1; // packed
    // Quantized types — size depends on block size, return null
    case GgufTensorType.TQ1_0:
    case GgufTensorType.TQ2_0:
    case GgufTensorType.IQ1_S:
    case GgufTensorType.IQ1_M:
    case GgufTensorType.IQ2_XXS:
    case GgufTensorType.IQ2_XS:
    case GgufTensorType.IQ3_XXS:
    case GgufTensorType.IQ3_XS:
    case GgufTensorType.IQ4_XXS:
    case GgufTensorType.IQ3_S:
    case GgufTensorType.IQ2_S:
    case GgufTensorType.Q2_K:
    case GgufTensorType.Q3_K:
    case GgufTensorType.Q4_K:
    case GgufTensorType.Q5_K:
    case GgufTensorType.Q6_K:
    case GgufTensorType.Q8_1:
    default: return null;
  }
}

export function getTensorTypeName(type: GgufTensorType): string {
  return GgufTensorType[type] ?? `UNKNOWN(${type})`;
}
