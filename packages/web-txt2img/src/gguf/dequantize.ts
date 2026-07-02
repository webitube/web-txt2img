// Dequantization helpers for GGUF tensor types
// Supports ternary {-1, 0, 1} and binary {-1, 1} unpacking
// Reference: Bonsai-Image-Demo gemlite/HQQ packing format

import type { GgufTensorData } from './types.js';
import { GgufTensorType } from './types.js';

// Re-export for consumers
export { GgufTensorType } from './types.js';
export type { GgufTensorData } from './types.js';

// ─── Standard Type Dequantization ────────────────────────────────────────────

/**
 * Dequantize F32 tensor (identity - already FP32).
 */
export function dequantizeF32(raw: Uint8Array): Float32Array {
  return new Float32Array(raw.buffer);
}

/**
 * Dequantize F16 tensor to FP32.
 */
export function dequantizeF16(raw: Uint8Array): Float32Array {
  const f16 = new Uint16Array(raw.buffer);
  const result = new Float32Array(f16.length);

  for (let i = 0; i < f16.length; i++) {
    const bits = f16[i];
    const sign = (bits >> 15) & 0x1;
    const exp = (bits >> 10) & 0x1F;
    const frac = bits & 0x3FF;

    if (exp === 0) {
      // Subnormal or zero
      if (frac === 0) {
        result[i] = sign ? -0.0 : 0.0;
      } else {
        result[i] = (sign ? -1.0 : 1.0) * frac * 6.103515625e-05;
      }
    } else if (exp === 31) {
      // Inf or NaN
      result[i] = frac === 0 ? (sign ? -Infinity : Infinity) : NaN;
    } else {
      // Normal
      result[i] = (sign ? -1.0 : 1.0) * (1.0 + frac / 1024.0) * Math.pow(2, exp - 15);
    }
  }

  return result;
}

/**
 * Dequantize BF16 tensor to FP32.
 */
export function dequantizeBF16(raw: Uint8Array): Float32Array {
  const f16 = new Uint16Array(raw.buffer);
  const result = new Float32Array(f16.length);

  for (let i = 0; i < f16.length; i++) {
    // BF16 → FP32: just zero-extend the lower 16 bits
    result[i] = new Float32Array(new Uint16Array([f16[i] << 16]).buffer)[0];
  }

  return result;
}

// ─── Ternary Dequantization (TQ1_0, TQ2_0) ──────────────────────────────────
// Ternary quantization packs {-1, 0, 1} values using ~1.58 bits per element.
// The packing scheme uses 2 bits per value with a scale factor per block.
// Block size is typically 256 elements.

const TERNARY_BLOCK_SIZE = 256;

/**
 * Dequantize TQ1_0 ternary tensor to FP32.
 * Ternary values {-1, 0, 1} packed with scale factors.
 *
 * Packing format (per block of 256 elements):
 * - 2 bits per value (4 possible states, but only 3 used: -1, 0, 1)
 * - Scale factor (F32) for the block
 */
export function dequantizeTQ1_0(raw: Uint8Array, shape: number[]): Float32Array {
  const totalElements = shape.reduce((a, b) => a * b, 1);
  const result = new Float32Array(totalElements);

  // Each block has:
  // - 256 elements * 2 bits = 64 bytes of packed ternary values
  // - 1 F32 scale = 4 bytes
  // Total per block: 68 bytes
  const bytesPerBlock = Math.ceil(TERNARY_BLOCK_SIZE * 2 / 8) + 4; // 64 + 4
  const numBlocks = Math.ceil(totalElements / TERNARY_BLOCK_SIZE);

  let offset = 0;
  let elementIdx = 0;

  for (let block = 0; block < numBlocks; block++) {
    // Read scale factor (F32)
    const scale = new Float32Array(raw.buffer, offset, 1)[0];
    offset += 4;

    // Read packed ternary values (2 bits per value)
    const blockEnd = Math.min(elementIdx + TERNARY_BLOCK_SIZE, totalElements);
    for (let i = elementIdx; i < blockEnd; i++) {
      const bitIdx = (i - elementIdx) * 2;
      const byteIdx = Math.floor(bitIdx / 8);
      const bitOffset = bitIdx % 8;

      const val = (raw[offset + byteIdx] >> bitOffset) & 0x3;
      // 0 → -1, 1 → 0, 2 → 1, 3 → unused (treat as 0)
      let ternary: number;
      if (val === 0) ternary = -1;
      else if (val === 1) ternary = 0;
      else if (val === 2) ternary = 1;
      else ternary = 0;

      result[i] = ternary * scale;
    }

    offset += Math.ceil(TERNARY_BLOCK_SIZE * 2 / 8);
    elementIdx = blockEnd;
  }

  return result;
}

/**
 * Dequantize TQ2_0 2-bit ternary tensor to FP32.
 * Similar to TQ1_0 but with potentially different block structure.
 */
export function dequantizeTQ2_0(raw: Uint8Array, shape: number[]): Float32Array {
  // TQ2_0 uses the same packing as TQ1_0 for Bonsai weights
  return dequantizeTQ1_0(raw, shape);
}

// ─── Binary Dequantization (IQ1_S, IQ1_M) ───────────────────────────────────
// Binary quantization packs {-1, 1} or {-1, 0, 1} using 1-2 bits per element.

const BINARY_BLOCK_SIZE = 256;

/**
 * Dequantize IQ1_S binary tensor to FP32.
 * Binary values {-1, 1} packed with scale factors.
 *
 * Packing format (per block of 256 elements):
 * - 1 bit per value (0 → -1, 1 → +1)
 * - Scale factor (F32) for the block
 */
export function dequantizeIQ1_S(raw: Uint8Array, shape: number[]): Float32Array {
  const totalElements = shape.reduce((a, b) => a * b, 1);
  const result = new Float32Array(totalElements);

  // Each block has:
  // - 256 elements * 1 bit = 32 bytes of packed binary values
  // - 1 F32 scale = 4 bytes
  // Total per block: 36 bytes
  const bytesPerBlock = Math.ceil(BINARY_BLOCK_SIZE / 8) + 4; // 32 + 4
  const numBlocks = Math.ceil(totalElements / BINARY_BLOCK_SIZE);

  let offset = 0;
  let elementIdx = 0;

  for (let block = 0; block < numBlocks; block++) {
    // Read scale factor (F32)
    const scale = new Float32Array(raw.buffer, offset, 1)[0];
    offset += 4;

    // Read packed binary values (1 bit per value)
    const blockEnd = Math.min(elementIdx + BINARY_BLOCK_SIZE, totalElements);
    for (let i = elementIdx; i < blockEnd; i++) {
      const bitIdx = i - elementIdx;
      const byteIdx = Math.floor(bitIdx / 8);
      const bitOffset = bitIdx % 8;

      const val = (raw[offset + byteIdx] >> bitOffset) & 0x1;
      // 0 → -1, 1 → +1
      result[i] = (val === 0 ? -1 : 1) * scale;
    }

    offset += Math.ceil(BINARY_BLOCK_SIZE / 8);
    elementIdx = blockEnd;
  }

  return result;
}

/**
 * Dequantize IQ1_M binary tensor to FP32.
 * Binary values {-1, 0, 1} packed with scale factors.
 * Uses 2 bits per value (similar to ternary).
 */
export function dequantizeIQ1_M(raw: Uint8Array, shape: number[]): Float32Array {
  // IQ1_M uses the same packing as TQ1_0 for Bonsai weights
  return dequantizeTQ1_0(raw, shape);
}

// ─── Unified Dequantize Function ─────────────────────────────────────────────

/**
 * Dequantize a tensor based on its type.
 * Returns Float32Array with dequantized values.
 */
export function dequantize(tensor: GgufTensorData): Float32Array {
  switch (tensor.type) {
    case GgufTensorType.F32:
      return dequantizeF32(tensor.raw);
    case GgufTensorType.F16:
      return dequantizeF16(tensor.raw);
    case GgufTensorType.BF16:
      return dequantizeBF16(tensor.raw);
    case GgufTensorType.TQ1_0:
      return dequantizeTQ1_0(tensor.raw, tensor.shape);
    case GgufTensorType.TQ2_0:
      return dequantizeTQ2_0(tensor.raw, tensor.shape);
    case GgufTensorType.IQ1_S:
      return dequantizeIQ1_S(tensor.raw, tensor.shape);
    case GgufTensorType.IQ1_M:
      return dequantizeIQ1_M(tensor.raw, tensor.shape);
    default:
      throw new Error(`Unsupported tensor type for dequantization: ${tensor.type}`);
  }
}

/**
 * Dequantize in-place (modifies the tensor object).
 */
export function dequantizeInPlace(tensor: GgufTensorData): void {
  tensor.dequantized = dequantize(tensor);
}

// ─── Batch Dequantization ────────────────────────────────────────────────────

/**
 * Dequantize multiple tensors.
 */
export function dequantizeBatch(tensors: GgufTensorData[]): Float32Array[] {
  return tensors.map(dequantize);
}

/**
 * Dequantize all tensors matching a pattern from a GGUF file.
 */
export async function dequantizeTensors(
  tensorDataList: GgufTensorData[],
  onProgress?: (index: number, total: number, name: string) => void
): Promise<Map<string, Float32Array>> {
  const result = new Map<string, Float32Array>();

  for (let i = 0; i < tensorDataList.length; i++) {
    const tensor = tensorDataList[i];
    onProgress?.(i, tensorDataList.length, tensor.name);

    try {
      const dequantized = dequantize(tensor);
      result.set(tensor.name, dequantized);
    } catch (e) {
      console.warn(`Failed to dequantize tensor "${tensor.name}":`, e);
    }
  }

  return result;
}

// ─── Statistics ──────────────────────────────────────────────────────────────

/**
 * Get statistics about a dequantized tensor.
 */
export function getTensorStats(data: Float32Array): {
  min: number;
  max: number;
  mean: number;
  std: number;
  nonZero: number;
  sparsity: number;
} {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sumSq = 0;
  let nonZero = 0;

  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    sumSq += v * v;
    if (v !== 0) nonZero++;
  }

  const mean = sum / data.length;
  const std = Math.sqrt(sumSq / data.length - mean * mean);
  const sparsity = 1.0 - nonZero / data.length;

  return { min, max, mean, std, nonZero, sparsity };
}
