// Unit tests for GGUF parser
// These tests create synthetic GGUF buffers and verify parsing correctness

import { describe, test, expect, beforeEach } from 'vitest';
import {
  GgufTensorType,
  GgufMetadataType,
  GgufFile,
  GGUF_MAGIC,
  GGUF_VERSION,
} from '../types.js';
import {
  parseGguf,
  extractTensor,
  extractTensorsByName,
  getGgufFileInfo,
  getMetadataValue,
  getMetadataInt,
  getMetadataString,
} from '../parser.js';
import {
  dequantize,
  dequantizeF32,
  dequantizeF16,
  dequantizeTQ1_0,
  dequantizeIQ1_S,
  getTensorStats,
} from '../dequantize.js';

// ─── Helpers ─────────────────────────────────────────────────────────────────

function createSyntheticGguf(
  metadata: Map<string, any> = new Map(),
  tensors: { name: string; shape: number[]; type: GgufTensorType; data: Uint8Array }[] = []
): ArrayBuffer {
  const writer = new DataView(new ArrayBuffer(1024 * 1024)); // 1 MB buffer
  let offset = 0;

  // Write header
  const magic = GGUF_MAGIC.split('').map(c => c.charCodeAt(0));
  for (let i = 0; i < 4; i++) writer.setUint8(offset + i, magic[i]);
  offset += 4;

  writer.setUint32(offset, GGUF_VERSION, true);
  offset += 4;

  writer.setBigUint64(offset, BigInt(tensors.length), true);
  offset += 8;

  writer.setBigUint64(offset, BigInt(metadata.size), true);
  offset += 8;

  // Write metadata
  for (const [key, value] of metadata.entries()) {
    const keyBytes = new TextEncoder().encode(key);
    writer.setUint32(offset, keyBytes.length, true);
    offset += 4;
    for (let i = 0; i < keyBytes.length; i++) writer.setUint8(offset + i, keyBytes[i]);
    offset += keyBytes.length;

    if (typeof value === 'string') {
      writer.setUint32(offset, GgufMetadataType.STRING, true);
      offset += 4;
      const valBytes = new TextEncoder().encode(value);
      writer.setUint32(offset, valBytes.length, true);
      offset += 4;
      for (let i = 0; i < valBytes.length; i++) writer.setUint8(offset + i, valBytes[i]);
      offset += valBytes.length;
    } else if (typeof value === 'number') {
      writer.setUint32(offset, GgufMetadataType.FLOAT32, true);
      offset += 4;
      writer.setFloat32(offset, value, true);
      offset += 4;
    } else if (typeof value === 'boolean') {
      writer.setUint32(offset, GgufMetadataType.BOOL, true);
      offset += 4;
      writer.setUint8(offset, value ? 1 : 0);
      offset += 1;
    }
  }

  // Write tensor info
  for (const tensor of tensors) {
    const nameBytes = new TextEncoder().encode(tensor.name);
    writer.setUint32(offset, nameBytes.length, true);
    offset += 4;
    for (let i = 0; i < nameBytes.length; i++) writer.setUint8(offset + i, nameBytes[i]);
    offset += nameBytes.length;

    writer.setUint32(offset, tensor.shape.length, true);
    offset += 4;

    for (const dim of tensor.shape) {
      writer.setBigUint64(offset, BigInt(dim), true);
      offset += 8;
    }

    writer.setUint32(offset, tensor.type, true);
    offset += 4;

    // Offset will be filled after tensor data
    const tensorOffset = offset + 8;
    writer.setBigUint64(offset, BigInt(tensorOffset), true);
    offset += 8;
  }

  // Write tensor data
  for (const tensor of tensors) {
    for (let i = 0; i < tensor.data.length; i++) {
      writer.setUint8(offset + i, tensor.data[i]);
    }
    offset += tensor.data.length;
  }

  return writer.buffer.slice(0, offset);
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('GGUF Parser', () => {
  test('parses valid GGUF header', async () => {
    const buffer = createSyntheticGguf();
    const gguf = await parseGguf(buffer);
    
    expect(gguf.header.magic).toBe('GGUF');
    expect(gguf.header.version).toBe(GGUF_VERSION);
    expect(gguf.header.tensorCount).toBe(0n);
    expect(gguf.header.metadataCount).toBe(0n);
  });

  test('rejects invalid magic', async () => {
    const buffer = new ArrayBuffer(64);
    new Uint8Array(buffer).set([0x00, 0x00, 0x00, 0x00], 0); // Invalid magic
    
    await expect(parseGguf(buffer)).rejects.toThrow('Invalid GGUF file');
  });

  test('parses metadata entries', async () => {
    const metadata = new Map([
      ['general.architecture', 'bonsai-dit'],
      ['bonsai-dit.hidden_size', 3072],
      ['bonsai-dit.quantization_type', 'ternary'],
      ['bonsai-dit.default_steps', 4],
    ]);
    
    const buffer = createSyntheticGguf(metadata);
    const gguf = await parseGguf(buffer);
    
    expect(gguf.metadata.get('general.architecture')).toBe('bonsai-dit');
    expect(gguf.metadata.get('bonsai-dit.hidden_size')).toBe(3072);
    expect(gguf.metadata.get('bonsai-dit.quantization_type')).toBe('ternary');
  });

  test('parses tensor info entries', async () => {
    const tensorData = new Float32Array([1.0, 2.0, 3.0, 4.0]).buffer;
    const tensors = [
      {
        name: 'transformer.blocks.0.attn.wq',
        shape: [2, 2],
        type: GgufTensorType.F32,
        data: new Uint8Array(tensorData),
      },
    ];
    
    const buffer = createSyntheticGguf(new Map(), tensors);
    const gguf = await parseGguf(buffer);
    
    expect(gguf.tensorInfos.length).toBe(1);
    expect(gguf.tensorInfos[0].name).toBe('transformer.blocks.0.attn.wq');
    expect(gguf.tensorInfos[0].shape).toEqual([2, 2]);
    expect(gguf.tensorInfos[0].type).toBe(GgufTensorType.F32);
  });

  test('extracts tensor by name', async () => {
    const tensorData = new Float32Array([1.0, 2.0, 3.0, 4.0]).buffer;
    const tensors = [
      {
        name: 'transformer.blocks.0.attn.wq',
        shape: [2, 2],
        type: GgufTensorType.F32,
        data: new Uint8Array(tensorData),
      },
      {
        name: 'transformer.blocks.0.attn.wk',
        shape: [2, 2],
        type: GgufTensorType.F32,
        data: new Uint8Array(new Float32Array([5.0, 6.0, 7.0, 8.0]).buffer),
      },
    ];
    
    const buffer = createSyntheticGguf(new Map(), tensors);
    const gguf = await parseGguf(buffer);
    
    const tensor = extractTensor(gguf, 'transformer.blocks.0.attn.wq');
    expect(tensor).not.toBeNull();
    expect(tensor!.name).toBe('transformer.blocks.0.attn.wq');
    expect(tensor!.shape).toEqual([2, 2]);
  });

  test('extracts tensors by pattern', async () => {
    const tensorData = new Float32Array([1.0, 2.0, 3.0, 4.0]).buffer;
    const tensors = [
      {
        name: 'transformer.blocks.0.attn.wq',
        shape: [2, 2],
        type: GgufTensorType.F32,
        data: new Uint8Array(tensorData),
      },
      {
        name: 'transformer.blocks.0.attn.wk',
        shape: [2, 2],
        type: GgufTensorType.F32,
        data: new Uint8Array(new Float32Array([5.0, 6.0, 7.0, 8.0]).buffer),
      },
      {
        name: 'transformer.blocks.1.attn.wq',
        shape: [2, 2],
        type: GgufTensorType.F32,
        data: new Uint8Array(new Float32Array([9.0, 10.0, 11.0, 12.0]).buffer),
      },
    ];
    
    const buffer = createSyntheticGguf(new Map(), tensors);
    const gguf = await parseGguf(buffer);
    
    const tensors = extractTensorsByName(gguf, 'transformer.blocks.0.*');
    expect(tensors.length).toBe(2);
  });

  test('gets file info', async () => {
    const metadata = new Map([
      ['general.architecture', 'bonsai-dit'],
      ['bonsai-dit.hidden_size', 3072],
    ]);
    
    const tensorData = new Float32Array([1.0, 2.0, 3.0, 4.0]).buffer;
    const tensors = [
      {
        name: 'transformer.blocks.0.attn.wq',
        shape: [2, 2],
        type: GgufTensorType.F32,
        data: new Uint8Array(tensorData),
      },
    ];
    
    const buffer = createSyntheticGguf(metadata, tensors);
    const gguf = await parseGguf(buffer);
    
    const info = getGgufFileInfo(gguf);
    expect(info.version).toBe(GGUF_VERSION);
    expect(info.tensorCount).toBe(1);
    expect(info.metadataKeys).toContain('general.architecture');
    expect(info.tensorNames).toContain('transformer.blocks.0.attn.wq');
  });

  test('gets metadata values', async () => {
    const metadata = new Map([
      ['general.architecture', 'bonsai-dit'],
      ['bonsai-dit.hidden_size', 3072],
      ['bonsai-dit.quantization_type', 'ternary'],
    ]);
    
    const buffer = createSyntheticGguf(metadata);
    const gguf = await parseGguf(buffer);
    
    expect(getMetadataString(gguf, 'general.architecture')).toBe('bonsai-dit');
    expect(getMetadataInt(gguf, 'bonsai-dit.hidden_size')).toBe(3072);
    expect(getMetadataValue(gguf, 'nonexistent')).toBeUndefined();
  });
});

describe('Dequantization', () => {
  test('dequantizes F32 tensor', () => {
    const data = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const result = dequantizeF32(new Uint8Array(data.buffer));
    
    expect(result.length).toBe(4);
    expect(result[0]).toBe(1.0);
    expect(result[1]).toBe(2.0);
  });

  test('dequantizes F16 tensor', () => {
    // F16 representation of 1.0, 2.0, 3.0, 4.0
    const f16Data = new Uint8Array([
      0x00, 0x3c, // 1.0
      0x00, 0x40, // 2.0
      0x00, 0x41, // 3.0 (approximate)
      0x00, 0x42, // 4.0
    ]);
    
    const result = dequantizeF16(f16Data);
    expect(result.length).toBe(4);
    expect(result[0]).toBeCloseTo(1.0, 5);
    expect(result[1]).toBeCloseTo(2.0, 5);
  });

  test('dequantizes TQ1_0 ternary tensor', () => {
    // Create synthetic ternary data
    const shape = [4];
    const totalElements = 4;
    
    // Scale factor (F32) + packed ternary values
    const scale = 2.0;
    const bytesPerBlock = 4 + Math.ceil(totalElements * 2 / 8); // scale + packed values
    
    const raw = new Uint8Array(bytesPerBlock);
    new Float32Array(raw.buffer, 0, 1)[0] = scale;
    
    // Pack ternary values: {-1, 0, 1, 0}
    // 0 → -1, 1 → 0, 2 → 1
    const ternaryValues = [0, 1, 2, 1]; // -1, 0, 1, 0
    for (let i = 0; i < ternaryValues.length; i++) {
      const bitIdx = i * 2;
      const byteIdx = Math.floor(bitIdx / 8);
      const bitOffset = bitIdx % 8;
      raw[4 + byteIdx] |= (ternaryValues[i] & 0x3) << bitOffset;
    }
    
    const result = dequantizeTQ1_0(raw, shape);
    expect(result.length).toBe(4);
    expect(result[0]).toBeCloseTo(-2.0, 5); // -1 * scale
    expect(result[1]).toBeCloseTo(0.0, 5);  // 0 * scale
    expect(result[2]).toBeCloseTo(2.0, 5);  // 1 * scale
    expect(result[3]).toBeCloseTo(0.0, 5);  // 0 * scale
  });

  test('dequantizes IQ1_S binary tensor', () => {
    const shape = [4];
    const totalElements = 4;
    
    // Scale factor (F32) + packed binary values
    const scale = 3.0;
    const bytesPerBlock = 4 + Math.ceil(totalElements / 8);
    
    const raw = new Uint8Array(bytesPerBlock);
    new Float32Array(raw.buffer, 0, 1)[0] = scale;
    
    // Pack binary values: {-1, 1, -1, 1}
    // 0 → -1, 1 → 1
    const binaryValues = [0, 1, 0, 1];
    for (let i = 0; i < binaryValues.length; i++) {
      const bitIdx = i;
      const byteIdx = Math.floor(bitIdx / 8);
      const bitOffset = bitIdx % 8;
      raw[4 + byteIdx] |= binaryValues[i] << bitOffset;
    }
    
    const result = dequantizeIQ1_S(raw, shape);
    expect(result.length).toBe(4);
    expect(result[0]).toBeCloseTo(-3.0, 5); // -1 * scale
    expect(result[1]).toBeCloseTo(3.0, 5);  // 1 * scale
    expect(result[2]).toBeCloseTo(-3.0, 5); // -1 * scale
    expect(result[3]).toBeCloseTo(3.0, 5);  // 1 * scale
  });

  test('computes tensor statistics', () => {
    const data = new Float32Array([1.0, 2.0, 3.0, 0.0, 0.0]);
    const stats = getTensorStats(data);
    
    expect(stats.min).toBe(0.0);
    expect(stats.max).toBe(3.0);
    expect(stats.mean).toBeCloseTo(1.2, 5);
    expect(stats.nonZero).toBe(3);
    expect(stats.sparsity).toBeCloseTo(0.6, 5);
  });
});
