// GGUF binary parser for browser-side reading of .gguf files
// Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

import type {
  GgufHeader,
  GgufTensorInfo,
  GgufMetadataValue,
  GgufFile,
  GgufTensorData,
} from './types.js';
import {
  GgufTensorType,
  GgufMetadataType,
  GGUF_MAGIC,
  GGUF_MAGIC_BYTES,
  GGUF_VERSION,
} from './types.js';

// ─── Alignment Helpers ───────────────────────────────────────────────────────

const GGUF_ALIGNMENT = 8;

function align(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

function alignBigInt(value: bigint, alignment: number): bigint {
  const alignBig = BigInt(alignment);
  return (value + alignBig - 1n) / alignBig * alignBig;
}

// ─── Binary Reader ───────────────────────────────────────────────────────────

class BinaryReader {
  private view: DataView;
  private offset: number;

  constructor(buffer: ArrayBuffer) {
    this.view = new DataView(buffer);
    this.offset = 0;
  }

  get position(): number {
    return this.offset;
  }

  set position(pos: number) {
    this.offset = pos;
  }

  get remaining(): number {
    return this.view.byteLength - this.offset;
  }

  readBytes(length: number): Uint8Array {
    const result = new Uint8Array(this.view.buffer, this.offset, length);
    this.offset += length;
    return result;
  }

  readUint8(): number {
    const val = this.view.getUint8(this.offset);
    this.offset += 1;
    return val;
  }

  readInt8(): number {
    const val = this.view.getInt8(this.offset);
    this.offset += 1;
    return val;
  }

  readUint16(): number {
    const val = this.view.getUint16(this.offset, true); // little-endian
    this.offset += 2;
    return val;
  }

  readInt16(): number {
    const val = this.view.getInt16(this.offset, true);
    this.offset += 2;
    return val;
  }

  readUint32(): number {
    const val = this.view.getUint32(this.offset, true);
    this.offset += 4;
    return val;
  }

  readInt32(): number {
    const val = this.view.getInt32(this.offset, true);
    this.offset += 4;
    return val;
  }

  readFloat32(): number {
    const val = this.view.getFloat32(this.offset, true);
    this.offset += 4;
    return val;
  }

  readFloat64(): number {
    const val = this.view.getFloat64(this.offset, true);
    this.offset += 8;
    return val;
  }

  readUint64(): bigint {
    const low = this.view.getUint32(this.offset, true);
    const high = this.view.getUint32(this.offset + 4, true);
    this.offset += 8;
    return BigInt(high) * 4294967296n + BigInt(low);
  }

  readInt64(): bigint {
    const low = this.view.getUint32(this.offset, true);
    const high = this.view.getInt32(this.offset + 4, true);
    this.offset += 8;
    // Handle signed 64-bit
    if (high < 0) {
      return -(BigInt(-high) * 4294967296n - BigInt(low) - 1n);
    }
    return BigInt(high) * 4294967296n + BigInt(low);
  }

  readString(): string {
    const length = this.readUint32();
    const bytes = this.readBytes(length);
    return new TextDecoder().decode(bytes);
  }

  readBool(): boolean {
    return this.readUint8() !== 0;
  }
}

// ─── Metadata Parsing ────────────────────────────────────────────────────────

function parseMetadataValue(reader: BinaryReader, type: GgufMetadataType): GgufMetadataValue {
  switch (type) {
    case GgufMetadataType.UINT8:
      return reader.readUint8();
    case GgufMetadataType.INT8:
      return reader.readInt8();
    case GgufMetadataType.UINT16:
      return reader.readUint16();
    case GgufMetadataType.INT16:
      return reader.readInt16();
    case GgufMetadataType.UINT32:
      return reader.readUint32();
    case GgufMetadataType.INT32:
      return reader.readInt32();
    case GgufMetadataType.FLOAT32:
      return reader.readFloat32();
    case GgufMetadataType.BOOL:
      return reader.readBool();
    case GgufMetadataType.STRING:
      return reader.readString();
    case GgufMetadataType.UINT64:
      return Number(reader.readUint64());
    case GgufMetadataType.INT64:
      return Number(reader.readInt64());
    case GgufMetadataType.FLOAT64:
      return reader.readFloat64();
    case GgufMetadataType.ARRAY:
      return parseArray(reader);
    default:
      throw new Error(`Unknown GGUF metadata type: ${type}`);
  }
}

function parseArray(reader: BinaryReader): GgufMetadataValue[] {
  const elementType = reader.readUint32();
  const elementCount = reader.readUint32();
  const result: GgufMetadataValue[] = [];
  for (let i = 0; i < elementCount; i++) {
    result.push(parseMetadataValue(reader, elementType));
  }
  return result;
}

// ─── Header Parsing ──────────────────────────────────────────────────────────

function parseHeader(reader: BinaryReader): GgufHeader {
  // Read magic bytes
  const magicBytes = reader.readBytes(4);
  const magic = new TextDecoder().decode(magicBytes);
  if (magic !== GGUF_MAGIC) {
    throw new Error(`Invalid GGUF file: expected magic "${GGUF_MAGIC}", got "${magic}"`);
  }

  // Read version
  const version = reader.readUint32();
  if (version !== GGUF_VERSION) {
    console.warn(`GGUF version ${version} may not be fully supported (expected ${GGUF_VERSION})`);
  }

  // Read tensor count and metadata count
  const tensorCount = reader.readUint64();
  const metadataCount = reader.readUint64();

  return {
    magic,
    version,
    tensorCount,
    metadataCount,
  };
}

// ─── Metadata Parsing ────────────────────────────────────────────────────────

async function parseMetadata(reader: BinaryReader, count: bigint): Promise<Map<string, GgufMetadataValue>> {
  const metadata = new Map<string, GgufMetadataValue>();

  for (let i = 0n; i < count; i++) {
    const key = reader.readString();
    const valueType = reader.readUint32();
    const value = parseMetadataValue(reader, valueType);
    metadata.set(key, value);
  }

  return metadata;
}

// ─── Tensor Info Parsing ─────────────────────────────────────────────────────

function parseTensorInfos(reader: BinaryReader, count: bigint): GgufTensorInfo[] {
  const tensorInfos: GgufTensorInfo[] = [];

  for (let i = 0n; i < count; i++) {
    const name = reader.readString();
    const numDims = reader.readUint32();
    const dimensions: bigint[] = [];
    for (let d = 0; d < numDims; d++) {
      dimensions.push(reader.readUint64());
    }
    const type = reader.readUint32() as GgufTensorType;
    const offset = reader.readUint64();

    // Compute shape and element count
    const shape = dimensions.map(d => Number(d));
    const elementCount = shape.reduce((acc, dim) => acc * dim, 1);

    tensorInfos.push({
      name,
      dimensions,
      type,
      offset,
      shape,
      elementCount,
    });
  }

  return tensorInfos;
}

// ─── Main Parser ─────────────────────────────────────────────────────────────

/**
 * Parse a complete GGUF file from an ArrayBuffer.
 * This is the main entry point for GGUF parsing.
 */
export async function parseGguf(buffer: ArrayBuffer): Promise<GgufFile> {
  const reader = new BinaryReader(buffer);

  // Parse header
  const header = parseHeader(reader);

  // Parse metadata entries
  const metadata = await parseMetadata(reader, header.metadataCount);

  // Parse tensor info entries
  const tensorInfos = parseTensorInfos(reader, header.tensorCount);

  return {
    header,
    metadata,
    tensorInfos,
    buffer,
  };
}

/**
 * Parse only the header and metadata (partial loading).
 * Useful for inspecting file metadata without loading tensor data.
 */
export async function parseGgufHeader(buffer: ArrayBuffer, maxBytes: number = 10 * 1024 * 1024): Promise<{
  header: GgufHeader;
  metadata: Map<string, GgufMetadataValue>;
}> {
  // Read only the first N bytes (enough for header + metadata)
  const sliceSize = Math.min(maxBytes, buffer.byteLength);
  const slice = buffer.slice(0, sliceSize);
  const reader = new BinaryReader(slice);

  const header = parseHeader(reader);
  const metadata = await parseMetadata(reader, header.metadataCount);

  return { header, metadata };
}

// ─── Tensor Extraction ───────────────────────────────────────────────────────

/**
 * Extract a raw tensor by name from a parsed GGUF file.
 */
export function extractTensor(gguf: GgufFile, tensorName: string): GgufTensorData | null {
  const tensorInfo = gguf.tensorInfos.find(t => t.name === tensorName);
  if (!tensorInfo) {
    return null;
  }
  return extractTensorByInfo(gguf, tensorInfo);
}

/**
 * Extract multiple tensors matching a pattern (simple wildcard with *).
 */
export function extractTensorsByName(gguf: GgufFile, pattern: string): GgufTensorData[] {
  // Convert simple glob pattern to regex
  const regexPattern = pattern.replace(/\*/g, '.*');
  const regex = new RegExp(regexPattern);

  return gguf.tensorInfos
    .filter(t => regex.test(t.name))
    .map(t => extractTensorByInfo(gguf, t))
    .filter((t): t is GgufTensorData => t !== null);
}

function extractTensorByInfo(gguf: GgufFile, tensorInfo: GgufTensorInfo): GgufTensorData | null {
  try {
    // Calculate tensor data size based on type and shape
    const byteLength = getTensorDataByteLength(tensorInfo);
    if (byteLength <= 0) {
      console.warn(`Cannot extract tensor "${tensorInfo.name}": unknown byte length for type ${tensorInfo.type}`);
      return null;
    }

    const offset = Number(tensorInfo.offset);
    const raw = new Uint8Array(gguf.buffer, offset, byteLength);

    return {
      name: tensorInfo.name,
      shape: tensorInfo.shape,
      type: tensorInfo.type,
      raw: new Uint8Array(raw), // copy to avoid ArrayBuffer reference
    };
  } catch (e) {
    console.warn(`Failed to extract tensor "${tensorInfo.name}":`, e);
    return null;
  }
}

/**
 * Calculate the byte length of tensor data based on type and shape.
 * For standard types, this is elementCount * bytesPerElement.
 * For quantized types, this depends on the block structure.
 */
function getTensorDataByteLength(tensorInfo: GgufTensorInfo): number {
  const elementCount = tensorInfo.elementCount;

  switch (tensorInfo.type) {
    case GgufTensorType.F32:
      return elementCount * 4;
    case GgufTensorType.F16:
    case GgufTensorType.BF16:
      return elementCount * 2;
    case GgufTensorType.I8:
      return elementCount;
    case GgufTensorType.I16:
      return elementCount * 2;
    case GgufTensorType.I32:
      return elementCount * 4;
    case GgufTensorType.I64:
      return elementCount * 8;
    case GgufTensorType.F64:
      return elementCount * 8;
    // Quantized types — approximate sizes based on block structure
    // TQ1_0: ternary {-1, 0, 1}, ~1.58 bits per element
    case GgufTensorType.TQ1_0:
      return Math.ceil(elementCount * 2 / 8); // ~2 bits per element (rounded up)
    // TQ2_0: 2-bit ternary
    case GgufTensorType.TQ2_0:
      return Math.ceil(elementCount * 2 / 8);
    // IQ1_S: binary {-1, 1}, 1 bit per element
    case GgufTensorType.IQ1_S:
      return Math.ceil(elementCount / 8);
    // IQ1_M: binary {-1, 0, 1}, ~1.58 bits per element
    case GgufTensorType.IQ1_M:
      return Math.ceil(elementCount * 2 / 8);
    // Fallback: estimate based on type
    default:
      return elementCount; // rough estimate
  }
}

// ─── GGUF File Info ──────────────────────────────────────────────────────────

/**
 * Get a summary of a GGUF file without loading tensor data.
 */
export function getGgufFileInfo(gguf: GgufFile): {
  version: number;
  tensorCount: number;
  metadataKeys: string[];
  tensorNames: string[];
  totalElements: number;
} {
  return {
    version: gguf.header.version,
    tensorCount: gguf.tensorInfos.length,
    metadataKeys: Array.from(gguf.metadata.keys()),
    tensorNames: gguf.tensorInfos.map(t => t.name),
    totalElements: gguf.tensorInfos.reduce((sum, t) => sum + t.elementCount, 0),
  };
}

/**
 * Get metadata value by key with type safety.
 */
export function getMetadataValue(gguf: GgufFile, key: string, expectedType?: 'number' | 'string' | 'boolean' | 'array'): GgufMetadataValue | undefined {
  const value = gguf.metadata.get(key);
  if (value === undefined) {
    return undefined;
  }
  if (expectedType && typeof value !== expectedType && (expectedType !== 'array' || !Array.isArray(value))) {
    console.warn(`Metadata key "${key}" has unexpected type: expected ${expectedType}, got ${typeof value}`);
  }
  return value;
}

/**
 * Get an integer metadata value.
 */
export function getMetadataInt(gguf: GgufFile, key: string): number | undefined {
  const value = getMetadataValue(gguf, key, 'number');
  if (typeof value === 'number') {
    return Math.round(value);
  }
  return undefined;
}

/**
 * Get a string metadata value.
 */
export function getMetadataString(gguf: GgufFile, key: string): string | undefined {
  const value = getMetadataValue(gguf, key, 'string');
  if (typeof value === 'string') {
    return value;
  }
  return undefined;
}
