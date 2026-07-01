import type { BackendId, ModelId, ModelInfo, RegistryEntry } from './types.js';
import { SDTurboAdapter } from './adapters/sd-turbo.js';
import { JanusProAdapter } from './adapters/janus-pro.js';
import { BonsaiAdapter } from './adapters/bonsai.js';

const REGISTRY: RegistryEntry[] = [
  {
    id: 'sd-turbo',
    displayName: 'SD-Turbo (ONNX Runtime Web)',
    task: 'text-to-image',
    supportedBackends: ['webgpu', 'wasm'],
    notes: 'Image size 512×512; seed supported.',
    sizeBytesApprox: 2398 * 1024 * 1024,
    sizeGBApprox: 2.34,
    sizeNotes: 'UNet ~640MB, text_encoder ~1700MB, vae_decoder ~95MB',
    createAdapter: () => new SDTurboAdapter(),
  },
  {
    id: 'janus-pro-1b',
    displayName: 'Janus-Pro-1B (Transformers.js)',
    task: 'text-to-image',
    supportedBackends: ['webgpu'],
    notes: 'Seed unsupported.',
    sizeBytesApprox: 2305 * 1024 * 1024,
    sizeGBApprox: 2.25,
    sizeNotes: 'Mixed-precision ONNX; varies slightly by device/dtype',
    createAdapter: () => new JanusProAdapter(),
  },
  {
    id: 'bonsai-ternary',
    displayName: 'Bonsai-Image-4B Ternary (1.58-bit)',
    task: 'text-to-image',
    supportedBackends: ['webgpu'],
    notes: 'Euler flow-matching, 4 steps, guidance 1.0. GGUF weights.',
    sizeBytesApprox: 1570 * 1024 * 1024, // ~1.21 GB DiT + ~360 MB VAE
    sizeGBApprox: 1.5,
    sizeNotes: 'DiT ~1.21GB ternary, VAE ~360MB, text_encoder ~350MB (ONNX, evictable)',
    createAdapter: () => new BonsaiAdapter('ternary'),
  },
  {
    id: 'bonsai-binary',
    displayName: 'Bonsai-Image-4B Binary (1-bit)',
    task: 'text-to-image',
    supportedBackends: ['webgpu'],
    notes: 'Euler flow-matching, 4 steps, guidance 1.0. GGUF weights.',
    sizeBytesApprox: 960 * 1024 * 1024, // ~0.6 GB DiT + ~360 MB VAE
    sizeGBApprox: 0.92,
    sizeNotes: 'DiT ~0.6GB binary, VAE ~360MB, text_encoder ~350MB (ONNX, evictable)',
    createAdapter: () => new BonsaiAdapter('binary'),
  },
];

export function listSupportedModels(): ModelInfo[] {
  return REGISTRY.map(({ createAdapter, ...info }) => info);
}

export function getModelInfo(id: ModelId): ModelInfo {
  const found = REGISTRY.find((m) => m.id === id);
  if (!found) throw new Error(`Unknown model id: ${id}`);
  const { createAdapter, ...info } = found;
  return info;
}

export function getRegistryEntry(id: ModelId): RegistryEntry {
  const found = REGISTRY.find((m) => m.id === id);
  if (!found) throw new Error(`Unknown model id: ${id}`);
  return found;
}

export function defaultBackendPreferenceFor(id: ModelId): BackendId[] {
  switch (id) {
    case 'sd-turbo':
      return ['webgpu', 'wasm'];
    case 'janus-pro-1b':
      return ['webgpu'];
    case 'bonsai-ternary':
    case 'bonsai-binary':
      return ['webgpu'];
  }
}
