declare module 'onnxruntime-web';
declare module 'onnxruntime-web/webgpu';
declare module 'onnxruntime-web/wasm';
declare module '@huggingface/transformers';
declare module '@xenova/transformers';

// WebGPU type declarations (not in ES2021 DOM lib)
declare interface GPUDevice {
  createBuffer(options: GPUBufferDescriptor): GPUBuffer;
  queue: GPUQueue;
}
declare interface GPUBuffer {
  destroy?(): void;
}
declare interface GPUQueue {
  writeBuffer(buffer: GPUBuffer, offset: number, data: ArrayBufferView | ArrayBuffer): void;
}
declare interface GPUBufferDescriptor {
  size: number;
  usage: number;
}
declare const GPUBufferUsage: {
  STORAGE: number;
  COPY_DST: number;
  COPY_SRC: number;
  UNIFORM: number;
  INDEX: number;
  VERTEX: number;
};

declare var navigator: { gpu?: { requestDevice(): Promise<GPUDevice> } };

declare class OffscreenCanvas {
  constructor(width: number, height: number);
  width: number;
  height: number;
  getContext(contextId: string): any;
  convertToBlob(options?: { type?: string }): Promise<Blob>;
}

declare class ImageData {
  constructor(data: Uint8ClampedArray, width: number, height: number);
}
