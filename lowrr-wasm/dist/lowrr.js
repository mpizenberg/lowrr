import { crop as wasmCrop, default as wasmInit } from "./pkg/lowrr_wasm.js";

export { init, crop };

// Global variable in the module.
let wasm = undefined;

// Initialize the wasm module.
async function init() {
  if (wasm != undefined) {
    wasm = await wasmInit("./pkg/lowrr_wasm_bg.wasm");
  }
}

// Crop an image file.
async function crop(arrayBuffer) {
  const cropped = wasmCrop(new Uint8Array(arrayBuffer));
  return cropped.buffer;
}
