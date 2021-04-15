import { crop as wasmCrop, default as wasmInit } from "./pkg/lowrr_wasm.js";

export { init, crop };

// Global variable in the module.
let lowrr_wasm = undefined;

// Initialize the wasm module.
async function init() {
  if (lowrr_wasm == undefined) {
    lowrr_wasm = await wasmInit("./pkg/lowrr_wasm_bg.wasm");
  }
  console.log("Hello from wasm");
}

// Crop an image file.
async function crop(arrayBuffer) {
  const cropped = wasmCrop(new Uint8Array(arrayBuffer));
  return cropped.buffer;
}
