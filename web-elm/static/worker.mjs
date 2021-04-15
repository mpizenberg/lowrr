// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

// Import and initialize the WebAssembly module.
// Remark: ES modules are not supported in Web Workers,
// so you have to process this file with esbuild:
// esbuild worker.mjs --bundle --outfile=worker.js
import { Lowrr as LowrrWasm, default as init } from "./pkg/lowrr_wasm.js";

// Initialize the wasm module.
// Let's hope this finishes before someone needs to call a Lowrr method.
let Lowrr;
(async function () {
  await init("./pkg/lowrr_wasm_bg.wasm");
  Lowrr = LowrrWasm.init();
})();

console.log("Hello from worker");

// Listener for messages containing data of the shape: { type, data }
// where type can be one of:
//   - "image-loaded": received when an image was loaded in the main thread
//   - "run": run the algorithm on all images
onmessage = async function (event) {
  console.log(`worker message: ${event.data.type}`);
  if (event.data.type == "image-loaded") {
    await load(event.data.data);
  } else if (event.data.type == "run") {
    await run(event.data.data);
  }
};

// Load image into wasm memory.
async function load({ id, url, width, height }) {
  console.log("Loading into wasm: " + id);
  const response = await fetch(url);
  const arrayBuffer = await response.arrayBuffer();
  Lowrr.load(new Uint8Array(arrayBuffer));
}

// Main algorithm with the parameters passed as arguments.
async function run(params) {
  croppedImages.length = 0;
  console.log("worker running with parameters:", params);
  for (let img of images) {
    croppedImages.push(await wasmCrop(img));
  }

  // Send back to main thread all cropped images.
  postMessage({ type: "cropped-images", data: croppedImages });
}

// Temporary function just to crop a given image.
async function wasmCrop(img) {
  console.log("Cropping image ", img);
  const response = await fetch(img.url);
  const arrayBuffer = await response.arrayBuffer();
  const cropped = crop(new Uint8Array(arrayBuffer));
  const croppedUrl = URL.createObjectURL(new Blob([cropped]));
  log(2, `Cropped ${img.id}`);
  return {
    id: img.id,
    url: croppedUrl,
    width: Math.floor(img.width / 2),
    height: Math.floor(img.height / 2),
  };
}

// Log something in the interface with the provided verbosity level.
function log(lvl, content) {
  postMessage({ type: "log", data: { lvl, content } });
}

// Small utility function.
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
