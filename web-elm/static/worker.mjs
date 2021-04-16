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
//   - "decode-image": decode an image provided with its url
//   - "run": run the algorithm on all images
onmessage = async function (event) {
  console.log(`worker message: ${event.data.type}`);
  if (event.data.type == "decode-image") {
    await decode(event.data.data);
    postMessage({ type: "image-decoded", data: event.data.data });
  } else if (event.data.type == "run") {
    await run(event.data.data);
  }
};

// Load image into wasm memory and decode it.
async function decode({ id, url }) {
  console.log("Loading into wasm: " + id);
  const response = await fetch(url);
  const arrayBuffer = await response.arrayBuffer();
  Lowrr.load(id, new Uint8Array(arrayBuffer));
}

// Main algorithm with the parameters passed as arguments.
async function run(params) {
  console.log("worker running with parameters:", params);
  let LowrrResult = Lowrr.run(params);
  // Send back to main thread all cropped images.
  for (let id of Lowrr.imageIds()) {
    let croppedImgArrayBuffer = LowrrResult.croppedImgFile(id);
    // Transfer the array buffer back to main thread.
    postMessage(
      {
        type: "cropped-image",
        data: { id, arrayBuffer: croppedImgArrayBuffer },
      },
      [croppedImgArrayBuffer]
    );
  }
  // Signal that we are done.
  postMessage({ type: "registration-done" });
}

// Log something in the interface with the provided verbosity level.
function log(lvl, content) {
  postMessage({ type: "log", data: { lvl, content } });
}

// Small utility function.
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
