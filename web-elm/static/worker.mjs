// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

// Import and initialize the WebAssembly module.
// Remark: ES modules are not supported in Web Workers,
// so you have to process this file with esbuild:
// esbuild worker.mjs --bundle --preserve-symlinks --outfile=worker.js
import { Lowrr as LowrrWasm, default as init } from "./pkg/lowrr_wasm.js";

// Initialize the wasm module.
// Let's hope this finishes before someone needs to call a Lowrr method.
let Lowrr;
(async function () {
  await init("./pkg/lowrr_wasm_bg.wasm");
  Lowrr = LowrrWasm.init();
})();

// Global module variable recording if the algorithm was asked to stop.
let stopOrder = false;

console.log("Hello from worker");

// Listener for messages containing data of the shape: { type, data }
// where type can be one of:
//   - "decode-image": decode an image provided with its url
//   - "run": run the algorithm on all images
//   - "stop": stop the alogorithm
onmessage = async function (event) {
  console.log(`worker message: ${event.data.type}`);
  if (event.data.type == "decode-image") {
    await decode(event.data.data);
    postMessage({ type: "image-decoded", data: event.data.data });
  } else if (event.data.type == "run") {
    await run(event.data.data);
  } else if (event.data.type == "stop") {
    console.log("Received STOP in worker");
    stopOrder = true;
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
  // Convert params to what is expected by the Rust code.
  const args = {
    config: {
      lambda: params.lambda,
      rho: params.rho,
      max_iterations: params.maxIterations,
      threshold: params.convergenceThreshold,
      sparse_ratio_threshold: params.sparse,
      levels: params.levels,
      verbosity: params.maxVerbosity,
    },
    equalize: 0.5,
    crop: params.crop,
  };

  // Run lowrr main registration algorithm.
  stopOrder = false;
  let motion = await Lowrr.run(args);

  // Send back to main thread all cropped images.
  const image_ids = Lowrr.image_ids();
  const imgCount = image_ids.length;
  console.log(`Encoding ${imgCount} cropped aligned images:`);
  for (let i = 0; i < imgCount; i++) {
    await shouldStop("encoding", i);
    const id = image_ids[i];
    console.log("   Encoding ", id, " ...");
    let croppedImgArrayU8 = Lowrr.cropped_img_file(i);
    // Transfer the array buffer back to main thread.
    postMessage(
      {
        type: "cropped-image",
        data: { id, arrayBuffer: croppedImgArrayU8.buffer, imgCount },
      },
      [croppedImgArrayU8.buffer]
    );
  }
  await shouldStop("done", null);
}

// Log something in the interface with the provided verbosity level.
export function appLog(lvl, content) {
  postMessage({ type: "log", data: { lvl, content } });
}

// Function regularly called in the algorithm to check if it should stop.
export async function shouldStop(step, progress) {
  postMessage({ type: "should-stop", data: { step, progress } });
  await sleep(0); // Force to give control back.
  return stopOrder;
}

// Small utility function.
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
