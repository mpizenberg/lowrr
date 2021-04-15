// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

export function activatePorts(app, containerSize) {
  // Inform the Elm app when its container div gets resized.
  window.addEventListener("resize", () =>
    app.ports.resizes.send(containerSize())
  );

  let worker = new Worker("worker.js");

  // Listen to worker messages.
  worker.onmessage = async function (event) {
    if (event.data.type == "log") {
      app.ports.log.send(event.data.data);
    } else if (event.data.type == "cropped-images") {
      // Build and HTMLImageElement for each cropped image.
      const decodedImages = await Promise.all(
        event.data.data.map(async ({ id, url }) => {
          console.log("map url: " + url);
          const decodedImg = await utils.decodeImage(url);
          return { id, img: decodedImg };
        })
      );
      app.ports.receiveCroppedImages.send(decodedImages);
    } else {
      console.warn("Unknown message type:", event.data.type);
    }
  };

  // Listen for images to decode.
  app.ports.decodeImages.subscribe(async (imgs) => {
    console.log("Received images to decode");
    try {
      for (let img of imgs) {
        const image = await utils.createImageObject(img.name, img);
        const workerImage = {
          id: img.name,
          url: image.img.src,
          width: image.img.width,
          height: image.img.height,
        };
        worker.postMessage({ type: "image-loaded", data: workerImage });
        await sleep((image.width * image.height) / 50000);
        app.ports.imageDecoded.send(image);
      }
    } catch (error) {
      console.error(error);
    }
  });

  // Capture pointer events to detect a pointerup even outside the area.
  app.ports.capture.subscribe((event) => {
    event.target.setPointerCapture(event.pointerId);
  });

  function sendLog(lvl, content) {
    app.ports.log.send({ lvl, content });
  }

  // Run the registration algorithm with the provided parameters.
  app.ports.run.subscribe(async (params) => {
    worker.postMessage({ type: "run", data: params });
  });

  // Replace elm Browser.onAnimationFrameDelta that seems to have timing issues.
  // startAnimationFrameLoop(app.ports.animationFrame);

  // // Transfer archive data to wasm when the file is loaded.
  // app.ports.loadDataset.subscribe(({file: archive, camera: cam}) => {
  // 	console.log("Loading tar archive ...");
  // 	file_reader.readAsArrayBuffer(archive);
  // });

  // // Transfer archive data to wasm when the file is loaded.
  // function transferContent(arrayBuffer) {
  // 	Renderer.wasm_tracker.allocate(arrayBuffer.byteLength);
  // 	const wasm_buffer = new Uint8Array(Renderer.wasm.memory.buffer);
  // 	const start = Renderer.wasm_tracker.memory_pos();
  // 	let file_buffer = new Uint8Array(arrayBuffer);
  // 	wasm_buffer.set(file_buffer, start);
  // 	file_buffer = null; arrayBuffer = null; // Free memory.
  // }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function startAnimationFrameLoop(port) {
  let timestamp = performance.now();
  let loop = (time) => {
    window.requestAnimationFrame(loop);
    let delta = time - timestamp;
    timestamp = time;
    port.send(delta);
  };
  loop();
}

// // Port to save dependencies to the store.
// app.ports.saveDependencies.subscribe((dep) => {
// 	db.put('dependencies', dep.value, dep.key);
// });

// // Use ports to communicate with local storage.
// app.ports.store.subscribe(function(value) {
// 	if (value === null) {
// 		localStorage.removeItem(storageKey);
// 	} else {
// 		localStorage.setItem(storageKey, JSON.stringify(value));
// 	}
// });
//
// // Whenever localStorage changes in another tab, report it if necessary.
// window.addEventListener("storage", function(event) {
// 	if (event.storageArea === localStorage && event.key === storageKey) {
// 		app.ports.onStoreChange.send(event.newValue);
// 	}
// }, false);
//
//

//
// // Read config file as text and send it back to the Elm app.
// app.ports.loadConfigFile.subscribe(file => {
//   utils
//     .readJsonFile(file)
//     .then(fileAsText => app.ports.configLoaded.send(fileAsText))
//     .catch(error => console.log(error));
// });
//
// // Export / save annotations
// app.ports.export.subscribe(value => {
//   utils.download(JSON.stringify(value), "annotations.json", "application/json");
// });
