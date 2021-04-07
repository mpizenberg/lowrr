// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

console.log("Hello from worker");

let images = [];

// Listener for messages containing data of the shape: { type, data }
// where type can be one of:
//   - "image-loaded": received when an image was loaded in the main thread
onmessage = function (event) {
  console.log(`worker message: ${event.data.type}`);
  if (event.data.type == "image-loaded") {
    images.push(event.data.data);
  }
};
