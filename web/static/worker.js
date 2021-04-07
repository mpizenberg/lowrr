// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

console.log("Hello from worker");

const images = [];
const croppedImages = [];
// TODO: Temporary, chrome-only
const canvas = new OffscreenCanvas(100, 100);
const ctx = canvas.getContext("2d");

// Listener for messages containing data of the shape: { type, data }
// where type can be one of:
//   - "image-loaded": received when an image was loaded in the main thread
onmessage = async function (event) {
  console.log(`worker message: ${event.data.type}`);
  if (event.data.type == "image-loaded") {
    images.push(event.data.data);
  } else if (event.data.type == "run") {
    await run(event.data.data);
  }
};

// Main algorithm with the parameters passed as arguments.
async function run(params) {
  console.log("worker running with parameters:", params);
  // Reshape the canvas to the appropriate cropped size.
  if (params.crop != null) {
    canvas.width = params.crop.right - params.crop.left;
    canvas.height = params.crop.right - params.crop.left;
  } else {
    canvas.width = images[0].width;
    canvas.height = images[0].height;
  }
  // Crop all images
  let x, y;
  if (params.crop == null) {
    x = 0;
    y = 0;
  } else {
    x = params.crop.left;
    y = params.crop.top;
  }
  let width = canvas.width;
  let height = canvas.height;
  for (let img of images) {
    let croppedImg = await crop(img, x, y, width, height);
    croppedImages.push(croppedImg);
  }
}

// Temporary function just to crop a given image.
async function crop(img, x, y, width, height) {
  console.log("Cropping image ", img);
  const response = await fetch(img.url);
  const data = await response.blob();
  const imgBitmap = await createImageBitmap(data, x, y, width, height);
  ctx.drawImage(imgBitmap, 0, 0);
  imgBitmap.close();
  const croppedBlob = await canvas.convertToBlob();
  const croppedUrl = URL.createObjectURL(croppedBlob);
  return { id: img.id, url: croppedUrl, width, height };
}
