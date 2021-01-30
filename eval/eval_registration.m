% Main script to evaluate the performances of lowrr
% against other alignment algorithms.

close all;
clear all;

diligent_sequences = ...
	{ 'ball'
	, 'bear'
	, 'buddha'
	, 'cat'
	, 'cow'
	, 'goblet'
	, 'harvest'
	, 'pot1'
	, 'pot2'
	, 'reading'
	}

% left, top, right, bottom
crop_areas = ...
	[ 230, 190, 390, 340 % ball
	; 200, 100, 400, 370 % bear
	; 200, 80, 400, 350 % buddha
	; 220, 90, 440, 370 % cat
	; 200, 180, 400, 340 % cow
	; 180, 80, 430, 350 % goblet
	; 150, 120, 460, 350 % harvest
	; 150, 140, 480, 370 % pot1
	; 180, 120, 450, 350 % pot2
	; 200, 140, 410, 340 % reading
	]

% For each set of images:
%  1. Generate a random translation of 1% and crop the defined crop area.
%     This is a rust program "warp_crop" that we can call like this:
%     system('warp_crop --flow 0.01 --out-dir generated path/to/images/0*.png');
%     It will generate images in generated/cropped/.
%     First image should be warped with identity to keep the same reference frame.
%     Save the warp transformation of every image, in the frame of the cropped area,
%     in a file called generated/warp-gt.txt.
%  2. Run the low rank registration (rust) algorithm on those images:
%     system('lowrr --no-img-write --out aligned/warp-lowrr.txt generated/cropped/*.png');
%     If that crashed, save the attempt as failed for all images.
%  3. Run another matlab algorithm, to align all cropped images with the first one.
%     Save every failed alignement as such.
%     Otherwise save all transformations in aligned/warp-[algo].txt
%     Try to be consistent with the format with the output of lowrr.
%  4. For every pixel, compare the true and estimated optical flow induced by each method.
%     We can use the rmse of the optical flow error as an evaluation measure.
