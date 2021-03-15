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
	; 200,  80, 400, 350 % buddha
	; 220,  90, 440, 370 % cat
	; 200, 180, 400, 340 % cow
	; 180,  80, 430, 350 % goblet
	; 150, 120, 460, 350 % harvest
	; 150, 140, 480, 370 % pot1
	; 180, 120, 450, 350 % pot2
	; 200, 140, 410, 340 % reading
	]

% For each set of images:
%  1. Generate a small random translation and crop it to the given frame.
%     This is a rust program "warp_crop" that we can call like this:
%     system('warp_crop --crop left top right bottom --equalize 0.15 path/to/images/0*.png');
%     It will generate images in generated/.
%     First image should be warped with identity to keep the same reference frame.
%     Save the warp transformation of every image, in the frame of the cropped area,
%     in a file called generated/warp-gt.txt.
%  2. Run the low rank registration (rust) algorithm on those images:
%     system('lowrr --levels 1 generated/*.png > aligned/warp-lowrr.txt');
%     If that crashed, save the attempt as failed for all images.
%  3. Run another matlab algorithm, to align all cropped images with the first one.
%     Save every failed alignement as such.
%     Otherwise save all transformations in aligned/warp-[algo].txt
%     Try to be consistent with the format with the output of lowrr.
%  4. For every pixel, compare the true and estimated displacement induced by each method.
%     We can use the rmse of the displacement error as an evaluation measure.

% Number of random generation for every sequence.
nb_random = 20;

% DiLiGenT dataset location.
diligent_dir = '~/Downloads/DiLiGenT/pmsData';

% Output directory for all computed warps.
output_dir = 'out';
[~,~] = mkdir(output_dir);

for seq_id = 3:4
% for seq_id = 1:length(diligent_sequences)
	name = diligent_sequences{seq_id};
	crop = crop_areas(seq_id, :);
	crop_params = [ int2str(crop(1)) ',' int2str(crop(2)) ',' int2str(crop(3)) ',' int2str(crop(4)) ];
	disp(['Processing ' name]);

	% Path to the images dataset.
	img_dir = [ diligent_dir '/' name 'PNG' ];

	% Create directory for outputs.
	[~,~] = mkdir([ output_dir '/' name ]);

	% Start randomizing warps.
	for rand_id = 1:2
	% for rand_id = 1:nb_random
		disp(['  random iteration: ' int2str(rand_id)]);
		this_out_dir = [ output_dir '/' name '/rand_' sprintf('%02d',rand_id) ];
		[~,~] = rmdir(this_out_dir, 's');
		[~,~] = mkdir(this_out_dir);

		% Generate random warps and warp-gt.txt inside directory <output_dir>/cropped/.
		cropped_dir = [ output_dir '/cropped' ];
		[~,~] = rmdir(cropped_dir, 's');
		system(['warp_crop --crop ' crop_params ' --out-dir ' cropped_dir ' --equalize 0.15 '  img_dir '/0*.png']);
		[~,~] = copyfile([cropped_dir '/warp-gt.txt'], this_out_dir);

		% Run low rank registration on those images.
		display('    Running low rank registration');
		warps_txt = [ this_out_dir '/warp-lowrr.txt' ];
		system(['lowrr --levels 1 ' cropped_dir '/*.png > ' warps_txt ' 2> /dev/null']);
		warps = dlmread(warps_txt);
		% Change output params to fit with matlab registration functions (it's the inverse).
		for row = 1:size(warps,1)
			warp_params = lowrr2warp(warps(row,:));
			warps(row,:) = warp_params;
		end
		dlmwrite(warps_txt, warps);

		% Run matlab intensity image registration on those images.
		warning ('off','all');
		display('    Running matlab imregtform');
		warps = register_tform(cropped_dir);
		dlmwrite([this_out_dir '/warp-tform.txt'], warps);

		% Run matlab image registration based on phase correlation on those images.
		warning ('off','all');
		display('    Running matlab imregcorr');
		warps = register_corr(cropped_dir);
		dlmwrite([this_out_dir '/warp-corr.txt'], warps);

		% Run matlab image registration based on SURF feature matching.
		warning ('off','all');
		display('    Running matlab SURF registration');
		warps = register_surf(cropped_dir);
		dlmwrite([this_out_dir '/warp-surf.txt'], warps);

	end
end
