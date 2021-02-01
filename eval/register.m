close all;
clear all;

im_files = dir('data/*.png');

% [optimizer, metric] = imregconfig('multimodal');
[optimizer, metric] = imregconfig('monomodal');

folder = im_files(1).folder;
im_ref_file = [folder '/' im_files(1).name];
im_ref = imread(im_ref_file);
imwrite(im_ref, ['out/', im_files(1).name]);

nb_files = length(im_files);
for i = 2:nb_files
	name = im_files(i).name;
	im_mov_file = [folder '/' name];
	disp(im_mov_file);
	im_mov = imread(im_mov_file);

	% Compute registration
	% warp = imregtform(im_mov, im_ref, 'affine', optimizer, metric, 'PyramidLevels', 1);
	% warp = imregtform(im_mov, im_ref, 'translation', optimizer, metric, 'PyramidLevels', 1);
	warp = imregcorr(im_mov, im_ref, 'similarity');
	im_registered = imwarp(im_mov, warp, 'OutputView',imref2d(size(im_ref)));

	% Save registered image
	imwrite(im_registered, ['out/' name]);
end
