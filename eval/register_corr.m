function warps = register_tform(folder)

im_files = dir([folder '/*.png']);

folder = im_files(1).folder;
im_ref_file = [folder '/' im_files(1).name];
im_ref = imread(im_ref_file);
% imwrite(im_ref, ['out/', im_files(1).name]);

nb_files = length(im_files);
warps = repmat([1 0 0 1 0 0], nb_files, 1);
for i = 2:nb_files
	name = im_files(i).name;
	im_mov_file = [folder '/' name];
	% disp(im_mov_file);
	im_mov = imread(im_mov_file);

	% Compute registration
	warp = imregcorr(im_mov, im_ref, 'similarity');
	warp_params = transpose(warp.T(:,1:2));
	warp_params = transpose(warp_params(:));
	warps(i,:) = warp_params;

	% Save registered image
	% im_registered = imwarp(im_mov, warp, 'OutputView',imref2d(size(im_ref)));
	% imwrite(im_registered, ['out/' name]);
end

end % function
