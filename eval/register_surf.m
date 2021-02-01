function warps = register_tform(folder)

im_files = dir([folder '/*.png']);

folder = im_files(1).folder;
im_ref_file = [folder '/' im_files(1).name];
im_ref = imread(im_ref_file);
% imwrite(im_ref, ['out/', im_files(1).name]);

% Detect and extract SURF features in reference image.
ptsOriginal  = detectSURFFeatures(im_ref, 'MetricThreshold', 100);
[featuresOriginal,  validPtsOriginal]  = extractFeatures(im_ref,  ptsOriginal);

nb_files = length(im_files);
warps = repmat([1 0 0 1 0 0], nb_files, 1);
for i = 2:nb_files
	name = im_files(i).name;
	im_mov_file = [folder '/' name];
	% disp(im_mov_file);
	im_mov = imread(im_mov_file);

	% Detect and extract SURF features.
	ptsDistorted = detectSURFFeatures(im_mov, 'MetricThreshold', 100);
	[featuresDistorted, validPtsDistorted] = extractFeatures(im_mov, ptsDistorted);

	% Match features.
	indexPairs = matchFeatures(featuresOriginal, featuresDistorted);

	% Retrieve points locations.
	matchedOriginal  = validPtsOriginal(indexPairs(:,1));
	matchedDistorted = validPtsDistorted(indexPairs(:,2));

	% Estimate transformation.
	try
		[warp, inlierIdx] = estimateGeometricTransform(matchedDistorted, matchedOriginal, 'affine');
		warp_params = transpose(warp.T(:,1:2));
		warp_params = transpose(warp_params(:));
		warps(i,:) = warp_params;
	catch
		warp = affine2d(eye(3));
	end
	% inlierOriginal  = matchedOriginal(inlierIdx, :);
	% inlierDistorted = matchedDistorted(inlierIdx, :);

	% Save image.
	% im_registered = imwarp(im_mov, warp, 'OutputView',imref2d(size(im_ref)));

	% Save registered image
	% imwrite(im_registered, ['out/' name]);
end

end
