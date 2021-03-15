% Eval displacement error on a given transformation.
function [mean_displacement_errors, failures] = eval_displacement_error(warps_gt, warps_estimated, nrows, ncols, threshold)

% Output directory for displacement errors computations.
output_dir = 'out_displacement';
[~,~] = mkdir(output_dir);

% Generate homogeneous pixels coordinates.
[x,y] = meshgrid(0:ncols-1, 0:nrows-1);
coords = transpose([x(:), y(:), ones(numel(x), 1)]);

% Ground truth translations.
all_tx = warps_gt(:,5);
all_ty = warps_gt(:,6);

% Initialize the returned matrices.
nb_warps = size(warps_gt, 1);
mean_displacement_errors = zeros(nb_warps, 1);
failures = false(nb_warps, 1);

% Ignore 1st reference image.
for i = 2:nb_warps
	% Compute ground truth (translated) warped coordinates.
	tx = all_tx(i);
	ty = all_ty(i);
	coords_warp_gt = coords(1:2, :) + [ tx; ty ];

	% Compute estimated warped coordinates.
	warp = reshape(warps_estimated(i,:), 2, 3);
	coords_warp_estimated = warp * coords;

	% Compute displacement error for each pixel.
	displacement_error = coords_warp_estimated - coords_warp_gt;
	displacement_distance_error = sqrt(sum(displacement_error .^ 2, 1));

	% Visualize displacement error.
	% imagesc(reshape(displacement_distance_error, nrows, ncols));
	% pause;

	% Compute mean displacement error.
	mean_displacement_err = mean(displacement_distance_error);
	mean_displacement_errors(i) = mean_displacement_err;
	
	% Consider that the registration failed if this is > threshold,
	% or if the warp is perfectly [1 0 0 1 0 0] (default in case of failure).
	if (mean_displacement_err > threshold || isequal(warp,[1 0 0; 0 1 0])) % TODO: check that equality
		failures(i) = true;
	end
end

end % function
