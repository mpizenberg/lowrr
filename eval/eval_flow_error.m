% Eval flow error on a given transformation.
function [mean_flow_errors, failures] = eval_flow_error(warps_gt, warps_estimated, nrows, ncols)

% Output directory for flow errors computations.
output_dir = 'out_flow';
[~,~] = mkdir(output_dir);

% Generate homogeneous pixels coordinates.
[x,y] = meshgrid(0:ncols-1, 0:nrows-1);
coords = transpose([x(:), y(:), ones(numel(x), 1)]);

% Ground truth translations.
all_tx = warps_gt(:,5);
all_ty = warps_gt(:,6);

% Initialize the returned matrices.
nb_warps = size(warps_gt, 1);
mean_flow_errors = zeros(nb_warps, 1);
failures = false(nb_warps, 1);

% Ignore 1st reference image.
for i = 2:nb_warps
	% Compute ground truth (translated) warped coordinates.
	tx = all_tx(i);
	ty = all_ty(i);
	coords_warp_gt = coords(1:2, :) + [ tx; ty ]; % TODO: check correct sign

	% Compute estimated warped coordinates.
	warp = reshape(warps_estimated(i,:), 2, 3);
	coords_warp_estimated = warp * coords;

	% Compute flow error for each pixel.
	flow_error = coords_warp_estimated - coords_warp_gt;
	flow_distance_error = sqrt(sum(flow_error .^ 2, 1));

	% Visualize flow error.
	% imagesc(reshape(flow_distance_error, nrows, ncols));
	% pause;

	% Compute mean flow error.
	mean_flow_err = mean(flow_distance_error);
	mean_flow_errors(i) = mean_flow_err;
	
	% Consider that the registration failed if this is > 10,
	% or if the warp is perfectly [1 0 0 1 0 0] (default in case of failure).
	if (mean_flow_err > 10 || isequal(warp,[1 0 0; 0 1 0])) % TODO: check that equality
		failures(i) = true;
	end
end

end % function
