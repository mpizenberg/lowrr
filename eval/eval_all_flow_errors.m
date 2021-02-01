% Eval flow error on all estimated warps.

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

% Number of random generation for every sequence.
nb_random = 20;

% Output directory for all computed flow errors.
output_dir = 'out';

for seq_id = 1:length(diligent_sequences)
	name = diligent_sequences{seq_id};
	disp(['Sequence: ' name]);
	crop = crop_areas(seq_id, :);
	nrows = crop(3) - crop(1);
	ncols = crop(4) - crop(2);

	% Initialize flow error for each algorithm.
	flow_error_tform = zeros(96, nb_random);
	flow_error_corr = zeros(96, nb_random);
	flow_error_surf = zeros(96, nb_random);
	flow_error_lowrr = zeros(96, nb_random);

	% Initialize failures for each algorithm.
	failures_tform = false(96, nb_random);
	failures_corr = false(96, nb_random);
	failures_surf = false(96, nb_random);
	failures_lowrr = false(96, nb_random);

	% Create directory for outputs.
	[~,~] = mkdir([ output_dir '/' name ]);

	% for rand_id = 1:nb_random
	for rand_id = 3:nb_random
		% disp(['  random iteration: ' int2str(rand_id)]);
		this_out_dir = [ output_dir '/' name '/rand_' sprintf('%02d',rand_id) ];
		[~,~] = mkdir(this_out_dir);

		% Read matrix for ground truth warps.
		warps_gt = readmatrix([this_out_dir '/warp-gt.txt']);

		% Evaluate flow error for lowrr.
		warps_lowrr = readmatrix([this_out_dir '/warp-lowrr.txt']);
		[mean_flow_errors, failures] = eval_flow_error(warps_gt, warps_lowrr, nrows, ncols);
		flow_error_lowrr(:, rand_id) = mean_flow_errors;
		failures_lowrr(:, rand_id) = failures;

		% Evaluate flow error for tform.
		warps_tform = readmatrix([this_out_dir '/warp-tform.txt']);
		[mean_flow_errors, failures] = eval_flow_error(warps_gt, warps_tform, nrows, ncols);
		flow_error_tform(:, rand_id) = mean_flow_errors;
		failures_tform(:, rand_id) = failures;

		% Evaluate flow error for corr.
		warps_corr = readmatrix([this_out_dir '/warp-corr.txt']);
		[mean_flow_errors, failures] = eval_flow_error(warps_gt, warps_corr, nrows, ncols);
		flow_error_corr(:, rand_id) = mean_flow_errors;
		failures_corr(:, rand_id) = failures;

		% Evaluate flow error for surf.
		warps_surf = readmatrix([this_out_dir '/warp-surf.txt']);
		[mean_flow_errors, failures] = eval_flow_error(warps_gt, warps_surf, nrows, ncols);
		flow_error_surf(:, rand_id) = mean_flow_errors;
		failures_surf(:, rand_id) = failures;
	end
	
	% Visualize flow errors.
	flow_error_lowrr(failures_lowrr) = 16;
	flow_error_tform(failures_tform) = 16;
	flow_error_corr(failures_corr) = 16;
	flow_error_surf(failures_surf) = 16;
	
	[~,~] = mkdir([ output_dir '/' name ]);

	imagesc(flow_error_lowrr, [0,16]);
	colorbar;
	saveas(gcf,[ output_dir '/' name '/mean_flow_error_lowrr.png']);

	imagesc(flow_error_tform, [0,16]);
	colorbar;
	saveas(gcf,[ output_dir '/' name '/mean_flow_error_tform.png']);

	imagesc(flow_error_corr, [0,16]);
	colorbar;
	saveas(gcf,[ output_dir '/' name '/mean_flow_error_corr.png']);

	imagesc(flow_error_surf, [0,16]);
	colorbar;
	saveas(gcf,[ output_dir '/' name '/mean_flow_error_surf.png']);

	close all;

end
