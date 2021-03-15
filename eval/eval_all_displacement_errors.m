% Eval displacement error on all estimated warps.

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

% Number of random generation for every sequence.
nb_random = 20;

% Output directory for all computed displacement errors.
output_dir = 'out';

% Displacement threshold to mark registration as failed.
threshold = 10;

% Number of images per sequence.
nb_images = 96;

% Initialize failures for each algorithm.
success_rate_tform = zeros(length(diligent_sequences), 1);
success_rate_corr = zeros(length(diligent_sequences), 1);
success_rate_surf = zeros(length(diligent_sequences), 1);
success_rate_lowrr = zeros(length(diligent_sequences), 1);

% Initialize mean displacement errors for each algorithm.
mean_all_displacement_errors_tform = zeros(length(diligent_sequences), 1);
mean_all_displacement_errors_corr = zeros(length(diligent_sequences), 1);
mean_all_displacement_errors_surf = zeros(length(diligent_sequences), 1);
mean_all_displacement_errors_lowrr = zeros(length(diligent_sequences), 1);

% Initialize median displacement errors for each algorithm.
median_all_displacement_errors_tform = zeros(length(diligent_sequences), 1);
median_all_displacement_errors_corr = zeros(length(diligent_sequences), 1);
median_all_displacement_errors_surf = zeros(length(diligent_sequences), 1);
median_all_displacement_errors_lowrr = zeros(length(diligent_sequences), 1);

for seq_id = 1:length(diligent_sequences)
% for seq_id = 3:4
	name = diligent_sequences{seq_id};
	disp(['Sequence: ' name]);
	crop = crop_areas(seq_id, :);
	nrows = crop(3) - crop(1);
	ncols = crop(4) - crop(2);

	% Initialize displacement error for each algorithm.
	displacement_error_tform = zeros(nb_images, nb_random);
	displacement_error_corr = zeros(nb_images, nb_random);
	displacement_error_surf = zeros(nb_images, nb_random);
	displacement_error_lowrr = zeros(nb_images, nb_random);

	% Initialize failures for each algorithm.
	failures_tform = false(nb_images, nb_random);
	failures_corr = false(nb_images, nb_random);
	failures_surf = false(nb_images, nb_random);
	failures_lowrr = false(nb_images, nb_random);

	% Create directory for outputs.
	[~,~] = mkdir([ output_dir '/' name ]);

	for rand_id = 1:nb_random
	% for rand_id = 1:2
		% disp(['  random iteration: ' int2str(rand_id)]);
		this_out_dir = [ output_dir '/' name '/rand_' sprintf('%02d',rand_id) ];
		[~,~] = mkdir(this_out_dir);

		% Read matrix for ground truth warps.
		warps_gt = dlmread([this_out_dir '/warp-gt.txt']);

		% Evaluate displacement error for lowrr.
		warps_lowrr = dlmread([this_out_dir '/warp-lowrr.txt']);
		[mean_displacement_errors, failures] = eval_displacement_error(warps_gt, warps_lowrr, nrows, ncols, threshold);
		displacement_error_lowrr(:, rand_id) = mean_displacement_errors;
		failures_lowrr(:, rand_id) = failures;

		% Evaluate displacement error for tform.
		warps_tform = dlmread([this_out_dir '/warp-tform.txt']);
		[mean_displacement_errors, failures] = eval_displacement_error(warps_gt, warps_tform, nrows, ncols, threshold);
		displacement_error_tform(:, rand_id) = mean_displacement_errors;
		failures_tform(:, rand_id) = failures;

		% Evaluate displacement error for corr.
		warps_corr = dlmread([this_out_dir '/warp-corr.txt']);
		[mean_displacement_errors, failures] = eval_displacement_error(warps_gt, warps_corr, nrows, ncols, threshold);
		displacement_error_corr(:, rand_id) = mean_displacement_errors;
		failures_corr(:, rand_id) = failures;

		% Evaluate displacement error for surf.
		warps_surf = dlmread([this_out_dir '/warp-surf.txt']);
		[mean_displacement_errors, failures] = eval_displacement_error(warps_gt, warps_surf, nrows, ncols, threshold);
		displacement_error_surf(:, rand_id) = mean_displacement_errors;
		failures_surf(:, rand_id) = failures;
	end
	
	% Visualize displacement errors.
	displacement_error_lowrr(failures_lowrr) = threshold;
	displacement_error_tform(failures_tform) = threshold;
	displacement_error_corr(failures_corr) = threshold;
	displacement_error_surf(failures_surf) = threshold;

	imagesc(displacement_error_lowrr, [0,threshold]);
	colorbar;
	% title('Mean displacement error (in pixels) of every warp estimation with lowrr (lower is better)');
	ylabel('Image index $$i$$','Interpreter','Latex','Fontsize',16)
	xlabel('Random warp sampling','Interpreter','Latex','Fontsize',16);
	set(gcf, 'Position', [100, 100, 400, 300]);
	saveas(gcf,[ output_dir '/' name '/mean_displacement_error_lowrr.png']);

	imagesc(displacement_error_tform, [0,threshold]);
	colorbar;
	% title('Mean displacement error (in pixels) of every warp estimation with imregtform (lower is better)');
	ylabel('Image index $$i$$','Interpreter','Latex','Fontsize',16)
	xlabel('Random warp sampling','Interpreter','Latex','Fontsize',16);
	set(gcf, 'Position', [100, 100, 400, 300]);
	saveas(gcf,[ output_dir '/' name '/mean_displacement_error_tform.png']);

	imagesc(displacement_error_corr, [0,threshold]);
	colorbar;
	% title('Mean displacement error (in pixels) of every warp estimation with imregcorr (lower is better)');
	ylabel('Image index $$i$$','Interpreter','Latex','Fontsize',16)
	xlabel('Random warp sampling','Interpreter','Latex','Fontsize',16);
	set(gcf, 'Position', [100, 100, 400, 300]);
	saveas(gcf,[ output_dir '/' name '/mean_displacement_error_corr.png']);

	imagesc(displacement_error_surf, [0,threshold]);
	colorbar;
	% title('Mean displacement error (in pixels) of every warp estimation with SURF features (lower is better)');
	ylabel('Image index $$i$$','Interpreter','Latex','Fontsize',16)
	xlabel('Random warp sampling','Interpreter','Latex','Fontsize',16);
	set(gcf, 'Position', [100, 100, 400, 300]);
	saveas(gcf,[ output_dir '/' name '/mean_displacement_error_surf.png']);

	close all;

	% Compute failures rates and mean all displacement errors.
	success_rate_lowrr(seq_id) = sum(~failures_lowrr(:)) / numel(failures_lowrr);
	success_rate_tform(seq_id) = sum(~failures_tform(:)) / numel(failures_tform);
	success_rate_corr(seq_id) = sum(~failures_corr(:)) / numel(failures_corr);
	success_rate_surf(seq_id) = sum(~failures_surf(:)) / numel(failures_surf);

	mean_all_displacement_errors_lowrr(seq_id) = mean(displacement_error_lowrr(~failures_lowrr));
	mean_all_displacement_errors_tform(seq_id) = mean(displacement_error_tform(~failures_tform));
	mean_all_displacement_errors_corr(seq_id) = mean(displacement_error_corr(~failures_corr));
	mean_all_displacement_errors_surf(seq_id) = mean(displacement_error_surf(~failures_surf));

	median_all_displacement_errors_lowrr(seq_id) = median(displacement_error_lowrr(:));
	median_all_displacement_errors_tform(seq_id) = median(displacement_error_tform(:));
	median_all_displacement_errors_corr(seq_id) = median(displacement_error_corr(:));
	median_all_displacement_errors_surf(seq_id) = median(displacement_error_surf(:));

end % for

% Display success rate.
figure;
h = bar([success_rate_lowrr, success_rate_tform, success_rate_corr, success_rate_surf]);
set(h, {'DisplayName'}, {'lowrr', 'tform', 'corr', 'surf'}');
legend();
title('Success rate of the 4 alignment algorithms on each sequence');
ylabel('Success rate');
set(gca, 'XTick', 1:10, 'XTickLabel', diligent_sequences);

% Display mean displacement errors.
figure;
h = bar([mean_all_displacement_errors_lowrr, mean_all_displacement_errors_tform, mean_all_displacement_errors_corr,  mean_all_displacement_errors_surf]);
set(h, {'DisplayName'}, {'lowrr', 'tform', 'corr', 'surf'}');
legend();
title('Mean displacement error of successfully registered images on each sequence');
ylabel('Mean displacement error');
set(gca, 'XTick', 1:10, 'XTickLabel', diligent_sequences);

% Display median displacement errors.
figure;
h = bar([median_all_displacement_errors_lowrr(2:end), median_all_displacement_errors_tform(2:end), median_all_displacement_errors_corr(2:end)]);
set(h, {'DisplayName'}, {'lowrr', 'tform', 'corr'}');
legend();
title('Median displacement error of all registered images on each sequence');
ylabel('Median displacement error');
set(gca, 'XTick', 1:9, 'XTickLabel', diligent_sequences(2:end));
