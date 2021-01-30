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
