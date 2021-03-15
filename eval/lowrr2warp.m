function warp_params = lowrr2warp(lowrr_params)
% LOWRR2WARP transform the output parameters of the low-rank registration algorithm
% into the equivalent matlab warp parameters.
%
% This adds 1.0 to the diagonal parameters (scaling)
% and inverse the transformation.
%
% The input lowrr_params must be given as a row vector.
% The output warp is a row vector.

warp = reshape(lowrr_params + [1 0 0 1 0 0], 2, 3);
warp = [ warp; 0 0 1 ];
tform = invert(affine2d(transpose(warp)));
warp_params = transpose(tform.T(:,1:2));
warp_params = transpose(warp_params(:));

end % function
