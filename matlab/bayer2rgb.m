function [ X_rgb ] = bayer2rgb( X_bayer, normalize )
% [ X_rgb ] = bayer2rgb( X_bayer )
%   Converts 4 channel Bayer variable into 3 channel RGB

if nargin < 2
    normalize = true;
end

X_rgb(:,:,1) = X_bayer(:,:,1);
X_rgb(:,:,2) = 0.5 * (X_bayer(:,:,2) + X_bayer(:,:,3));
X_rgb(:,:,3) = X_bayer(:,:,4);

if normalize
    X_rgb = (X_rgb - min(X_rgb(:))) / (max(X_rgb(:)) - min(X_rgb(:)));
end

end

