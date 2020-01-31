function [ Y ] = raw_to_bayer( im, calib )
% [ Y ] = flatcam_to_bayer( im, calib )
%   im is a RAW capture (such as RAW FlatCam sensor measurements)
%   separate color channels and rotate measurement (to account for mask
%   misalignment)

im = im2double(im);

Y(:,:,1) = im(2:2:end,2:2:end); % R
Y(:,:,2) = im(1:2:end,2:2:end); % Gb
Y(:,:,3) = im(2:2:end,1:2:end); % Gr
Y(:,:,4) = im(1:2:end,1:2:end); % B

% Rotate measurement for alignment
Y = imrotate(Y, calib.angle, 'bilinear', 'crop');

% Crop sensor portion used in calibration
senSize = calib.cSize; % # of pixels to be used for one color channel
start_row = (size(Y,1) - senSize(1))/2 + 1;
end_row = start_row + senSize(1) - 1;
start_col = (size(Y,2) - senSize(2))/2 + 1;
end_col = start_col + senSize(2) - 1;
Y = Y(start_row:end_row,start_col:end_col,:);

end

