%% Example script on how to reconstruct an image from FlatCam measurements
% For questions: jaspertan@rice.edu

meas = imread('../sample_capture.png'); % load flatcam measurement
calib = load('../flatcam_calibdata.mat'); % load calibration data

lmbd = 3e-4; % regularization parameter
recon = reconstruct_flatcam(meas, calib, lmbd); % perform reconstruction

% show images
figure
subplot(1,2,1), imshow(meas), title('FlatCam measurement');
subplot(1,2,2), imshow(recon), title('FlatCam reconstruction');
