function [ recn, X_bayer ] = reconstruct_flatcam( meas, calib, lmbd )
% [ recn ] = reconstruct_flatcam( meas, calib, lmbd )
%   Performs l2 regularized FlatCam reconstruction
%   meas: FlatCam measurement
%   calib: FlatCam calibration data (contains SVD of calibration matrices)
%   lmbd: l2 regularization parameter (default: 1e-3)
%   Refer to Asif, et al (2017)

if nargin < 3
    lmbd = 1e-3;
end

Y = raw_to_bayer(meas, calib); % separate RAW measurement into color channels R,Gb,Gr,B (Bayer pattern)
Y = makeSeparable(Y); % Perform mean subtraction
% Optimal l2 regularization for each color channel
for c = 1:4
    UL = calib.UL_all(:,:,c);
    DL = calib.DL_all(:,:,c);
    VL = calib.VL_all(:,:,c);
    singL = calib.singL_all(:,c);
    UR = calib.UR_all(:,:,c);
    DR = calib.DR_all(:,:,c);
    VR = calib.VR_all(:,:,c);
    singR = calib.singR_all(:,c);
    X_bayer(:,:,c) = VL * ((DL'*UL'*Y(:,:,c)*UR*DR) ./ ((singL.^2)*(singR.^2)' + lmbd*ones(length(singL)))) * VR';
end
X_bayer(X_bayer<0) = 0; % non-negative constraint
recn = bayer2rgb(X_bayer, true); % convert Bayer pattern (R,Gb,Gr,B) into RGB

end

