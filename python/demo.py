from scipy.io import loadmat
import flatcam
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc

# Load data
meas = mpimg.imread('../sample_capture.png')  # load flatcam measurement
calib = loadmat('../flatcam_calibdata.mat')  # load calibration data
flatcam.clean_calib(calib)

# Reconstruct
lmbd = 3e-4  # L2 regularization parameter
recon = flatcam.fcrecon(meas, calib, lmbd)

# Show images
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(meas, cmap='gray')
plt.axis('off')
plt.title('FlatCam measurement')
plt.subplot(1, 2, 2)
plt.imshow(recon)
plt.axis('off')
plt.title('FlatCam reconstruction')
plt.show()
