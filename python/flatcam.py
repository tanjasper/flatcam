import numpy as np
import math
from scipy.ndimage import rotate as imrotate
from numpy.linalg import multi_dot


def clean_calib( calib ):
    # Fix any formatting issues from Matlab to Python
    calib['cSize'] = np.squeeze(calib['cSize'])
    calib['angle'] = np.squeeze(calib['angle'])


def obtain_calib_svd( calib ):
    calib_svd = calib # note, what happens to calib_svd happens to calib. To make just a copy, calib_svd = dict(calib)
    clean_calib(calib_svd)
    P1 = np.dstack((calib['P1r'], calib['P1gb'], calib['P1gr'], calib['P1b']))
    Q1 = np.dstack((calib['Q1r'], calib['Q1gb'], calib['Q1gr'], calib['Q1b']))
    # Initialize new entries for calib data struct
    calib_svd['UL_all'] = np.empty([P1.shape[0], P1.shape[0], 4])
    calib_svd['DL_all'] = np.empty([P1.shape[0], P1.shape[1], 4])
    calib_svd['VL_all'] = np.empty([P1.shape[1], P1.shape[1], 4])
    calib_svd['singL_all'] = np.empty([P1.shape[1], 4])
    calib_svd['UR_all'] = np.empty([Q1.shape[0], Q1.shape[0], 4])
    calib_svd['DR_all'] = np.empty([Q1.shape[0], Q1.shape[1], 4])
    calib_svd['VR_all'] = np.empty([Q1.shape[1], Q1.shape[1], 4])
    calib_svd['singR_all'] = np.empty([Q1.shape[1], 4])
    for i in range(4):
        # Left matrices (P1)
        u, s, vh = np.linalg.svd(P1[:, :, i], full_matrices=True)
        calib_svd['UL_all'][:, :, i] = u
        calib_svd['DL_all'][:, :, i] = np.concatenate((np.diag(s), np.zeros([P1.shape[0] - s.size, s.size])))
        calib_svd['VL_all'][:, :, i] = vh.T
        calib_svd['singL_all'][:, i] = s
        # Right matrices (Q1)
        u, s, vh = np.linalg.svd(Q1[:, :, i], full_matrices=True)
        calib_svd['UR_all'][:, :, i] = u
        calib_svd['DR_all'][:, :, i] = np.concatenate((np.diag(s), np.zeros([Q1.shape[0] - s.size, s.size])))
        calib_svd['VR_all'][:, :, i] = vh.T
        calib_svd['singR_all'][:, i] = s


def fc2bayer( im, calib ):
    # split up different color channels
    r = im[1::2, 1::2]
    gb = im[0::2, 1::2]
    gr = im[1::2, 0::2]
    b = im[0::2, 0::2]
    Y = np.dstack([r, gb, gr, b])
    # rotate capture
    Y = imrotate(Y, calib['angle'], reshape=False)
    # crop usable sensor measurements
    csize = calib['cSize']
    start_row = int((Y.shape[0] - csize[0])/2)
    end_row = int(start_row + csize[0]) # omit -1 because Python indexing does not include end index
    start_col = int((Y.shape[1] - csize[1])/2)
    end_col = int(start_col + csize[1])
    Y = Y[start_row:end_row, start_col:end_col, :]
    return Y


def make_separable( Y ):
    rowMeans = Y.mean(axis=1, keepdims=True)
    colMeans = Y.mean(axis=0, keepdims=True)
    allMean = rowMeans.mean()
    Ysep = Y - rowMeans - colMeans + allMean
    return Ysep


def bayer2rgb( X_bayer, normalize=True ):
    X_rgb = np.empty(X_bayer.shape[:-1] + (3,))  # here, + means append to the tuple
    X_rgb[:, :, 0] = X_bayer[:, :, 0]
    X_rgb[:, :, 1] = 0.5 * (X_bayer[:, :, 1] + X_bayer[:, :, 2])
    X_rgb[:, :, 2] = X_bayer[:, :, 3]
    # normalize to be from 0 to 1
    if normalize:
        X_rgb = (X_rgb - X_rgb.min()) / (X_rgb.max() - X_rgb.min())
    return X_rgb


def fcrecon( cap, calib, lmbd):
    # check if SVDs have been taken
    if not 'UL_all' in calib:
        obtain_calib_svd(calib)
    Y = fc2bayer( cap, calib )  # convert RAW output to Bayer color channels
    Y = make_separable(Y) # let rows and columns have 0-mean
    X_bayer = np.empty([calib['VL_all'].shape[0], calib['VR_all'].shape[0], 4])
    for c in range(4):
        UL = calib['UL_all'][:, :, c]
        DL = calib['DL_all'][:, :, c]
        VL = calib['VL_all'][:, :, c]
        singLsq = np.square(calib['singL_all'][:, c])
        UR = calib['UR_all'][:, :, c]
        DR = calib['DR_all'][:, :, c]
        VR = calib['VR_all'][:, :, c]
        singRsq = np.square(calib['singR_all'][:, c])
        Yc = Y[:, :, c]
        inner = multi_dot([DL.T,UL.T,Yc,UR,DR]) / (np.outer(singLsq, singRsq) + np.full(X_bayer.shape[0:1], lmbd))
        X_bayer[:, :, c] = multi_dot([VL, inner, VR.T])
    X_bayer = X_bayer.clip(min=0)  # non-negative constraint: set all negative values to 0
    return bayer2rgb(X_bayer, True)  # bring back to RGB and normalize

