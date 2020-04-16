import numpy as np


def sp2xyz(s, lightsource, xbar, ybar, zbar, normalize=False):
    #  The input spectrum s can be a vector or a n x 36 spectra matrix.
    #  With sa n x 36 input matrix, a n x 3 output matrix is produced with one XYZ
    k = 100 / (np.transpose(lightsource*ybar))
    # Element wise multiplication of each row by the illuminant
    Vs = s * np.kron(np.ones((s.shape[0], 1)), lightsource)
    # Producing final XYZ value by dot products between _bar predefined
    # values & previosly computed Vs
    XYZ = np.dot(k*Vs, np.column_stack((xbar, ybar, zbar)))

    if normalize:
        XYZ[XYZ < 0] = 0
        XYZ = XYZ / np.max(XYZ)

    return XYZ


def xyz2lab(XYZ, refWhite):
    # XYZ may be a n x 3 matrix containg in each row one XYZ value.
    # In that case n x 3 matrix will be returned containing
    # one Lab value for each XYZ value
    X = XYZ[:, 0]
    Y = XYZ[:, 1]
    Z = XYZ[:, 2]

    # element wise division
    xxn = np.divide(X, refWhite[:, 0])
    yyn = np.divide(Y, refWhite[:, 1])
    zzn = np.divide(Z, refWhite[:, 2])

    # See Lab_2_XYZ.m for explanation
    i1 = np.where(xxn > 0.008856)
    i1y = np.count_nonzero(xxn > 0.008856)
    i2 = np.where(xxn <= 0.008856)
    i2y = np.count_nonzero(xxn <= 0.008856)
    fxxn = np.ones((i1y + i2y, 1))
    if i1y > 0:
        fxxn1 = np.power(xxn[i1], 1.0/3)
        fxxn1 = fxxn1.reshape(fxxn1.shape[0], 1)
        fxxn[i1] = fxxn1
    if i2y > 0:
        fxxn2 = 7.787 * xxn[i2] + 16.0/116
        fxxn2 = fxxn2.reshape(fxxn2.shape[0], 1)
        fxxn[i2] = fxxn2

    i1 = np.where(yyn > 0.008856)
    i1y = np.count_nonzero(yyn > 0.008856)
    i2 = np.where(yyn <= 0.008856)
    i2y = np.count_nonzero(yyn <= 0.008856)
    fyyn = np.ones((i1y + i2y, 1))
    L = np.ones((i1y + i2y, 1))
    if i1y > 0:
        fyyn1 = np.power(yyn[i1], 1.0/3)
        fyyn1 = fyyn1.reshape(fyyn1.shape[0], 1)
        L1 = 116 * fyyn1 - 16
        L[i1] = L1
        fyyn[i1] = fyyn1
    if i2y > 0:
        fyyn2 = 7.787 * yyn[i2] + 16.0/116
        fyyn2 = fyyn2.reshape(fyyn2.shape[0], 1)
        L2 = 903.3 * yyn[i2]
        L2 = fyyn2.reshape(L2.shape[0], 1)
        L[i2] = L2
        fyyn[i2] = fyyn2

    i1 = np.where(zzn > 0.008856)
    i1y = np.count_nonzero(zzn > 0.008856)
    i2 = np.where(zzn <= 0.008856)
    i2y = np.count_nonzero(zzn <= 0.008856)
    fzzn = np.ones((i1y + i2y, 1))
    if i1y > 0:
        fzzn1 = np.power(zzn[i1], 1.0/3)
        fzzn1 = fzzn1.reshape(fzzn1.shape[0], 1)
        fzzn[i1] = fzzn1
    if i2y > 0:
        fzzn2 = 7.787 * zzn[i2] + 16.0/116
        fzzn2 = fzzn2.reshape(fzzn2.shape[0], 1)
        fzzn[i2] = fzzn2

    a = 500 * (fxxn - fyyn)
    b = 200 * (fyyn - fzzn)

    Lab = np.column_stack((L, a, b))
    return Lab


def XYZ2sRGB_exgamma(XYZ):
    # See IEC_61966-2-1.pdf
    # No gamma correction has been incorporated here, nor any clipping, so this
    # transformation remains strictly linear.  Nor is there any data-checking.
    # DHF 9-Feb-11

    # Image dimensions
    d = XYZ.shape
    r = np.prod(d[0:-1])   # product of sizes of all dimensions except last, wavelength
    w = d[-1]             # size of last dimension, wavelength

    # Reshape for calculation, converting to w columns with r rows.
    XYZ = np.reshape(XYZ, (r, w))

    # Forward transformation from 1931 CIE XYZ values to sRGB values (Eqn 6 in
    # IEC_61966-2-1.pdf).
    M = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689, 1.8758, 0.0414],
                  [0.0557, -0.2040, 1.0570]])
    sRGB = np.dot(M, XYZ.transpose()).transpose()

    # Reshape to recover shape of original input.
    sRGB = np.reshape(sRGB, d)

    return sRGB
