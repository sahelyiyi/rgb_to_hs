import scipy.io
import numpy as np

from change_color_space import XYZ2sRGB_exgamma

reflectances = scipy.io.loadmat('data/datasets/scene4_sample/ref4_scene4.mat')['reflectances']
reflectances = reflectances/np.max(reflectances)

illum_6500 = scipy.io.loadmat('data/datasets/scene4_sample/illum_6500.mat')['illum_6500']

radiances_6500 = np.zeros((reflectances.shape))  # initialize array
for i in range(33):
  radiances_6500[:, :, i] = reflectances[:, :, i] * illum_6500[i]

radiances = radiances_6500
r, c, w = radiances.shape
radiances = np.reshape(radiances, (r*c, w))

xyzbar = scipy.io.loadmat('data/datasets/scene4_sample/xyzbar.mat')['xyzbar']
XYZ = np.dot(xyzbar.transpose(), radiances.transpose()).transpose()

XYZ = np.reshape(XYZ, (r, c, 3))
XYZ[XYZ<0] = 0
XYZ = XYZ/np.max(XYZ)

RGB = XYZ2sRGB_exgamma(XYZ)
RGB[RGB < 0] = 0
RGB[RGB > 1] = 1

z = np.max(RGB[244, 17, :])
RGB[RGB > z] = z
RGB_clip = RGB / z
RGB_clip = np.power(RGB_clip, 0.4)
