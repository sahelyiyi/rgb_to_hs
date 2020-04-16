import os
import random
import scipy.io
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import color

from change_color_space import hs_to_rgb, XYZ2sRGB_exgamma


def create_rgb_from_hs(hs_file_path):
    rgb_file_path = '%s_D65.jpg' % hs_file_path.split('.')[0]
    if not os.path.exists(rgb_file_path):
        reflectances = scipy.io.loadmat(hs_file_path)['reflectances']
        reflectances = reflectances[:, :, :31]

        RGB_clip = hs_to_rgb(reflectances)

        plt.imsave(rgb_file_path, RGB_clip)

    rgb_image = cv2.imread(rgb_file_path)
    return rgb_image


def load_data(folder_path, method='rgb'):
    low_channel_img, hs_img = None, None

    file_names = os.listdir(folder_path)
    for file_name in file_names:
        if '.mat' in file_name:
            hs_file_path = os.path.join(folder_path, file_name)

            rgb_img = create_rgb_from_hs(hs_file_path)
            if method == 'lab':
                low_channel_img = color.rgb2lab(rgb_img)
            elif method == 'xyz':
                low_channel_img = color.rgb2xyz(rgb_img)
            else:
                low_channel_img = rgb_img

            data = scipy.io.loadmat(hs_file_path)
            if 'reflectances' in data:
                hs_img = data['reflectances']
                hs_img = hs_img[:, :, :31]
                hs_img = hs_img / np.max(hs_img)

    return low_channel_img, hs_img


def random_patches(rgb_img, hs_img, patches_num, patches_size):
    h, w = rgb_img.shape[0], rgb_img.shape[1]
    patches = []
    rgb_patches = []
    hs_patches = []
    while len(patches) < patches_num:
        patch_x = random.randint(0, h)
        patch_y = random.randint(0, w)
        if patch_x + patches_size < h and patch_y + patches_size < w:
            patches.append((patch_x, patch_y))
            rgb_patches.append(np.array(rgb_img[patch_y: patch_y + patches_size, patch_x: patch_x + patches_size]))
            hs_patches.append(np.array(hs_img[patch_y: patch_y + patches_size, patch_x: patch_x + patches_size]))

    return np.array(rgb_patches), np.array(hs_patches), np.array(patches)


def simulate_spectrophotometer(img_patches):
    average_patches = []
    for i in range(img_patches.shape[0]):
        average_patch = []
        for d in range(img_patches.shape[-1]):
            average_patch.append(np.mean(img_patches[i, :, :, d]))
        average_patches.append(average_patch)
    return np.array(average_patches)
