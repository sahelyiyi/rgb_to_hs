import os
from scipy import misc
import random
import scipy.io
import numpy as np


def load_data(folder_path):
    file_names = os.listdir(folder_path)
    for file_name in file_names:
        if '.bmp' in file_name:
            rgb_img = misc.imread(os.path.join(folder_path, file_name), flatten=0)
        if '.mat' in file_name:
            data = scipy.io.loadmat(os.path.join(folder_path, file_name))
            if 'reflectances' in data:
                hs_img = data['reflectances']
    return rgb_img, hs_img


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
            rgb_patches.append(rgb_img[patch_y: patch_y + patches_size, patch_x: patch_x + patches_size])
            hs_patches.append(hs_img[patch_y: patch_y + patches_size, patch_x: patch_x + patches_size])

    return np.array(rgb_patches), np.array(hs_patches), np.array(patches)


def simulate_spectrophotometer(img_patches):
    average_patches = []
    for i in range(img_patches.shape[0]):
        average_patch = []
        for d in range(img_patches.shape[-1]):
            average_patch.append(np.sum(img_patches[i, :, :, d]))
        average_patches.append(average_patch)
    return np.array(average_patches)
