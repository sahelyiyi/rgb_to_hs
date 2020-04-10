import os
from scipy import misc
import cv2
import scipy.io
import numpy as np


def generate_mat_file(folder_path):
    file_names = os.listdir(folder_path)
    for file_name in file_names:
        if '.bmp' in file_name:
            image = misc.imread(os.path.join(folder_path, file_name), flatten=0)
            image_name = file_name.split('_')[0]

    aggregate_spectrum = np.empty((image.shape[0], image.shape[1], 31))
    for file_name in file_names:
        if '.png' in file_name:
            spectrum_num = int(file_name.split('.')[0].split('_')[-1]) - 1
            aggregate_spectrum[:, :, spectrum_num] = cv2.imread(os.path.join(folder_path, file_name), 0)

    scipy.io.savemat(os.path.join(folder_path, image_name+'.mat'), {'reflectances': aggregate_spectrum})
