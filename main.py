import os
import numpy as np

from sklearn.metrics import mean_squared_error
from math import sqrt
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_diff import delta_e_cie2000

from config import DATASETS_DIR, PATCHES_NUM, PATCHES_SIZE, TRAIN_RATIO
from constants import D65, X_BAR, Y_BAR, Z_BAR
from data.data_loader.simulate_data import load_data, random_patches, simulate_spectrophotometer
from models.regression import Regression
from utils import sp2xyz, xyz2lab


def get_rmse(y, predictions):
    return sqrt(mean_squared_error(y, predictions))


def get_delta_e_2000(y, predictions):
    xbar = np.array(X_BAR[2:33])
    ybar = np.array(Y_BAR[2:33])
    zbar = np.array(Z_BAR[2:33])
    lightsource = np.array(D65[2:33])

    ref_white_refl = np.ones((1, 31))
    ref_white_xyz = sp2xyz(ref_white_refl, lightsource, xbar, ybar, zbar)

    y_xyz = sp2xyz(y, lightsource, xbar, ybar, zbar)
    pred_xyz = sp2xyz(predictions, lightsource, xbar, ybar, zbar)

    y_lab = xyz2lab(y_xyz, ref_white_xyz)
    pred_lab = xyz2lab(pred_xyz, ref_white_xyz)

    delta_e_2000s = []
    for i in range(y_lab.shape[0]):
        color1_lab = LabColor(*y_lab[i])
        color2_lab = LabColor(*pred_lab[i])
        delta_e = delta_e_cie2000(color1_lab, color2_lab)
        delta_e_2000s.append(delta_e)

    return np.mean(delta_e_2000s)


def run(patches_num=PATCHES_NUM, method='rgb'):
    folder_path = os.path.join(DATASETS_DIR, 'CAVE', 'balloons_ms')

    rgb_img, hs_img = load_data(folder_path, method)
    rgb_patches, hs_patches, patches = random_patches(rgb_img, hs_img, patches_num, PATCHES_SIZE)

    avg_rgb_patches = simulate_spectrophotometer(rgb_patches)
    avg_hs_patches = simulate_spectrophotometer(hs_patches)

    train_samples = int(TRAIN_RATIO * patches_num)
    train_rgb, test_rgb = avg_rgb_patches[:train_samples], avg_rgb_patches[train_samples:]
    train_hs, test_hs = avg_hs_patches[:train_samples], avg_hs_patches[train_samples:]

    regresstion = Regression(train_rgb, train_hs)
    regresstion.train()
    predictions = regresstion.model.predict(test_rgb)
    rmse = get_rmse(test_hs, predictions)
    delta_e = get_delta_e_2000(test_hs, predictions)
    return rmse, delta_e


if __name__ == "__main__":
    run()
