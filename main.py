import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from math import sqrt
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_diff import delta_e_cie2000

from config import DATASETS_DIR, PATCHES_NUM, PATCHES_SIZE, TRAIN_RATIO
from constants import D65, X_BAR, Y_BAR, Z_BAR
from data.data_loader.simulate_data import load_data, random_patches, simulate_spectrophotometer
from models.regression import Regression
from change_color_space import sp2xyz, xyz2lab, hs_to_xyz, hs_to_rgb


def get_rmse(y, predictions):
    return sqrt(mean_squared_error(y, predictions))


def get_delta_e_2000(y, predictions):
    # xbar = np.array(X_BAR[2:33])
    # ybar = np.array(Y_BAR[2:33])
    # zbar = np.array(Z_BAR[2:33])
    # lightsource = np.array(D65[2:33])
    #
    # ref_white_refl = np.ones((1, 31))
    # ref_white_xyz = sp2xyz(ref_white_refl, lightsource, xbar, ybar, zbar)
    #
    # y_xyz = sp2xyz(y, lightsource, xbar, ybar, zbar, normalize)
    # pred_xyz = sp2xyz(predictions, lightsource, xbar, ybar, zbar, normalize)

    if len(y.shape) == 2:
        y = y.reshape((1, y.shape[0], y.shape[1]))
        predictions = predictions.reshape((1, predictions.shape[0], predictions.shape[1]))

    y_xyz = hs_to_xyz(y)
    pred_xyz = hs_to_xyz(predictions)

    if len(y_xyz.shape) == 3:
        y_xyz = y_xyz.reshape((y_xyz.shape[1], y_xyz.shape[2]))
        pred_xyz = pred_xyz.reshape((pred_xyz.shape[1], pred_xyz.shape[2]))

    ref_white_xyz = None
    y_lab = xyz2lab(y_xyz, ref_white_xyz)
    pred_lab = xyz2lab(pred_xyz, ref_white_xyz)

    delta_e_2000s = []
    for i in range(y_lab.shape[0]):
        color1_lab = LabColor(*y_lab[i])
        color2_lab = LabColor(*pred_lab[i])
        delta_e = delta_e_cie2000(color1_lab, color2_lab)
        delta_e_2000s.append(delta_e)

    return np.mean(delta_e_2000s)


def get_rgbs(hs_samples):
    rgbs = []
    for i in range(hs_samples.shape[0]):
        vec = hs_samples[i]
        sub_img = np.empty((20, 20, vec.shape[0]))
        sub_img[:, :] = vec

        rgbs.append(hs_to_rgb(sub_img))
    return rgbs


def visualize_results(y, predictions):
    y_rgbs = get_rgbs(y)
    y_rgbs = np.concatenate(y_rgbs, axis=1)
    pred_rgbs = get_rgbs(predictions)
    pred_rgbs = np.concatenate(pred_rgbs, axis=1)
    black_pad = np.ones((5, y_rgbs.shape[1], 3))
    return np.concatenate((y_rgbs, black_pad, pred_rgbs), axis=0)


def run(patches_num=PATCHES_NUM,  method='rgb', patches_size=PATCHES_SIZE, visualize=False,
        folder_path=os.path.join(DATASETS_DIR, 'CAVE', 'balloons_ms')):

    ls_img, hs_img = load_data(folder_path, method)
    ls_patches, hs_patches, patches = random_patches(ls_img, hs_img, patches_num, patches_size)

    avg_ls_patches = simulate_spectrophotometer(ls_patches)
    avg_hs_patches = simulate_spectrophotometer(hs_patches)

    train_samples = int(TRAIN_RATIO * patches_num)
    train_ls, test_ls = avg_ls_patches[:train_samples], avg_ls_patches[train_samples:]
    train_hs, test_hs = avg_hs_patches[:train_samples], avg_hs_patches[train_samples:]

    regresstion = Regression(train_ls, train_hs)
    regresstion.train()
    predictions = regresstion.model.predict(test_ls)
    rmse = get_rmse(test_hs, predictions)
    delta_e = get_delta_e_2000(test_hs, predictions)

    if visualize:
        result = visualize_results(test_hs, predictions)
        plt.imshow(result)

    return rmse, delta_e


if __name__ == "__main__":
    run()
