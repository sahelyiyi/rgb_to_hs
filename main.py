import os

from sklearn.metrics import mean_squared_error
from math import sqrt

from config import DATASETS_DIR, PATCHES_NUM, PATCHES_SIZE, TRAIN_RATIO
from data.data_loader.simulate_data import load_data, random_patches, simulate_spectrophotometer
from models.regression import Regression


def rmse(y, predictions):
    return sqrt(mean_squared_error(y, predictions))


def run(patches_num=PATCHES_NUM):
    folder_path = os.path.join(DATASETS_DIR, 'CAVE', 'balloons_ms')

    rgb_img, hs_img = load_data(folder_path)
    rgb_patches, hs_patches, patches = random_patches(rgb_img, hs_img, patches_num, PATCHES_SIZE)

    avg_rgb_patches = simulate_spectrophotometer(rgb_patches)
    avg_hs_patches = simulate_spectrophotometer(hs_patches)

    train_samples = int(TRAIN_RATIO * patches_num)
    train_rgb, test_rgb = avg_rgb_patches[:train_samples], avg_rgb_patches[train_samples:]
    train_hs, test_hs = avg_hs_patches[:train_samples], avg_hs_patches[train_samples:]

    regresstion = Regression(train_rgb, train_hs)
    regresstion.train()
    predictions = regresstion.model.predict(test_rgb)
    return rmse(test_hs, predictions)


if __name__ == "__main__":
    run()
