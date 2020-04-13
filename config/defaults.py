import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS_DIR = os.path.join(BASE_DIR, 'data', 'datasets')

PATCHES_NUM = 200
PATCHES_SIZE = 10

TRAIN_RATIO = 0.8
