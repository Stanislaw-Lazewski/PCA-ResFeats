import os
from types import SimpleNamespace

import numpy as np
from torchvision import transforms


RANDOM_STATE = 97
N_SPLITS = 5
CONFIGS_ROOT = "configs"
CBIR_RESULTS_ROOT = "cbir_results"
CBIR_ROOT = "cbir"
MODELS_ROOT = "models"
MODELS_CONFIGS_ROOT = "models_configs"
DATASETS_ROOT = "datasets"
RAW_FEATURES_ROOT = "raw_features"
FEATURES_ROOT = "features"
DATASET_INFO_ROOT = "dataset_info"
RESULTS_ROOT = "results"
SUMMARY_CBIR_RESULTS_ROOT = "summary_cbir_results"
SUMMARY_RESULTS_ROOT = "summary_results"
CBIR_PLOTS_ROOT = "cbir_plots"

CONFIG_EXT = ".pkl"
MODEL_CONFIG_EXT = ".json"

DATASETS_NAMES = [
    'Caltech-101',
    'Caltech-256',
    'Oxford Flowers',
    'MLC2008',
    'Dogs vs. Cats',
    'EuroSAT',
    'Kather_5000',  # https://zenodo.org/records/53169
    'BreaKHis'  # https://www.kaggle.com/datasets/ambarish/breakhis?resource=download
]
OBJECT_DATASETS_NAMES = [
    'Caltech-101',
    'Caltech-256',
    'Oxford Flowers',
    # 'MLC2008',
    'Dogs vs. Cats',
    'EuroSAT',
]
HISTO_DATASET_NAMES = [
    'Kather_5000',
    'BreaKHis'
]

ALL_HISTO_DATASET_NAMES = [
    'Kather_5000',
    'BreaKHis_C2',
    'BreaKHis_40X_C2',
    'BreaKHis_100X_C2',
    'BreaKHis_200X_C2',
    'BreaKHis_400X_C2',
    'BreaKHis_C8',
    'BreaKHis_40X_C8',
    'BreaKHis_100X_C8',
    'BreaKHis_200X_C8',
    'BreaKHis_400X_C8',
]

DATASETS_MIDDLE_DIRS = {
    # 'Kather_5000': os.path.join("Kather_texture_2016_image_tiles_5000", "Kather_texture_2016_image_tiles_5000"),
    'BreaKHis': os.path.join("BreaKHis_v1", "histology_slides", "breast"),
    }
SPLIT_STRATEGY = {
    'Caltech-101': "cross-validation",
    'Caltech-256': "cross-validation",
    'Oxford Flowers': "cross-validation",
    'MLC2008': "cross-validation",
    'Dogs vs. Cats': "cross-validation",
    'EuroSAT': "cross-validation",
    'Kather_5000': "cross-validation",
    # 'Kather': "cross-validation",
    'BreaKHis': "predetermined_split",
    }
BREAKHIS_MAGS = ["40X", "100X", "200X", "400X"]

BREAKHIS_CLASS_SUBDIRS = [
    ("malignant", "SOB", "ductal_carcinoma"),
    ("malignant", "SOB", "lobular_carcinoma"),
    ("malignant", "SOB", "mucinous_carcinoma"),
    ("malignant", "SOB", "papillary_carcinoma"),
    ("benign", "SOB", "adenosis"),
    ("benign", "SOB", "fibroadenoma"),
    ("benign", "SOB", "phyllodes_tumor"),
    ("benign", "SOB", "tubular_adenoma")]

PREPROCESS_DICT = {
    "0": transforms.ToTensor(),
    "1": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            ]),
    "2": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
            ])
    }

TRAIN_PREPROCESS_DICT = {
    "0": transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ]),
    "1": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ]),
    "2": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
            ])
    }


def parse(d):
    x = SimpleNamespace()
    _ = [setattr(x, k, parse(v)) if isinstance(v, dict) else setattr(x, k, v) for k, v in d.items()]
    return x


def beautify_config_str(config_str):
    # TODO: improve it
    return config_str.replace("<class ", "").replace(">", "")


def get_shape_from_file(shape_txt_path):
    with open(shape_txt_path) as f:
        lines = f.readlines()
        return eval(lines[0].strip())


def minimal_numpy_int_type(np_array):
    type_list = [np.int8, np.int16, np.int32, np.int64]
    for type_ in type_list:
        if np.array_equal(np_array, np_array.astype(type_)):
            return type_
    msg = "Array not in int type."
    raise ValueError(msg)
