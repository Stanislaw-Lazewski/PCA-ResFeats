import os
import pickle
import shutil
import time
from typing import Dict
import warnings

import csv
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from tqdm import tqdm

from classification import classify_with_raport
from datasets_preparation import DatasetInfo, get_strategy
from features_extraction import get_model_name
from utils import (
    CONFIG_EXT,
    CONFIGS_ROOT,
    DATASET_INFO_ROOT,
    FEATURES_ROOT,
    RAW_FEATURES_ROOT,
    get_shape_from_file
)


def get_unique_config_id():
    """CONFIGS_ROOT contains files named in convention: id.pkl (0.pkl, 1.pkl, ...)"""
    configs_list = os.listdir(CONFIGS_ROOT)
    if len(configs_list) == 0:
        return "0"
    else:
        return str(sorted([int(os.path.splitext(config)[0]) for config in configs_list])[-1] + 1)


def find_config_id(config: Dict):
    config_id = None
    for c_name in os.listdir(CONFIGS_ROOT):
        c = pickle.load(open(os.path.join(CONFIGS_ROOT, c_name), 'rb'))
        if c == config:
            if config_id is None:
                config_id = os.path.splitext(c_name)[0]
            else:
                msg = f"Configs duplicate: {config_id}{CONFIG_EXT} and {c_name} are the same."
                raise ValueError(msg)
    return config_id


def build_network_output_config(backbone_model, raw_features_out_name, preprocess_id, npy_type):
    config = {
        "features_type": "features",
        "input": [
            {
                "features_type": "raw_features",
                "params": {
                    "backbone_model": vars(backbone_model),
                    "layer": raw_features_out_name,
                    "preprocess_id": preprocess_id
                    }
                }
            ],
        "params": None,
        "transformation": "network-output",
        "type": npy_type
        }
    return config


def build_max_pooling_config(backbone_model, raw_features_name, preprocess_id, dim, npy_type):
    config = {
        "features_type": "features",
        "input": [
            {
                "features_type": "raw_features",
                "params": {
                    "backbone_model": vars(backbone_model),
                    "layer": raw_features_name,
                    "preprocess_id": preprocess_id
                    }
                }
            ],
        "params": {
            "dim": dim
            },
        "transformation": "max-pooling",
        "type": npy_type
        }
    return config


def build_concatenation_config(config_list, npy_type):
    config = {
        'features_type': 'features',
        'input': config_list,
        'params': None,
        'transformation': 'concatenation',
        'type': npy_type
        }
    return config


def build_pca_config(component_features_config, n_components, svd_solver, npy_type):
    config = {
        'features_type': 'features',
        'input': [
            component_features_config
            ],
        'params': {
            "n_components": n_components,
            "svd_solver": svd_solver
            },
        'transformation': 'pca',
        'type': npy_type
        }
    return config


def build_nca_config(component_features_config, n_components, init, random_state, npy_type):
    config = {
        'features_type': 'features',
        'input': [
            component_features_config
            ],
        'params': {
            "n_components": n_components,
            "init": init,
            "random_state": random_state
            },
        'transformation': 'nca',
        'type': npy_type
        }
    return config


def get_all_config_idx(last_transformation: str):
    config_idx = []
    for c_name in os.listdir(CONFIGS_ROOT):
        config = pickle.load(open(os.path.join(CONFIGS_ROOT, c_name), 'rb'))
        if config["transformation"] == last_transformation:
            config_idx.append(os.path.splitext(c_name)[0])
    return config_idx


def purge_config(config_id: str):
    print("Removing following file and directories:")
    path = os.path.join(CONFIGS_ROOT, f"{config_id}{CONFIG_EXT}")
    if os.path.exists(path):
        os.remove(path)
        print(f"Config file {path} removed.")
    for dataset_name in os.listdir(FEATURES_ROOT):
        strategy = get_strategy(dataset_name)
        strategy_dir = os.path.join(FEATURES_ROOT, dataset_name, strategy)
        for fold in os.listdir(strategy_dir):
            features_id_dir = os.path.join(strategy_dir, fold, config_id)
            if os.path.exists(features_id_dir):
                shutil.rmtree(features_id_dir, ignore_errors=False)
                print(f"Config directory {features_id_dir} removed.")


def raw_features_out(dataset_name, preprocess_id, backbone_model, raw_features_out_name, npy_type=np.float16):
    model_name = get_model_name(backbone_model,
                                with_ext=False)
    raw_features_dir = os.path.join(RAW_FEATURES_ROOT, dataset_name, f"preprocess_{preprocess_id}", model_name, raw_features_out_name)
    strategy = get_strategy(dataset_name)
    dataset_info_dir = os.path.join(DATASET_INFO_ROOT, dataset_name, strategy)

    config = build_network_output_config(
        backbone_model=backbone_model,
        raw_features_out_name=raw_features_out_name,
        preprocess_id=preprocess_id,
        npy_type=npy_type
        )

    if not os.path.exists(CONFIGS_ROOT):
        os.makedirs(CONFIGS_ROOT)

    features_id = find_config_id(config)
    if features_id is None:
        features_id = get_unique_config_id()
        with open(os.path.join(CONFIGS_ROOT, f"{features_id}{CONFIG_EXT}"), 'wb') as pickle_file:
            pickle.dump(config, pickle_file)
    for fold in tqdm(os.listdir(dataset_info_dir)):
        features_dir = os.path.join(FEATURES_ROOT, dataset_name, strategy, fold, features_id)
        if os.path.exists(features_dir):
            msg = f"Features directory {features_dir} already exists."
            raise FileExistsError(msg)
        fold_dir = os.path.join(dataset_info_dir, fold)
        X = dict()
        y = dict()
        t = dict()
        for train_or_test in ["train", "test"]:
            with open(os.path.join(fold_dir, f'{train_or_test}_idx.npy'), 'rb') as npy_file:
                train_or_test_idx = np.load(npy_file)
            dataset_info = pickle.load(open(os.path.join(fold_dir, f"{train_or_test}_dataset_info.pkl"), 'rb'))
            t[train_or_test] = dict()
            start_time = time.time()
            X[train_or_test] = []
            for idx in train_or_test_idx:
                x = torch.load(os.path.join(raw_features_dir, f"{idx}.pt"))
                X[train_or_test].append(torch.squeeze(x))
            X[train_or_test] = torch.stack(X[train_or_test]).cpu().numpy().astype(npy_type)
            end_time = time.time()
            t[train_or_test]["squeezing"] = end_time - start_time
            y[train_or_test] = dataset_info.encoded_labels
            os.makedirs(features_dir, exist_ok=True)
            with open(os.path.join(features_dir, f'X_{train_or_test}.npy'), 'wb') as npy_file:
                np.save(npy_file, X[train_or_test])
            with open(os.path.join(features_dir, f'y_{train_or_test}.npy'), 'wb') as npy_file:
                np.save(npy_file, y[train_or_test])
            with open(os.path.join(features_dir, f"shape_{train_or_test}.txt"), 'w') as fd:
                fd.write(f'{X[train_or_test].shape}\n')
        start_time = time.time()
        sc = StandardScaler()
        sc.fit(X["train"])
        end_time = time.time()
        t["train"]["standard_scaler_fit"] = end_time - start_time
        start_time = time.time()
        X["train"] = sc.transform(X["train"]).astype(npy_type)
        end_time = time.time()
        t["train"]["standard_scaler_transform"] = end_time - start_time
        start_time = time.time()
        X["test"] = sc.transform(X["test"]).astype(npy_type)
        end_time = time.time()
        t["test"]["standard_scaler_transform"] = end_time - start_time
        with open(os.path.join(features_dir, 'standard_scaler.sav'), 'wb') as pickle_file:
            pickle.dump(sc, pickle_file)
        with open(os.path.join(features_dir, f"config{CONFIG_EXT}"), 'wb') as pickle_file:
            pickle.dump(config, pickle_file)

        t = classify_with_raport(X["train"], y["train"], X["test"], y["test"], features_dir, dataset_info.idx_to_class, t)
        for train_or_test in ["train", "test"]:
            with open(os.path.join(features_dir, f'time_{train_or_test}.csv'), 'w') as f:
                w = csv.DictWriter(f, t[train_or_test].keys())
                w.writeheader()
                w.writerow(t[train_or_test])
    return features_id


def raw_features_amax(dataset_name, preprocess_id, backbone_model, raw_features_name, dim=(0, 2, 3), npy_type=np.float16):
    model_name = get_model_name(backbone_model,
                                with_ext=False)
    raw_features_dir = os.path.join(RAW_FEATURES_ROOT, dataset_name, f"preprocess_{preprocess_id}", model_name, raw_features_name)
    strategy = get_strategy(dataset_name)
    dataset_info_dir = os.path.join(DATASET_INFO_ROOT, dataset_name, strategy)

    config = build_max_pooling_config(backbone_model=backbone_model,
                                      raw_features_name=raw_features_name,
                                      preprocess_id=preprocess_id,
                                      dim=dim,
                                      npy_type=npy_type)

    if not os.path.exists(CONFIGS_ROOT):
        os.makedirs(CONFIGS_ROOT)

    features_id = find_config_id(config)
    if features_id is None:
        features_id = get_unique_config_id()
        with open(os.path.join(CONFIGS_ROOT, f"{features_id}{CONFIG_EXT}"), 'wb') as pickle_file:
            pickle.dump(config, pickle_file)
    for fold in tqdm(os.listdir(dataset_info_dir)):
        features_dir = os.path.join(FEATURES_ROOT, dataset_name, strategy, fold, features_id)
        if os.path.exists(features_dir):
            msg = f"Features directory {features_dir} already exists."
            raise FileExistsError(msg)
        fold_dir = os.path.join(dataset_info_dir, fold)
        X = dict()
        y = dict()
        t = dict()
        for train_or_test in ["train", "test"]:
            with open(os.path.join(fold_dir, f'{train_or_test}_idx.npy'), 'rb') as npy_file:
                train_or_test_idx = np.load(npy_file)
            dataset_info = pickle.load(open(os.path.join(fold_dir, f"{train_or_test}_dataset_info.pkl"), 'rb'))
            t[train_or_test] = dict()
            start_time = time.time()
            X[train_or_test] = []
            for idx in train_or_test_idx:
                x = torch.load(os.path.join(raw_features_dir, f"{idx}.pt"))
                X[train_or_test].append(torch.amax(x, dim=dim))
            X[train_or_test] = torch.stack(X[train_or_test]).cpu().numpy().astype(npy_type)
            end_time = time.time()
            t[train_or_test]["max-pooling"] = end_time - start_time
            y[train_or_test] = dataset_info.encoded_labels
            os.makedirs(features_dir, exist_ok=True)
            with open(os.path.join(features_dir, f'X_{train_or_test}.npy'), 'wb') as npy_file:
                np.save(npy_file, X[train_or_test])
            with open(os.path.join(features_dir, f'y_{train_or_test}.npy'), 'wb') as npy_file:
                np.save(npy_file, y[train_or_test])
            with open(os.path.join(features_dir, f"shape_{train_or_test}.txt"), 'w') as fd:
                fd.write(f'{X[train_or_test].shape}\n')
        start_time = time.time()
        sc = StandardScaler()
        sc.fit(X["train"])
        end_time = time.time()
        t["train"]["standard_scaler_fit"] = end_time - start_time
        start_time = time.time()
        X["train"] = sc.transform(X["train"]).astype(npy_type)
        end_time = time.time()
        t["train"]["standard_scaler_transform"] = end_time - start_time
        start_time = time.time()
        X["test"] = sc.transform(X["test"]).astype(npy_type)
        end_time = time.time()
        t["test"]["standard_scaler_transform"] = end_time - start_time
        with open(os.path.join(features_dir, 'standard_scaler.sav'), 'wb') as pickle_file:
            pickle.dump(sc, pickle_file)
        with open(os.path.join(features_dir, f"config{CONFIG_EXT}"), 'wb') as pickle_file:
            pickle.dump(config, pickle_file)

        t = classify_with_raport(X["train"], y["train"], X["test"], y["test"], features_dir, dataset_info.idx_to_class, t)
        for train_or_test in ["train", "test"]:
            with open(os.path.join(features_dir, f'time_{train_or_test}.csv'), 'w') as f:
                w = csv.DictWriter(f, t[train_or_test].keys())
                w.writeheader()
                w.writerow(t[train_or_test])
    return features_id


def features_concatenation(dataset_name, features_id_list, npy_type=np.float16):
    strategy = get_strategy(dataset_name)
    dataset_info_dir = os.path.join(DATASET_INFO_ROOT, dataset_name, strategy)
    config_list = list(map(lambda features_id: pickle.load(
            open(os.path.join(CONFIGS_ROOT, f"{features_id}{CONFIG_EXT}"), 'rb')),
        features_id_list))

    config = build_concatenation_config(config_list=config_list,
                                        npy_type=npy_type)

    features_id = find_config_id(config)
    if features_id is None:
        features_id = get_unique_config_id()
        with open(os.path.join(CONFIGS_ROOT, f"{features_id}{CONFIG_EXT}"), 'wb') as pickle_file:
            pickle.dump(config, pickle_file)

    for fold in tqdm(os.listdir(dataset_info_dir)):
        features_dir = os.path.join(FEATURES_ROOT, dataset_name, strategy, fold, features_id)
        if os.path.exists(features_dir):
            msg = f"Features directory {features_dir} already exists."
            raise FileExistsError(msg)
        X = dict()
        y = dict()
        t = dict()
        for train_or_test in ["train", "test"]:
            X[train_or_test] = []
            y[train_or_test] = None
            for component_feature in features_id_list:
                component_feature_dir = os.path.join(FEATURES_ROOT, dataset_name, strategy, fold, component_feature)
                with open(os.path.join(component_feature_dir, f'X_{train_or_test}.npy'), 'rb') as npy_file:
                    X[train_or_test].append(np.load(npy_file))
                with open(os.path.join(component_feature_dir, f'y_{train_or_test}.npy'), 'rb') as npy_file:
                    y_train_or_test = np.load(npy_file)
                if y[train_or_test] is None:
                    y[train_or_test] = y_train_or_test
                elif not np.array_equal(y[train_or_test], y_train_or_test):
                    msg = f"Features with id {features_id_list} have different y_{train_or_test}.npy files."
                    raise ValueError(msg)
            dataset_info = pickle.load(open(os.path.join(dataset_info_dir, fold, f"{train_or_test}_dataset_info.pkl"), 'rb'))
            if not np.array_equal(y[train_or_test], dataset_info.encoded_labels):
                msg = f"Features with id {features_id_list} have inconsistent y_{train_or_test}.npy files with {train_or_test}_dataset_info.pkl."
                raise ValueError(msg)
            t[train_or_test] = dict()
            start_time = time.time()
            X[train_or_test] = np.concatenate(X[train_or_test], axis=1).astype(npy_type)
            end_time = time.time()
            t[train_or_test]["concatenation"] = end_time - start_time
            os.makedirs(features_dir, exist_ok=True)
            with open(os.path.join(features_dir, f'X_{train_or_test}.npy'), 'wb') as npy_file:
                np.save(npy_file, X[train_or_test])
            with open(os.path.join(features_dir, f'y_{train_or_test}.npy'), 'wb') as npy_file:
                np.save(npy_file, y[train_or_test])
            with open(os.path.join(features_dir, f"shape_{train_or_test}.txt"), 'w') as fd:
                fd.write(f'{X[train_or_test].shape}\n')
        start_time = time.time()
        sc = StandardScaler()
        sc.fit(X["train"])
        end_time = time.time()
        t["train"]["standard_scaler_fit"] = end_time - start_time
        start_time = time.time()
        X["train"] = sc.transform(X["train"]).astype(npy_type)
        end_time = time.time()
        t["train"]["standard_scaler_transform"] = end_time - start_time
        start_time = time.time()
        X["test"] = sc.transform(X["test"]).astype(npy_type)
        end_time = time.time()
        t["test"]["standard_scaler_transform"] = end_time - start_time
        with open(os.path.join(features_dir, 'standard_scaler.sav'), 'wb') as pickle_file:
            pickle.dump(sc, pickle_file)
        with open(os.path.join(features_dir, f"config{CONFIG_EXT}"), 'wb') as pickle_file:
            pickle.dump(config, pickle_file)

        t = classify_with_raport(X["train"], y["train"], X["test"], y["test"], features_dir, dataset_info.idx_to_class, t)
        for train_or_test in ["train", "test"]:
            with open(os.path.join(features_dir, f'time_{train_or_test}.csv'), 'w') as f:
                w = csv.DictWriter(f, t[train_or_test].keys())
                w.writeheader()
                w.writerow(t[train_or_test])
    return features_id


def features_pca_nca(dataset_name, component_feature, pca_or_nca, n_components, svd_solver=None, init=None, npy_type=np.float16):
    if svd_solver is None and pca_or_nca == "pca":
        msg = "PCA requires `svd_solver` parameter."
        raise TypeError(msg)
    if init is None and pca_or_nca == "nca":
        msg = "NCA requires `init` parameter."
        raise TypeError(msg)
    random_state = 97
    strategy = get_strategy(dataset_name)
    dataset_info_dir = os.path.join(DATASET_INFO_ROOT, dataset_name, strategy)
    component_features_config = pickle.load(open(os.path.join(CONFIGS_ROOT, f"{component_feature}{CONFIG_EXT}"), 'rb'))

    min_train_shape = n_components
    for fold in os.listdir(dataset_info_dir):
        component_feature_dir = os.path.join(FEATURES_ROOT, dataset_name, strategy, fold, component_feature)
        train_shape = get_shape_from_file(
            shape_txt_path=os.path.join(component_feature_dir, "shape_train.txt")
            )
        min_train_shape = min(min_train_shape, *train_shape)
    if n_components > min_train_shape:
        msg = (f"{pca_or_nca}: The preferred dimensionality of the projected space `n_components` ({n_components}) cannot "
               f"be greater than the given data dimensionality ({min_train_shape})! SKIPPED...")
        warnings.warn(msg)
        return None

    if pca_or_nca == "pca":
        config = build_pca_config(component_features_config=component_features_config,
                                  n_components=n_components,
                                  svd_solver=svd_solver,
                                  npy_type=npy_type)
    elif pca_or_nca == "nca":
        config = build_nca_config(component_features_config=component_features_config,
                                  n_components=n_components,
                                  init=init,
                                  random_state=random_state,
                                  npy_type=npy_type)
    else:
        msg = f"{pca_or_nca} transformation is not implemented yet"
        raise NotImplementedError(msg)

    features_id = find_config_id(config)
    if features_id is None:
        features_id = get_unique_config_id()
        with open(os.path.join(CONFIGS_ROOT, f"{features_id}{CONFIG_EXT}"), 'wb') as pickle_file:
            pickle.dump(config, pickle_file)

    for fold in tqdm(os.listdir(dataset_info_dir)):
        features_dir = os.path.join(FEATURES_ROOT, dataset_name, strategy, fold, features_id)
        if os.path.exists(features_dir):
            msg = f"Features directory {features_dir} already exists."
            raise FileExistsError(msg)
        X = dict()
        y = dict()
        t = dict()
        component_feature_dir = os.path.join(FEATURES_ROOT, dataset_name, strategy, fold, component_feature)
        for train_or_test in ["train", "test"]:
            with open(os.path.join(component_feature_dir, f'X_{train_or_test}.npy'), 'rb') as npy_file:
                X[train_or_test] = np.load(npy_file)
            with open(os.path.join(component_feature_dir, f'y_{train_or_test}.npy'), 'rb') as npy_file:
                y[train_or_test] = np.load(npy_file)
            dataset_info = pickle.load(open(os.path.join(dataset_info_dir, fold, f"{train_or_test}_dataset_info.pkl"), 'rb'))
            if not np.array_equal(y[train_or_test], dataset_info.encoded_labels):
                msg = f"Features with id {component_feature} have inconsistent y_{train_or_test}.npy files with {train_or_test}_dataset_info.pkl."
                raise ValueError(msg)
            t[train_or_test] = dict()
        start_time = time.time()
        sc = StandardScaler()
        sc.fit(X["train"])
        end_time = time.time()
        t["train"]["standard_scaler_fit"] = end_time - start_time
        start_time = time.time()
        X["train"] = sc.transform(X["train"]).astype(npy_type)
        end_time = time.time()
        t["train"]["standard_scaler_transform"] = end_time - start_time
        start_time = time.time()
        X["test"] = sc.transform(X["test"]).astype(npy_type)
        end_time = time.time()
        t["test"]["standard_scaler_transform"] = end_time - start_time

        component_feature_sc_dir = os.path.join(component_feature_dir, "standard_scaler.sav")
        if os.path.exists(component_feature_sc_dir):
            component_feature_sc = pickle.load(open(component_feature_sc_dir, 'rb'))
            if not (np.array_equal(sc.scale_, component_feature_sc.scale_)
                    and np.array_equal(sc.mean_, component_feature_sc.mean_)
                    and np.array_equal(sc.var_, component_feature_sc.var_)):
                msg = f"Component feature `{component_feature}` standard scaler is inconsistent with recomputed standard scaler."
                raise ValueError(msg)
        else:
            msg = f"Component feature `{component_feature}` has no saved standard scaler applying before {pca_or_nca}."
            warnings.warn(msg)
        if pca_or_nca == "pca":
            start_time = time.time()
            compressor = PCA(n_components=n_components, svd_solver=svd_solver)
            compressor.fit(X["train"])
            end_time = time.time()
            t["train"][f"{pca_or_nca}_fit"] = end_time - start_time
        elif pca_or_nca == "nca":
            start_time = time.time()
            compressor = NeighborhoodComponentsAnalysis(n_components=n_components, init=init, random_state=random_state)
            compressor.fit(X["train"], y["train"].ravel())
            end_time = time.time()
            t["train"][f"{pca_or_nca}_fit"] = end_time - start_time
        start_time = time.time()
        X["train"] = compressor.transform(X["train"])
        end_time = time.time()
        t["train"][f"{pca_or_nca}_transform"] = end_time - start_time
        start_time = time.time()
        X["test"] = compressor.transform(X["test"])
        end_time = time.time()
        t["test"][f"{pca_or_nca}_transform"] = end_time - start_time
        os.makedirs(features_dir, exist_ok=False)
        for train_or_test in ["train", "test"]:
            with open(os.path.join(features_dir, f'X_{train_or_test}.npy'), 'wb') as npy_file:
                np.save(npy_file, X[train_or_test])
            with open(os.path.join(features_dir, f'y_{train_or_test}.npy'), 'wb') as npy_file:
                np.save(npy_file, y[train_or_test])
            with open(os.path.join(features_dir, f"shape_{train_or_test}.txt"), 'w') as fd:
                fd.write(f'{X[train_or_test].shape}\n')

        with open(os.path.join(features_dir, 'pre_standard_scaler.sav'), 'wb') as pickle_file:
            pickle.dump(sc, pickle_file)
        with open(os.path.join(features_dir, 'compressor.sav'), 'wb') as pickle_file:
            pickle.dump(compressor, pickle_file)
        with open(os.path.join(features_dir, f"config{CONFIG_EXT}"), 'wb') as pickle_file:
            pickle.dump(config, pickle_file)

        t = classify_with_raport(X["train"], y["train"], X["test"], y["test"], features_dir, dataset_info.idx_to_class, t)
        for train_or_test in ["train", "test"]:
            with open(os.path.join(features_dir, f'time_{train_or_test}.csv'), 'w') as f:
                w = csv.DictWriter(f, t[train_or_test].keys())
                w.writeheader()
                w.writerow(t[train_or_test])
    return features_id
