import os
import pickle
import numpy as np
from tqdm import tqdm

import time
from datasets_preparation import get_strategy
from utils import (
    CBIR_ROOT,
    DATASET_INFO_ROOT,
    FEATURES_ROOT,
    minimal_numpy_int_type,
    )


def make_cbir(dataset_name, features_id):
    strategy = get_strategy(dataset_name)
    dataset_info_dir = os.path.join(DATASET_INFO_ROOT, dataset_name, strategy)
    # features_config = pickle.load(open(os.path.join(CONFIGS_ROOT, f"{features_id}{CONFIG_EXT}"), 'rb'))

    for fold in os.listdir(dataset_info_dir):
        features_dir = os.path.join(FEATURES_ROOT, dataset_name, strategy, fold, features_id)
        cbir_dir = os.path.join(CBIR_ROOT, dataset_name, strategy, fold, features_id)
        if os.path.exists(cbir_dir):
            msg = f"CBIR directory {cbir_dir} already exists."
            raise FileExistsError(msg)
        else:
            os.makedirs(cbir_dir)
        X = dict()
        y = dict()
        dataset_info = dict()
        for train_or_test in ["train", "test"]:
            with open(os.path.join(features_dir, f'X_{train_or_test}.npy'), 'rb') as npy_file:
                X[train_or_test] = np.load(npy_file)
            with open(os.path.join(features_dir, f'y_{train_or_test}.npy'), 'rb') as npy_file:
                y[train_or_test] = np.load(npy_file)
            dataset_info[train_or_test] = pickle.load(open(os.path.join(dataset_info_dir, fold, f"{train_or_test}_dataset_info.pkl"), 'rb'))
            if not np.array_equal(y[train_or_test], dataset_info[train_or_test].encoded_labels):
                msg = f"Features with id {features_id} have inconsistent y_{train_or_test}.npy files with {train_or_test}_dataset_info.pkl."
                raise ValueError(msg)

        all_top_idx = []
        all_top_distances = []
        all_top_encoded_labels = []

        all_query_encoded_labels = y["test"]

        for idx, query in enumerate(tqdm(X["test"])):
            start_time = time.time()

            dists = np.linalg.norm(X["train"] - query, axis=1)

            top_idx = np.argsort(dists)
            top_distances = dists[top_idx]
            top_encoded_labels = dataset_info["train"].encoded_labels[top_idx]

            all_top_idx.append(top_idx)
            all_top_distances.append(top_distances)
            all_top_encoded_labels.append(top_encoded_labels)

            end_time = time.time()
            with open(os.path.join(cbir_dir, "time.csv"), 'a') as fd:
                fd.write(f'{end_time-start_time}\n')

        all_top_idx = np.vstack(all_top_idx)
        all_top_distances = np.vstack(all_top_distances)
        all_top_encoded_labels = np.vstack(all_top_encoded_labels)

        with open(os.path.join(cbir_dir, 'all_top_idx.npy'), 'wb') as npy_file:
            np.save(npy_file, all_top_idx.astype(minimal_numpy_int_type(all_top_idx)))
        with open(os.path.join(cbir_dir, 'all_top_distances.npy'), 'wb') as npy_file:
            np.save(npy_file, all_top_distances.astype(np.float16))
        with open(os.path.join(cbir_dir, 'all_top_encoded_labels.npy'), 'wb') as npy_file:
            np.save(npy_file, all_top_encoded_labels.astype(minimal_numpy_int_type(all_top_encoded_labels)))
        with open(os.path.join(cbir_dir, 'all_query_encoded_labels.npy'), 'wb') as npy_file:
            np.save(npy_file, all_query_encoded_labels.astype(minimal_numpy_int_type(all_query_encoded_labels)))


if __name__ == "__main__":
    features_idx = [6, 63, 51, 70, 88, 52, 53, 71, 54, 90, 201, 72]
    for features_id in features_idx:
        make_cbir(dataset_name="Caltech-256", features_id=str(features_id))
