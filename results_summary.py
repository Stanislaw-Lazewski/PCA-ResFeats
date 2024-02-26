import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from datasets_preparation import get_strategy
from utils import (
    beautify_config_str,
    CONFIG_EXT,
    FEATURES_ROOT,
    RESULTS_ROOT,
    SUMMARY_RESULTS_ROOT,
)

ONE_VALUE_INFO = {
    "accuracy": {
        "print_name": "Accuracy: ",
        "type": float
        },
    "balanced_accuracy": {
        "print_name": "Balanced_accuracy: ",
        "type": float
        },
    "n_features": {
        "print_name": "Features number: ",
        "type": int
        },
    }

RESULTS_METRICS = [
    "accuracy",
    "balanced_accuracy",
    ]

CLASSIFICATION_REPORT_PREFIX = "classification_report_"


def get_base_raw_features_list(config):
    raw_features_list = []
    if config["features_type"] == "raw_features":
        raw_features_list += [config["params"]["layer"]]
        return raw_features_list
    for input_config in config["input"]:
        raw_features_list += get_base_raw_features_list(input_config)
    return raw_features_list


def get_classifier_names(classification_report_dir):
    classifiers = []
    for filename in os.listdir(classification_report_dir):
        if CLASSIFICATION_REPORT_PREFIX in filename:
            classifier_name = os.path.splitext(filename)[0].split(CLASSIFICATION_REPORT_PREFIX)[-1]
            classifiers.append((classifier_name, filename))
    return classifiers


def get_info_from_classification_report(classification_report_path, info_name):
    if info_name not in ONE_VALUE_INFO:
        msg = f"{info_name} is not implemented yet."
        raise NotImplementedError(msg)
    with open(classification_report_path) as f:
        lines = f.readlines()
    info = []
    for line in lines:
        if ONE_VALUE_INFO[info_name]["print_name"] in line:
            info.append(line)
    if len(info) > 1:
        msg = f"File {classification_report_path} corrupted. Multiple {info_name} info."
        raise RuntimeError(msg)
    elif len(info) == 0:
        msg = f"File {classification_report_path} does not contain {info_name} info."
        ValueError(msg)
    info_value = ONE_VALUE_INFO[info_name]["type"](info[0].split(ONE_VALUE_INFO[info_name]["print_name"])[-1].strip())
    return info_value


def collect_results_from_classification_reports(dataset_name, info_name):
    data = {
        info_name: [],
        "n_features": [],
        "fold": [],
        "n_splits": [],
        "classifier": [],
        "features_id": [],
        "transformation": [],
        "base_raw_features": [],
        "config": [],
        "directory": []
        }
    strategy = get_strategy(dataset_name=dataset_name)
    strategy_dir = os.path.join(FEATURES_ROOT, dataset_name, strategy)
    for fold_str in tqdm(os.listdir(strategy_dir)):
        fold_dir = os.path.join(strategy_dir, fold_str)
        _, fold, n_splits = fold_str.split("_")
        fold = int(fold)
        n_splits = int(n_splits)
        for features_id in os.listdir(fold_dir):
            config_path = os.path.join(fold_dir, features_id, f"config{CONFIG_EXT}")
            if not os.path.exists(config_path):
                continue
            config = pickle.load(open(config_path, 'rb'))
            config_str = beautify_config_str(str(config))
            classifiers = get_classifier_names(os.path.join(fold_dir, features_id))
            for (classifier_name, filename) in classifiers:
                report_path = os.path.join(fold_dir, features_id, filename)
                info = get_info_from_classification_report(report_path, info_name)
                n_features = get_info_from_classification_report(report_path, "n_features")
                data[info_name].append(info)
                data["n_features"].append(n_features)
                data["fold"].append(fold)
                data["n_splits"].append(n_splits)
                data["classifier"].append(classifier_name)
                data["features_id"].append(features_id)
                data["transformation"].append(config["transformation"])
                data["base_raw_features"].append("_".join(get_base_raw_features_list(config)))
                data["config"].append(config_str)
                data["directory"].append(os.path.join(fold_dir, features_id))
    df = pd.DataFrame(data)
    results_dir = os.path.join(RESULTS_ROOT, dataset_name, strategy)
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, f"results_{info_name}.csv"), index=False, header=True)


def summary_results_from_classification_reports(dataset_name, info_name):
    strategy = get_strategy(dataset_name=dataset_name)
    results_dir = os.path.join(RESULTS_ROOT, dataset_name, strategy)
    data = dict()
    df = pd.read_csv(os.path.join(results_dir, f"results_{info_name}.csv"))
    for index, row in df.iterrows():
        key = (row["features_id"], row["classifier"])
        if key not in data:
            data[key] = {
                "n_splits": row["n_splits"],
                info_name: [row[info_name]],
                "n_features": row["n_features"],
                "transformation": row["transformation"],
                "base_raw_features": row["base_raw_features"],
                "config": row["config"],
                }
        else:
            assert data[key]["n_splits"] == row["n_splits"], "Values in results file are inconsistent."
            assert data[key]["n_features"] == row["n_features"], "Values in results file are inconsistent."
            assert data[key]["config"] == row["config"], "Values in results file are inconsistent."
            assert data[key]["transformation"] == row["transformation"], "Values in results file are inconsistent."
            assert data[key]["base_raw_features"] == row["base_raw_features"], "Values in results file are inconsistent."
            data[key][info_name].append(row[info_name])
    summary_data = {
        "features_id": [],
        "classifier": [],
        "completed": [],
        info_name: [],
        "n_features": [],
        "transformation": [],
        "base_raw_features": [],
        "config": [],
        }
    for key in data:
        (features_id, classifier) = key
        summary_data["features_id"].append(features_id)
        summary_data["classifier"].append(classifier)
        summary_data["completed"].append(True if data[key]["n_splits"] == len(data[key][info_name]) else False)
        summary_data[info_name].append(np.mean(data[key][info_name]))
        summary_data["n_features"].append(data[key]["n_features"])
        summary_data["transformation"].append(data[key]["transformation"])
        summary_data["base_raw_features"].append(data[key]["base_raw_features"])
        summary_data["config"].append(data[key]["config"])
    summary_df = pd.DataFrame(summary_data)
    summary_results_dir = os.path.join(SUMMARY_RESULTS_ROOT, dataset_name, strategy)
    os.makedirs(summary_results_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(summary_results_dir, f"summary_results_{info_name}.csv"), index=False, header=True)


if __name__ == "__main__":
    for dataset_name in os.listdir(FEATURES_ROOT):
        for info_name in RESULTS_METRICS:
            collect_results_from_classification_reports(dataset_name, info_name)
            summary_results_from_classification_reports(dataset_name, info_name)
