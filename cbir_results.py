import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from datasets_preparation import get_strategy
from utils import (
    CBIR_PLOTS_ROOT,
    CBIR_RESULTS_ROOT,
    CBIR_ROOT,
    DATASET_INFO_ROOT,
    SUMMARY_CBIR_RESULTS_ROOT,
    SUMMARY_RESULTS_ROOT
    )


def Pr_at_k(k, c, top_encoded_labels):
    return np.count_nonzero(top_encoded_labels[:k] == c)/k


def AP_at_k(k, c, all_query_encoded_labels, all_top_encoded_labels):
    ix = np.where(all_query_encoded_labels == c)[0]
    m_test = len(ix)
    sum_ = 0
    for i in ix:
        top_encoded_labels = all_top_encoded_labels[i][:k]
        sum_ += Pr_at_k(k, c, top_encoded_labels)
    return sum_/m_test


def MAP_over_class_at_k(k, all_query_encoded_labels, all_top_encoded_labels):
    cx = set(all_query_encoded_labels)
    C = len(cx)
    sum_ = 0
    for c in cx:
        sum_ += AP_at_k(k, c, all_query_encoded_labels, all_top_encoded_labels)
    return sum_/C


def MAP_overall_at_k(k, all_query_encoded_labels, all_top_encoded_labels):
    Q = len(all_query_encoded_labels)
    sum_ = 0
    for c, top_encoded_labels in zip(all_query_encoded_labels, all_top_encoded_labels):
        sum_ += Pr_at_k(k, c, top_encoded_labels)
    return sum_/Q


def calculate_metrics(k=20):
    for dataset_name in os.listdir(CBIR_ROOT):
        strategy = get_strategy(dataset_name)
        strategy_path = os.path.join(CBIR_ROOT, dataset_name, strategy)
        for fold in tqdm(os.listdir(strategy_path)):
            for feature_id in os.listdir(os.path.join(strategy_path, fold)):
                cbir_dir = os.path.join(strategy_path, fold, feature_id)
                with open(os.path.join(cbir_dir, 'all_top_encoded_labels.npy'), 'rb') as npy_file:
                    all_top_encoded_labels = np.load(npy_file)
                with open(os.path.join(cbir_dir, 'all_query_encoded_labels.npy'), 'rb') as npy_file:
                    all_query_encoded_labels = np.load(npy_file)

                ap_at_k = dict()
                for c in set(all_query_encoded_labels):
                    ap_at_k[c] = AP_at_k(k, c, all_query_encoded_labels, all_top_encoded_labels)
                cbir_results_dir = os.path.join(CBIR_RESULTS_ROOT, dataset_name, strategy, fold, feature_id)
                os.makedirs(cbir_results_dir, exist_ok=True)
                pickle.dump(ap_at_k, open(os.path.join(cbir_results_dir, f'AP@{k}.pkl'), 'wb'))
                f = open(os.path.join(cbir_results_dir, f"MAP_over_class@{k}.txt"), "w")
                f.write(str(MAP_over_class_at_k(k, all_query_encoded_labels, all_top_encoded_labels)))
                f.close()
                f = open(os.path.join(cbir_results_dir, f"MAP_overall@{k}.txt"), "w")
                f.write(str(MAP_overall_at_k(k, all_query_encoded_labels, all_top_encoded_labels)))
                f.close()


def summary_cbir(dataset_name="Caltech-256", k=20):
    the_best = {
        f"MAP_over_class@{k}": {
            "fold": None,
            "feature_id": None,
            "max_value": 0
            },
        f"MAP_overall@{k}": {
            "fold": None,
            "feature_id": None,
            "max_value": 0
            },
        }
    strategy = get_strategy(dataset_name)
    strategy_path = os.path.join(CBIR_RESULTS_ROOT, dataset_name, strategy)
    summary_dict = dict()
    for fold in tqdm(os.listdir(strategy_path)):
        for feature_id in os.listdir(os.path.join(strategy_path, fold)):
            if feature_id not in summary_dict:
                summary_dict[feature_id] = {
                    f"AP@{k}": dict(),
                    f"MAP_over_class@{k}": [],
                    f"MAP_overall@{k}": []
                    }
            cbir_results_dir = os.path.join(strategy_path, fold, feature_id)
            ap_at_k = pickle.load(open(os.path.join(cbir_results_dir, f'AP@{k}.pkl'), 'rb'))
            for c in ap_at_k:
                if c not in summary_dict[feature_id][f"AP@{k}"]:
                    summary_dict[feature_id][f"AP@{k}"][c] = []
                summary_dict[feature_id][f"AP@{k}"][c].append(ap_at_k[c])
            f = open(os.path.join(cbir_results_dir, f"MAP_over_class@{k}.txt"), "r")
            map_over_class = float(f.read())
            f.close()
            summary_dict[feature_id][f"MAP_over_class@{k}"].append(map_over_class)
            if map_over_class > the_best[f"MAP_over_class@{k}"]["max_value"]:
                the_best[f"MAP_over_class@{k}"]["max_value"] = map_over_class
                the_best[f"MAP_over_class@{k}"]["fold"] = fold
                the_best[f"MAP_over_class@{k}"]["feature_id"] = feature_id
            f = open(os.path.join(cbir_results_dir, f"MAP_overall@{k}.txt"), "r")
            map_overall = float(f.read())
            f.close()
            summary_dict[feature_id][f"MAP_overall@{k}"].append(map_overall)
            if map_overall > the_best[f"MAP_overall@{k}"]["max_value"]:
                the_best[f"MAP_overall@{k}"]["max_value"] = map_overall
                the_best[f"MAP_overall@{k}"]["fold"] = fold
                the_best[f"MAP_overall@{k}"]["feature_id"] = feature_id
    summary_mean_cbir = {
        "feature_id": [],
        f"MAP_over_class@{k}": [],
        f"MAP_overall@{k}": []
        }
    for feature_id in summary_dict:
        cbir_summary_results_dir = os.path.join(SUMMARY_CBIR_RESULTS_ROOT, dataset_name, strategy, feature_id)
        os.makedirs(cbir_summary_results_dir, exist_ok=True)

        ap_at_k_mean = dict()
        for c in summary_dict[feature_id][f"AP@{k}"]:
            ap_at_k_mean[c] = np.mean(summary_dict[feature_id][f"AP@{k}"][c])
        pickle.dump(ap_at_k_mean, open(os.path.join(cbir_summary_results_dir, f'AP@{k}.pkl'), 'wb'))
        f = open(os.path.join(cbir_summary_results_dir, f"MAP_over_class@{k}.txt"), "w")
        oc = np.mean(summary_dict[feature_id][f"MAP_over_class@{k}"])
        f.write(str(oc))
        f.close()
        f = open(os.path.join(cbir_summary_results_dir, f"MAP_overall@{k}.txt"), "w")
        oa = np.mean(summary_dict[feature_id][f"MAP_overall@{k}"])
        f.write(str(oa))
        f.close()
        summary_mean_cbir["feature_id"].append(feature_id)
        summary_mean_cbir[f"MAP_over_class@{k}"].append(oc)
        summary_mean_cbir[f"MAP_overall@{k}"].append(oa)
    pickle.dump(the_best, open(f'cbir_the_best@{k}.pkl', 'wb'))
    df_summary_mean_cbir = pd.DataFrame(summary_mean_cbir)
    df_summary_mean_cbir.to_csv(f"cbir_summary_results@{k}.csv", index=False, header=True)


def plot_MAP():
    dataset_name = "Caltech-256"
    info_name = "accuracy"
    summary_results_path = os.path.join(SUMMARY_RESULTS_ROOT, dataset_name, get_strategy(dataset_name), f"summary_results_{info_name}.csv")
    df_features = pd.read_csv(summary_results_path)

    feature_idx = [63, 51, 70, 88, 52, 53, 71, 54, 90, 201, 72, 6]
    df = pd.read_csv("cbir_summary_results@20.csv")

    variants = []
    MAP_oc = []
    MAP_oa = []

    for feature_id in feature_idx:
        tmp1 = df_features[df_features["features_id"] == feature_id].iloc[0]
        tmp2 = df[df["feature_id"] == feature_id].iloc[0]
        variant_str = (tmp1["base_raw_features"].replace("l", "L") + "; "
                       + (tmp1["transformation"].upper() if tmp1["transformation"] in ["pca", "nca"] else "-")
                       + "; " + str(tmp1["n_features"]))
        variants.append(variant_str)
        MAP_oc.append(tmp2["MAP_over_class@20"])
        MAP_oa.append(tmp2["MAP_overall@20"])

    oa = plt.scatter(variants, MAP_oa, c="red")
    oc = plt.scatter(variants, MAP_oc, c="blue")
    plt.xticks(rotation=90)
    plt.legend((oa, oc), ('MAP_overall@20', 'MAP_over_class@20'), loc='lower right')
    plt.gca().get_xticklabels()[np.argmax(MAP_oa)].set_color("red")
    plt.grid()
    plt.rc('axes', axisbelow=True)  # False # grid in the background
    os.makedirs(CBIR_PLOTS_ROOT, exist_ok=True)
    plt.savefig(os.path.join(CBIR_PLOTS_ROOT, "MAP.png"), dpi=1200, bbox_inches="tight")


def plot_cbir(dataset_name="Caltech-256", k=20, fold="fold_3_5", feature_id='88', query_id=2791):
    strategy = get_strategy(dataset_name)

    # AP = pickle.load(open(os.path.join(CBIR_RESULTS_ROOT, dataset_name, strategy, fold, feature_id, f'AP@{k}.pkl'), 'rb'))

    cbir_dir = os.path.join(CBIR_ROOT, dataset_name, strategy, fold, feature_id)
    with open(os.path.join(cbir_dir, 'all_top_idx.npy'), 'rb') as npy_file:
        all_top_idx = np.load(npy_file)
    with open(os.path.join(cbir_dir, 'all_top_encoded_labels.npy'), 'rb') as npy_file:
        all_top_encoded_labels = np.load(npy_file)
    with open(os.path.join(cbir_dir, 'all_query_encoded_labels.npy'), 'rb') as npy_file:
        all_query_encoded_labels = np.load(npy_file)
    with open(os.path.join(cbir_dir, 'all_top_distances.npy'), 'rb') as npy_file:
        all_top_distances = np.load(npy_file)
    dataset_info_dir = os.path.join(DATASET_INFO_ROOT, dataset_name, strategy, fold)
    query_dataset_info = pickle.load(open(os.path.join(dataset_info_dir, "test_dataset_info.pkl"), 'rb'))
    example_dataset_info = pickle.load(open(os.path.join(dataset_info_dir, "train_dataset_info.pkl"), 'rb'))

    query_class_name = query_dataset_info.idx_to_class[all_query_encoded_labels[query_id]]
    query_img_path = query_dataset_info.paths[query_id]
    example_distances = all_top_distances[query_id][:k]
    example_class_names = np.array([example_dataset_info.idx_to_class[idx] for idx in all_top_encoded_labels[query_id][:k]])
    example_img_idx = all_top_idx[query_id][:k]
    example_img_paths = example_dataset_info.paths[example_img_idx]

    # Visualize the result
    axes = []
    fig = plt.figure(figsize=(8, 8))
    for i in range(k):
        # score = scores[a]
        axes.append(fig.add_subplot(int(k/5), 5, i+1))
        subplot_title = str(round(example_distances[i], 2))+" ["+example_class_names[i]+"]"
        axes[-1].set_title(subplot_title)
        plt.axis('off')
        plt.imshow(Image.open(example_img_paths[i]))
    fig.tight_layout()
    os.makedirs(CBIR_PLOTS_ROOT, exist_ok=True)
    save_path = os.path.join(CBIR_PLOTS_ROOT, f"examples_{dataset_name}_{k}_{fold}_{feature_id}_{query_id}")
    # plt.savefig(save_path, dpi=1200, bbox_inches="tight")

    plt.clf()
    plt.title("["+query_class_name+"]", fontsize=20)
    plt.imshow(Image.open(query_img_path))
    plt.axis('off')
    save_path = os.path.join(CBIR_PLOTS_ROOT, f"query_{dataset_name}_{k}_{fold}_{feature_id}_{query_id}")
    print(query_img_path)
    # plt.savefig(save_path, dpi=1200, bbox_inches="tight")


if __name__ == "__main__":
    # calculate_metrics(k=20)
    # summary_cbir(dataset_name="Caltech-256", k=20)
    # plot_MAP()
    plot_cbir()
