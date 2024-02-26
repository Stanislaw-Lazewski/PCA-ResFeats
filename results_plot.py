# 1. Co wypada lepiej PCA czy NCA?
# 2. Które cechy są najlepsze?
# 3. Które cechy bazowe są najlepsze?
# 4. Czy cechy z outputu sieci są gorsze?
# 5. Który klasyfikator jest lepszy?
# 6. Najlepsze z każdej kategorii (max-pooling, concatenation, PCA, NCA)


import os
import pandas as pd
import numpy as np
import warnings

from datasets_preparation import get_strategy
from features_transformation import (
    build_concatenation_config,
    build_max_pooling_config,
    build_pca_config,
    find_config_id,
    get_all_config_idx,
    )
from utils import (
    SUMMARY_RESULTS_ROOT,
    parse,
    )


def get_top_k_shortest(df, max_val, pp, info_name, k=None):
    df_shortest = df[df[info_name] >= max_val - pp/100.]
    df_shortest = df_shortest.sort_values(["n_features", info_name], ascending=[True, False]).reset_index(drop=True)
    if k is not None:
        df_shortest = df_shortest.head(k)
    return df_shortest


def get_network_output_results(info_name="accuracy", pp=5):
    for dataset_name in os.listdir(SUMMARY_RESULTS_ROOT):
        summary_results_path = os.path.join(SUMMARY_RESULTS_ROOT, dataset_name, get_strategy(dataset_name), f"summary_results_{info_name}.csv")
        df = pd.read_csv(summary_results_path)

        # get best classifier
        idx = df.groupby('features_id')[info_name].idxmax()
        df = df.loc[idx].reset_index(drop=True)

        # Network output
        out = get_all_config_idx("network-output")
        if len(out) != 1:
            msg = f"Multiple `network-output` transformations detected: {out}"
            warnings.warn(msg)
        out = list(map(lambda x: int(x), out))
        df_out = df[df["features_id"].isin(out)].reset_index(drop=True)

        # Resfeats
        preprocess_id = "2"
        backbone_model = parse({
            "model_arch": "resnet50",
            "model_trained_on": "imagenet",
            "suffix": None
            })
        dim = (0, 2, 3)
        npy_type = np.float16
        n_components = 60
        svd_solver = "full"
        config_list = [build_max_pooling_config(backbone_model=backbone_model,
                                                raw_features_name=raw_features_name,
                                                preprocess_id=preprocess_id,
                                                dim=dim,
                                                npy_type=npy_type) for raw_features_name in ["l2", "l3", "l4"]]
        component_features_config = build_concatenation_config(config_list=config_list,
                                                               npy_type=npy_type)
        resfeats_id = find_config_id(config=component_features_config)
        df_res = df[df["features_id"] == int(resfeats_id)].reset_index(drop=True)

        # PCA-ResFeats
        pca_resfeats_config = build_pca_config(component_features_config=component_features_config,
                                               n_components=n_components,
                                               svd_solver=svd_solver,
                                               npy_type=npy_type)
        pca_resfeats_id = find_config_id(config=pca_resfeats_config)
        df_pca_res = df[df["features_id"] == int(pca_resfeats_id)].reset_index(drop=True)

        # top-10-info_name
        df_best = df.sort_values(info_name, ascending=False).reset_index(drop=True)
        df_best = df_best.head(10)

        # top-10-shortest-not-worse-than-1-p.p-than-network-output
        df_shortest1_out = get_top_k_shortest(df, df_out[info_name].max(), 1, info_name, k=10)

        # top-10-shortest-not-worse-than-pp-p.p-than-network-output
        df_shortest_out = get_top_k_shortest(df, df_out[info_name].max(), pp, info_name, k=10)

        # top-10-shortest-not-worse-than-1-p.p-than-the-best
        df_shortest1_best = get_top_k_shortest(df, df_best[info_name].max(), 1, info_name, k=10)

        # top-10-shortest-not-worse-than-pp-p.p-than-the-best
        df_shortest_best = get_top_k_shortest(df, df_best[info_name].max(), pp, info_name, k=10)

        # print results
        printed_columns = [info_name, 'n_features', "classifier", "transformation", "base_raw_features", "features_id"]
        print(dataset_name)
        print()
        print("network-output")
        print(df_out[printed_columns])
        print()
        print("resfeats")
        print(df_res[printed_columns])
        print()
        print("pca-resfeats")
        print(df_pca_res[printed_columns])
        print()
        print(f"top-10-{info_name}")
        print(df_best[printed_columns])
        print()
        print("top-10-shortest-not-worse-than-1-p.p-than-network-output")
        print(df_shortest1_out[printed_columns])
        print()
        print(f"top-10-shortest-not-worse-than-{pp}-p.p-than-network-output")
        print(df_shortest_out[printed_columns])
        print()
        print("top-10-shortest-not-worse-than-1-p.p-than-the-best")
        print(df_shortest1_best[printed_columns])
        print()
        print(f"top-10-shortest-not-worse-than-{pp}-p.p-than-the-best")
        print(df_shortest_best[printed_columns])
        print()
        print()


def get_top_1_for_each_combination(info_name="accuracy"):
    for dataset_name in os.listdir(SUMMARY_RESULTS_ROOT):
        summary_results_path = os.path.join(SUMMARY_RESULTS_ROOT, dataset_name, get_strategy(dataset_name), f"summary_results_{info_name}.csv")
        df = pd.read_csv(summary_results_path)

        # get best classifier
        idx = df.groupby('features_id')[info_name].idxmax()
        df = df.loc[idx].reset_index(drop=True)

        idx_ = df.groupby(['base_raw_features', 'transformation'])[info_name].idxmax()
        df_best = df.loc[idx_].reset_index(drop=True)

        printed_columns = [info_name, 'n_features', "classifier", "transformation", "base_raw_features", "features_id"]

        print(dataset_name)
        print()
        print(f"top-1-{info_name}-for-each-combination")
        print(df_best[printed_columns].sort_values(info_name, ascending=False))
        print()
        print()


def best_combination_ranking(info_name="accuracy", pp=1, k=10):
    full_df = []
    dataset_list = os.listdir(SUMMARY_RESULTS_ROOT)
    for dataset_name in dataset_list:
        summary_results_path = os.path.join(SUMMARY_RESULTS_ROOT, dataset_name, get_strategy(dataset_name), f"summary_results_{info_name}.csv")
        df = pd.read_csv(summary_results_path)

        # get best classifier
        idx = df.groupby('features_id')[info_name].idxmax()
        df = df.loc[idx]

        df = df.sort_values("n_features", ascending=True).reset_index(drop=True)
        best_result_id = df[info_name].idxmax()
        # all-not-worse-than-pp-p.p-than-the-best
        df_all_in_pp_limit = df[df[info_name] >= df.loc[best_result_id][info_name] - pp/100.].copy()

        df_all_in_pp_limit["dataset_name"] = dataset_name
        df_all_in_pp_limit["best_result"] = df.loc[best_result_id][info_name]
        df_all_in_pp_limit["best_result_n_features"] = df.loc[best_result_id]["n_features"]
        df_all_in_pp_limit["best_result_transformation"] = df.loc[best_result_id]["transformation"]
        df_all_in_pp_limit["best_result_base_raw_features"] = df.loc[best_result_id]["base_raw_features"]
        df_all_in_pp_limit["best_result_features_id"] = df.loc[best_result_id]["features_id"]

        full_df.append(df_all_in_pp_limit)
    full_df = pd.concat(full_df).reset_index(drop=True)
    grouped_df = full_df.groupby("features_id")
    printed_columns = ["features_id",
                       "transformation",
                       "base_raw_features",
                       "n_features",
                       info_name,
                       "best_result",
                       "best_result_n_features",
                       "best_result_transformation",
                       "best_result_base_raw_features",
                       "best_result_features_id",
                       "dataset_name"]
    full_df = []
    for features_id, group in grouped_df:
        if len(group) == len(dataset_list):
            n_features_array = group["n_features"].unique()
            if len(n_features_array) == 1:
                full_df.append((n_features_array.item(), group[printed_columns]))
            else:
                msg = f"Multiple `n_features` {n_features_array} info for {features_id} config."
                raise RuntimeError(msg)
    full_df.sort(key=lambda x: x[0])

    final_df = dict()
    columns = ["base_raw_features",
               "transformation",
               "n_features",
               "Caltech-256",
               "Caltech-101",
               "EuroSAT",
               "Oxford Flowers",
               "Dogs vs. Cats"]
    for c in columns:
        final_df[c] = []

    for n_features, df in full_df:
        print(df.to_string(index=False))
        print()
        for c in ["base_raw_features",
                  "transformation",
                  "n_features"]:
            val = df[c].unique()
            if len(val) == 1:
                final_df[c].append(val.item())
            else:
                msg = f"Multiple {c} value in full_df single group."
                raise RuntimeError(msg)
        for c in ["Caltech-256",
                  "Caltech-101",
                  "EuroSAT",
                  "Oxford Flowers",
                  "Dogs vs. Cats"]:
            final_df[c].append(round(df[df["dataset_name"] == c][info_name].item()*100, 2))
    final_df = pd.DataFrame(final_df)
    final_df.to_csv("final_df.csv", index=False)
    # agg_dict = dict()
    # for column in printed_columns:
    #     if column == "features_id_":
    #         agg_dict[column] = "count"
    #     elif column == info_name:
    #         agg_dict[column] = lambda x: [x]
    #     else:
    #         agg_dict[column] = "first"
    # df = (full_df
    #       .groupby('features_id_')
    #       .agg(agg_dict)
    #       .rename(columns={'features_id_': 'frequency', info_name: f'{info_name}_values'})
    #       .sort_values(['frequency', 'n_features'], ascending=[False, True])
    #       .reset_index(drop=True)
    #       )

    # print(f"top-combination-ranking-not-worse-than-{pp}-p.p-than-the-best")
    # print(df[df["frequency"] == len(dataset_list)])
    # print()


# get_network_output_results(info_name="accuracy")
# get_network_output_results(info_name="balanced_accuracy")

# get_top_1_for_each_combination(info_name="accuracy")

best_combination_ranking(info_name="accuracy", pp=1)
