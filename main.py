import os
import itertools

import numpy as np
import torch

from datasets_preparation import prepare_dataset
from features_extraction import extract_features
from features_transformation import (
    build_max_pooling_config,
    features_concatenation,
    features_pca_nca,
    find_config_id,
    get_all_config_idx,
    purge_config,
    raw_features_amax,
    raw_features_out
)
from utils import (
    ALL_HISTO_DATASET_NAMES,
    DATASETS_MIDDLE_DIRS,
    DATASETS_ROOT,
    HISTO_DATASET_NAMES,
    OBJECT_DATASETS_NAMES,
    parse
)


N_COMPONENTS_LIST = [30, 60, 100, 200, 400, 600, 800, 1000, 1200]


def main_histo():
    # # DATASET PREPRARTION
    # print("DATASET PREPRARTION")
    # for dataset_name in HISTO_DATASET_NAMES:
    #     if dataset_name in DATASETS_MIDDLE_DIRS:
    #         dataset_dir = os.path.join(DATASETS_ROOT, dataset_name, DATASETS_MIDDLE_DIRS[dataset_name])
    #     else:
    #         dataset_dir = os.path.join(DATASETS_ROOT, dataset_name)
    #     if dataset_name == "BreaKHis":
    #         prepare_dataset(dataset_name=dataset_name,
    #                         dataset_dir=dataset_dir,
    #                         n_classes=2)
    #         prepare_dataset(dataset_name=dataset_name,
    #                         dataset_dir=dataset_dir,
    #                         n_classes=8)
    #     else:
    #         prepare_dataset(dataset_name=dataset_name,
    #                         dataset_dir=dataset_dir)
    # ##

    # # DATASET standard ResNet-50 features extraction
    # print("DATASET standard ResNet-50 features extraction")
    # device = torch.device('cuda:0')
    # preprocess_id = "2"
    # backbone_model = parse({
    #     "model_arch": "resnet50",
    #     "model_trained_on": "imagenet",
    #     "suffix": None
    #     })
    # for dataset_name in ALL_HISTO_DATASET_NAMES:
    #     extract_features(dataset_name=dataset_name,
    #                      device=device,
    #                      preprocess_id=preprocess_id,
    #                      backbone_model=backbone_model)
    # ##

    # # ALL_HISTO_DATASET_NAMES NETWORK OUTPUTS
    # print("ALL_HISTO_DATASET_NAMES network-output transformation on raw features")
    # npy_type = np.float16
    # preprocess_id = "2"
    # backbone_model = parse({
    #     "model_arch": "resnet50",
    #     "model_trained_on": "imagenet",
    #     "suffix": None
    #     })
    # for dataset_name in ALL_HISTO_DATASET_NAMES:
    #     output_id = raw_features_out(dataset_name=dataset_name,
    #                                  preprocess_id=preprocess_id,
    #                                  backbone_model=backbone_model,
    #                                  raw_features_out_name="l4a",
    #                                  npy_type=npy_type
    #                                  )
    # ##

    # # ALL_HISTO_DATASET_NAMES PCA/NCA on network output
    # for dataset_name in ALL_HISTO_DATASET_NAMES:
    #     for n_components in N_COMPONENTS_LIST:
    #         print(f"{dataset_name} PCA({n_components}) on network output")
    #         features_pca_nca(dataset_name=dataset_name,
    #                          component_feature=output_id,
    #                          pca_or_nca="pca",
    #                          n_components=n_components,
    #                          svd_solver="full",
    #                          npy_type=npy_type)
    #         print(f"{dataset_name} NCA({n_components}) on network output")
    #         features_pca_nca(dataset_name=dataset_name,
    #                          component_feature=output_id,
    #                          pca_or_nca="nca",
    #                          n_components=n_components,
    #                          init="pca",
    #                          npy_type=npy_type)

    # ##

    # ALL_HISTO_DATASET_NAMES max-pooling transformation on raw features
    print("ALL_HISTO_DATASET_NAMES max-pooling transformation on raw features")
    preprocess_id = "2"
    backbone_model = parse({
        "model_arch": "resnet50",
        "model_trained_on": "imagenet",
        "suffix": None
        })
    dim = (0, 2, 3)
    npy_type = np.float16
    layers_list = ["l1", "l2", "l3", "l4"]
    for dataset_name in ALL_HISTO_DATASET_NAMES:
        max_pooling_idx = []
        for raw_features_name in layers_list:
            print(f"{dataset_name} max-pooling transformation on raw features {raw_features_name}")
            max_pooling_idx.append(
                raw_features_amax(dataset_name=dataset_name,
                                  preprocess_id=preprocess_id,
                                  backbone_model=backbone_model,
                                  raw_features_name=raw_features_name,
                                  dim=dim,
                                  npy_type=npy_type)
                )
    ##

    # OBJECT_DATASETS combinations after max-pooling
    print("ALL_HISTO_DATASET_NAMES combinations after max-pooling")
    layers_combinations = []
    for i in range(1, len(max_pooling_idx)+1):
        layers_combinations = layers_combinations + list(itertools.combinations(max_pooling_idx, i))
    layers_combinations = list(map(lambda x: list(x), layers_combinations))
    for dataset_name in ALL_HISTO_DATASET_NAMES:
        concatenation_idx = []
        for i, features_id_list in enumerate(layers_combinations):
            print(f"{dataset_name} combinations after max-pooling {features_id_list} {i+1}/{len(layers_combinations)}")
            concatenation_idx.append(
                features_concatenation(dataset_name=dataset_name,
                                       features_id_list=features_id_list,
                                       npy_type=npy_type)
                )
    ##

    # ALL_HISTO_DATASET_NAMES PCA/NCA on combinations
    for dataset_name in ALL_HISTO_DATASET_NAMES:
        for i, component_feature in enumerate(concatenation_idx):
            for n_components in N_COMPONENTS_LIST:
                print(f"{dataset_name} PCA({n_components}) on {layers_combinations[i]} {i+1}/{len(concatenation_idx)}")
                features_pca_nca(dataset_name=dataset_name,
                                 component_feature=component_feature,
                                 pca_or_nca="pca",
                                 n_components=n_components,
                                 svd_solver="full",
                                 npy_type=npy_type)
                print(f"{dataset_name} NCA({n_components}) on {layers_combinations[i]} {i+1}/{len(concatenation_idx)}")
                features_pca_nca(dataset_name=dataset_name,
                                 component_feature=component_feature,
                                 pca_or_nca="nca",
                                 n_components=n_components,
                                 init="pca",
                                 npy_type=npy_type)

    ##


def main():
    # # OBJECT_DATASETS preparation
    # print("OBJECT_DATASETS preparation")
    # for dataset_name in OBJECT_DATASETS_NAMES:
    #     if dataset_name in DATASETS_MIDDLE_DIRS:
    #         dataset_dir = os.path.join(DATASETS_ROOT, dataset_name, DATASETS_MIDDLE_DIRS[dataset_name])
    #     else:
    #         dataset_dir = os.path.join(DATASETS_ROOT, dataset_name)
    #     if dataset_name == "Caltech-101":
    #         prepare_dataset(dataset_name=dataset_name,
    #                         dataset_dir=dataset_dir,
    #                         skipped_classes=["BACKGROUND_Google"])
    #     elif dataset_name == "Caltech-256":
    #         prepare_dataset(dataset_name=dataset_name,
    #                         dataset_dir=dataset_dir,
    #                         skipped_classes=["257.clutter"])
    #     else:
    #         prepare_dataset(dataset_name=dataset_name,
    #                         dataset_dir=dataset_dir)
    # ##

    # # OBJECT_DATASETS standard ResNet-50 features extraction
    # print("OBJECT_DATASETS standard ResNet-50 features extraction")
    # device = torch.device('cuda:0')
    # preprocess_id = "2"
    # backbone_model = parse({
    #     "model_arch": "resnet50",
    #     "model_trained_on": "imagenet",
    #     "suffix": None
    #     })
    # for dataset_name in OBJECT_DATASETS_NAMES:
    #     extract_features(dataset_name=dataset_name,
    #                      device=device,
    #                      preprocess_id=preprocess_id,
    #                      backbone_model=backbone_model)
    # ##

    # # OBJECT_DATASETS max-pooling transformation on raw features
    # print("OBJECT_DATASETS max-pooling transformation on raw features")
    # dim = (0, 2, 3)
    # npy_type = np.float16
    # layers_list = ["l2", "l3", "l4"]
    # for dataset_name in OBJECT_DATASETS_NAMES:
    #     max_pooling_idx = []
    #     for raw_features_name in layers_list:
    #         max_pooling_idx.append(
    #             raw_features_amax(dataset_name=dataset_name,
    #                               preprocess_id=preprocess_id,
    #                               backbone_model=backbone_model,
    #                               raw_features_name=raw_features_name,
    #                               dim=dim,
    #                               npy_type=npy_type)
    #             )
    # ##

    # # OBJECT_DATASETS concatenation after max-pooling
    # print("OBJECT_DATASETS concatenation after max-pooling")
    # for dataset_name in OBJECT_DATASETS_NAMES:
    #     layers_combinations = list(itertools.combinations(max_pooling_idx, 2)) + [max_pooling_idx]
    #     concatenation_id = []
    #     for features_id_tuple in layers_combinations:
    #         concatenation_id.append(
    #             features_concatenation(dataset_name=dataset_name,
    #                                    features_id_list=list(features_id_tuple),
    #                                    npy_type=npy_type)
    #             )
    # ##

    # # OBJECT_DATASETS NCA on concatenated
    # print("OBJECT_DATASETS NCA on concatenated")
    # for dataset_name in OBJECT_DATASETS_NAMES:
    #     for component_feature in concatenation_id:
    #         for n_components in N_COMPONENTS_LIST:
    #             features_pca_nca(dataset_name=dataset_name,
    #                              component_feature=component_feature,
    #                              pca_or_nca="nca",
    #                              n_components=n_components,
    #                              init="pca",
    #                              npy_type=npy_type)
    # ##

    # # OBJECT_DATASETS PCA on concatenated
    # print("OBJECT_DATASETS PCA on concatenated")
    # for dataset_name in OBJECT_DATASETS_NAMES:
    #     for component_feature in concatenation_id:
    #         for n_components in N_COMPONENTS_LIST:
    #             features_pca_nca(dataset_name=dataset_name,
    #                              component_feature=component_feature,
    #                              pca_or_nca="pca",
    #                              n_components=n_components,
    #                              svd_solver="full",
    #                              npy_type=npy_type)
    # ##

    # OBJECT_DATASETS PCA/NCA on concatenated
    # for features_id in get_all_config_idx("nca") + get_all_config_idx("pca"):
    #     purge_config(features_id)
    # concatenation_idx = get_all_config_idx("concatenation")
    # npy_type = np.float16
    # for dataset_name in OBJECT_DATASETS_NAMES:
    #     for component_feature in concatenation_idx:
    #         for n_components in N_COMPONENTS_LIST:
    #             print(f"{dataset_name} PCA({n_components}) on concatenated")
    #             features_pca_nca(dataset_name=dataset_name,
    #                              component_feature=component_feature,
    #                              pca_or_nca="pca",
    #                              n_components=n_components,
    #                              svd_solver="full",
    #                              npy_type=npy_type)
    #             print(f"{dataset_name} NCA({n_components}) on concatenated")
    #             features_pca_nca(dataset_name=dataset_name,
    #                              component_feature=component_feature,
    #                              pca_or_nca="nca",
    #                              n_components=n_components,
    #                              init="pca",
    #                              npy_type=npy_type)

    ##

    # # OBJECT_DATASETS NETWORK OUTPUTS
    # print("OBJECT_DATASETS network-output transformation on raw features")
    # npy_type = np.float16
    # preprocess_id = "2"
    # backbone_model = parse({
    #     "model_arch": "resnet50",
    #     "model_trained_on": "imagenet",
    #     "suffix": None
    #     })
    # for dataset_name in OBJECT_DATASETS_NAMES:
    #     output_id = raw_features_out(dataset_name=dataset_name,
    #                                  preprocess_id=preprocess_id,
    #                                  backbone_model=backbone_model,
    #                                  raw_features_out_name="l4a",
    #                                  npy_type=npy_type
    #                                  )
    # # ##

    # # OBJECT_DATASETS PCA/NCA on network output
    # for dataset_name in OBJECT_DATASETS_NAMES:
    #     for n_components in N_COMPONENTS_LIST:
    #         print(f"{dataset_name} PCA({n_components}) on network output")
    #         features_pca_nca(dataset_name=dataset_name,
    #                          component_feature=output_id,
    #                          pca_or_nca="pca",
    #                          n_components=n_components,
    #                          svd_solver="full",
    #                          npy_type=npy_type)
    #         print(f"{dataset_name} NCA({n_components}) on network output")
    #         features_pca_nca(dataset_name=dataset_name,
    #                          component_feature=output_id,
    #                          pca_or_nca="nca",
    #                          n_components=n_components,
    #                          init="pca",
    #                          npy_type=npy_type)

    # ##

    # # OBJECT_DATASETS standard ResNet-50 features extraction
    # print("OBJECT_DATASETS standard ResNet-50 features extraction")
    # device = torch.device('cuda:0')
    # preprocess_id = "2"
    # backbone_model = parse({
    #     "model_arch": "resnet50",
    #     "model_trained_on": "imagenet",
    #     "suffix": None
    #     })
    # for dataset_name in OBJECT_DATASETS_NAMES:
    #     extract_features(dataset_name=dataset_name,
    #                      device=device,
    #                      preprocess_id=preprocess_id,
    #                      backbone_model=backbone_model)
    # ##


    # # CHECK l2, l3, l4 and MOVE l1 from faw_features_1 and CHANGE RAW_FEATURES_ROOT
    # OBJECT_DATASETS max-pooling transformation on raw features
    print("OBJECT_DATASETS max-pooling transformation on raw features")
    preprocess_id = "2"
    backbone_model = parse({
        "model_arch": "resnet50",
        "model_trained_on": "imagenet",
        "suffix": None
        })
    dim = (0, 2, 3)
    npy_type = np.float16
    for dataset_name in OBJECT_DATASETS_NAMES:
        max_pooling_1 = raw_features_amax(dataset_name=dataset_name,
                                          preprocess_id=preprocess_id,
                                          backbone_model=backbone_model,
                                          raw_features_name="l1",
                                          dim=dim,
                                          npy_type=npy_type)
    ##


    # # OBJECT_DATASETS concatenation after max-pooling
    print("OBJECT_DATASETS concatenation after max-pooling")
    max_pooling_1_copy = find_config_id(
        build_max_pooling_config(
            backbone_model=backbone_model,
            raw_features_name="l1",
            preprocess_id=preprocess_id,
            dim=dim,
            npy_type=npy_type
            )
        )
    assert max_pooling_1_copy == max_pooling_1

    max_pooling_2 = find_config_id(
        build_max_pooling_config(
            backbone_model=backbone_model,
            raw_features_name="l2",
            preprocess_id=preprocess_id,
            dim=dim,
            npy_type=npy_type
            )
        )

    max_pooling_3 = find_config_id(
        build_max_pooling_config(
            backbone_model=backbone_model,
            raw_features_name="l3",
            preprocess_id=preprocess_id,
            dim=dim,
            npy_type=npy_type
            )
        )

    max_pooling_4 = find_config_id(
        build_max_pooling_config(
            backbone_model=backbone_model,
            raw_features_name="l4",
            preprocess_id=preprocess_id,
            dim=dim,
            npy_type=npy_type
            )
        )

    layers_combinations = [
        [max_pooling_1, max_pooling_2],
        [max_pooling_1, max_pooling_3],
        [max_pooling_1, max_pooling_4],
        [max_pooling_1, max_pooling_2, max_pooling_3],
        [max_pooling_1, max_pooling_2, max_pooling_4],
        [max_pooling_1, max_pooling_3, max_pooling_4],
        [max_pooling_1, max_pooling_2, max_pooling_3, max_pooling_4],
        ]

    for dataset_name in OBJECT_DATASETS_NAMES:
        concatenation_idx = []
        for i, features_idx in enumerate(layers_combinations):
            print(dataset_name, "concatenation:", i+1, "/", len(layers_combinations))
            concatenation_idx.append(
                features_concatenation(dataset_name=dataset_name,
                                       features_id_list=features_idx,
                                       npy_type=npy_type)
                )
    ##

    # OBJECT_DATASETS PCA/NCA on concatenation with l1
    for dataset_name in OBJECT_DATASETS_NAMES:
        for i, concatenation_id in enumerate(concatenation_idx):
            for n_components in N_COMPONENTS_LIST:
                print(f"{dataset_name} PCA({n_components}) on {i+1}/{len(concatenation_idx)}")
                features_pca_nca(dataset_name=dataset_name,
                                 component_feature=concatenation_id,
                                 pca_or_nca="pca",
                                 n_components=n_components,
                                 svd_solver="full",
                                 npy_type=npy_type)
                print(f"{dataset_name} NCA({n_components}) on {i+1}/{len(concatenation_idx)}")
                features_pca_nca(dataset_name=dataset_name,
                                 component_feature=concatenation_id,
                                 pca_or_nca="nca",
                                 n_components=n_components,
                                 init="pca",
                                 npy_type=npy_type)

    ##

    # OBJECT_DATASETS PCA/NCA on l1
    for dataset_name in OBJECT_DATASETS_NAMES:
        for n_components in N_COMPONENTS_LIST:
            print(f"{dataset_name} PCA({n_components}) on l1")
            features_pca_nca(dataset_name=dataset_name,
                             component_feature=max_pooling_1,
                             pca_or_nca="pca",
                             n_components=n_components,
                             svd_solver="full",
                             npy_type=npy_type)
            print(f"{dataset_name} NCA({n_components}) on l1")
            features_pca_nca(dataset_name=dataset_name,
                             component_feature=max_pooling_1,
                             pca_or_nca="nca",
                             n_components=n_components,
                             init="pca",
                             npy_type=npy_type)

    ##


if __name__ == "__main__":
    # main()
    main_histo()
