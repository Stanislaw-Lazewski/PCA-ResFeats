import torch
from torchvision import transforms
import numpy as np


from run import (run_features_extraction, run_features_combination, 
                 run_out_features_conversion_to_npy, run_features_conversion_to_npy,
                 run_datasets_minimal_balancing, run_splitting_minimal_datasets,
                 run_pca_minimal_datasets, run_pca_classification_minimal_datasets,
                 run_features_classification_minimal_datasets, run_subsampled_classification_minimal_datasets,
                 run_random_subsampled_classification_minimal_datasets, 
                 run_zfp_pca_classification_minimal_datasets, run_save_minimal_image_datasets,
                 run_get_size, run_cbir_caltech256)


n__to_components ={
    10: 359,
    20: 180,
    30: 120,
    40: 90,
    50: 72,
    60: 60,
    70: 52,
    80: 45,
    90: 40, 
    100: 36
    }


benchmark_datasets = {
    "Flowers": "flowers",
    "Caltech101": "caltech101",
    "Catordog": "catordog",
    "EuroSAT": "eurosat",
    "Caltech256": "caltech256",
    "MLC2008": "mlc2008",
    }


# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])


model_types = ["imagenet"]
test_sizes = [0.2]
layers_combinations = [["l2", "l3", "l4"], ["l2", "l3"], ["l2", "l4"], ["l3", "l4"]]
features_combinations = ["out", "l2_l3_l4", "l2_l3", "l2_l4", "l3_l4"]
data_types = [np.float16, np.float32]
pca_arguments = list(n__to_components.values())
random_subsamples = list(n__to_components.values())
every_ns = list(n__to_components.keys())



# zfp
tolerance_list = [100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]




# FEATURES EXTRACTION

run_features_extraction(benchmark_datasets, model_types, preprocess, device)

# RESFEATS COMBINATION

run_features_combination(benchmark_datasets, model_types, layers_combinations)



# # CONVERSION TO NPY

# # OUT FEATURES

run_out_features_conversion_to_npy(benchmark_datasets, model_types, X_npy_type = np.float32)
run_out_features_conversion_to_npy(benchmark_datasets, model_types, X_npy_type = np.float16)

# # RESFEATS FEATURES

run_features_conversion_to_npy(benchmark_datasets, model_types, layers_combinations, X_npy_type = np.float32)
run_features_conversion_to_npy(benchmark_datasets, model_types, layers_combinations, X_npy_type = np.float16)


# # PREPARATION MINIMAL DATASETS 

run_datasets_minimal_balancing(benchmark_datasets, model_types, features_combinations, minimal_start_idx=0, minimal_iter=10)
      
# # TEST TRAIN SPLITTING  

run_splitting_minimal_datasets(benchmark_datasets, model_types, features_combinations, test_sizes, minimal_start_idx=0, minimal_iter=10, trial_start_idx=0, trial_iter=10)




# # PCA MINIMAL

run_pca_minimal_datasets(benchmark_datasets, model_types, features_combinations, test_sizes, minimal_start_idx=0, minimal_iter=10, trial_start_idx=0, trial_iter=10, pca_arguments=pca_arguments, data_types=data_types)
      

# # CLASSIFICATION PCA MINIMAL

run_pca_classification_minimal_datasets(benchmark_datasets, model_types, features_combinations, test_sizes, minimal_start_idx=0, minimal_iter=10, trial_start_idx=0, trial_iter=10, pca_arguments=pca_arguments, data_types=data_types, save_model=False)


# # CLASSIFICATION FEATURES MINIMAL

run_features_classification_minimal_datasets(benchmark_datasets, model_types, features_combinations, test_sizes, minimal_start_idx=0, minimal_iter=10, trial_start_idx=0, trial_iter=10, data_types = data_types, save_model=False)


# # CLASSIFICATION SUBSAMPLED MINIMAL

run_subsampled_classification_minimal_datasets(benchmark_datasets, model_types, features_combinations, test_sizes, minimal_start_idx=0, minimal_iter=10, trial_start_idx=0, trial_iter=10, every_ns = every_ns, data_types = data_types, save_model=False)


# # CLASSIFICATION RANDOM SUBSAMPLED MINIMAL

run_random_subsampled_classification_minimal_datasets(benchmark_datasets, model_types, features_combinations, test_sizes, minimal_start_idx=0, minimal_iter=10, trial_start_idx=0, trial_iter=10, random_subsamples = random_subsamples, data_types = data_types, save_model=False)


# # CLASSIFICATION ZFP PCA MINIMAL

run_zfp_pca_classification_minimal_datasets(benchmark_datasets, model_types, features_combinations, test_sizes, minimal_start_idx=0, minimal_iter=10, trial_start_idx=0, trial_iter=10, pca_arguments = [60], data_types = [np.float32], tolerance_list=tolerance_list, parallel=True)




# # SAVING MINIMAL IMAGE DATASETS

run_save_minimal_image_datasets(benchmark_datasets)


# # MEMORY USAGE
run_get_size(benchmark_datasets)



# CBIR

run_cbir_caltech256()


