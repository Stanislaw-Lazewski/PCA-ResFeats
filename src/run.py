import os
import numpy as np
from shutil import copy2
from tqdm import tqdm
import csv


from extract_resfeats import extract_features
from utils import (combine_layers_outputs, convert_to_npy, prepare_minimal_dataset_folder,
                   split_dataset, save_minimal_image_datasets, get_size, compute_cbir_caltech256_iterations_mean, 
                   compute_cbir_caltech256_iterations_mean_top_n, compute_cbir_caltech256_iterations_mean_mAP)
from sklearn_pca import make_pca
from classification import (linear_svm_classification_datasets, apply_features_on_X,
                            apply_pca_on_X, apply_subsampled_on_X, apply_random_subsampled_on_X,
                            linear_svm_classification_zfp_datasets)
from multithread import multithread_task
from cbir_caltech256 import run_multiple_caltech256_experiments
from top_n_cbir_caltech256 import run_multiple_caltech256_experiments_top_n


def run_features_extraction(datasets, model_types, preprocess, device):
    for dataset in datasets:
        for model_type in model_types:
            print("\nFeatures extraction: " + dataset + ", " + model_type)
            src_dir = os.path.join("data", dataset)
            dest_dir = os.path.join("features", dataset + "_" + model_type)
            extract_features(src_dir, dest_dir, preprocess, device, model_type)


def run_features_combination(datasets, model_types, layers_combinations):
    for dataset in datasets:
        for model_type in model_types:
            for layers_combination in layers_combinations:
                print("\nFeatures combination: " + dataset + ", " + model_type)
                src_dir = os.path.join("features", dataset + "_" + model_type)
                dest_dir = os.path.join("features", dataset + "_" + model_type, "_".join(layers_combination))
                combine_layers_outputs(src_dir, dest_dir, layers_combination)
            
            
def run_out_features_conversion_to_npy(datasets, model_types, X_npy_type = np.float32):
    for dataset in datasets:
        for model_type in model_types:
            feature_path = os.path.join("features", dataset + "_" + model_type)
            torch_path = os.path.join(feature_path, "out")
            print("\nOut features conversion to numpy (out.pt -> X.npy): " + dataset + ", " + model_type)
            if not os.path.exists(torch_path):
                os.makedirs(torch_path)
                copy2(os.path.join(feature_path, "out.pt"), os.path.join(torch_path, "X.pt"))
                copy2(os.path.join(feature_path, "y.pt"), os.path.join(torch_path, "y.pt"))
            npy_path = os.path.join(torch_path, X_npy_type.__name__, "X.npy") 
            torch_path = os.path.join(torch_path, "X.pt")
            convert_to_npy(torch_path, npy_path, X_npy_type)
                           
            torch_path = os.path.join("features", dataset + "_" + model_type, "out")
            if os.path.exists(torch_path): 
                print("\nOut labels conversion to numpy (y.pt -> y.npy): " + dataset + ", " + model_type)
                npy_path = os.path.join(torch_path, X_npy_type.__name__, "y.npy")
                torch_path = os.path.join(torch_path, "y.pt")
                # Only label type
                npy_type = np.int16
                convert_to_npy(torch_path, npy_path, npy_type)
            
            
def run_features_conversion_to_npy(datasets, model_types, layers_combinations, X_npy_type = np.float32):
    for dataset in datasets:
        for model_type in model_types:
            for layers_combination in layers_combinations:
                print("\nFeatures conversion to numpy (X.pt -> X.npy): " + dataset + ", " + model_type)
                torch_path = os.path.join("features", dataset + "_" + model_type, "_".join(layers_combination), "X.pt")
                npy_path = os.path.join("features", dataset + "_" + model_type, "_".join(layers_combination), X_npy_type.__name__, "X.npy")
                convert_to_npy(torch_path, npy_path, X_npy_type)
    
                print("\nLabels conversion to numpy (y.pt -> y.npy): " + dataset + ", " + model_type)
                torch_path = os.path.join("features", dataset + "_" + model_type, "_".join(layers_combination), "y.pt")
                npy_path = os.path.join("features", dataset + "_" + model_type, "_".join(layers_combination), X_npy_type.__name__, "y.npy")
                # Only label type
                npy_type = np.int16
                convert_to_npy(torch_path, npy_path, npy_type)
                

def run_datasets_minimal_balancing(datasets, model_types, features_combinations, minimal_start_idx=0, minimal_iter=10):
    for dataset in tqdm(datasets, desc="datasets", position=0):
        for model_type in tqdm(model_types, desc="model_types", position=1):
            for features_combination in tqdm(features_combinations, desc="features_combinations", position=2):
                print("\nBalancing : " + dataset + ", " + model_type + ", " + features_combination)
                for i in range(minimal_start_idx, minimal_start_idx + minimal_iter):
                    suffix_dir = os.path.join(dataset + "_" + model_type, features_combination)
                    src_dir = os.path.join("features", suffix_dir)
                    dest_dir = os.path.join("features_minimal", suffix_dir, "minimal_" + str(i))
                    prepare_minimal_dataset_folder(src_dir, dest_dir)
                        
                        
def run_splitting_minimal_datasets(datasets, model_types, features_combinations, test_sizes, minimal_start_idx=0, minimal_iter=10, trial_start_idx=0, trial_iter=10):
    for dataset in tqdm(datasets, desc="datasets", position=0):
        for model_type in tqdm(model_types, desc="model_types", position=1):
            for features_combination in tqdm(features_combinations, desc="features_combinations", position=2):
                print("\nSplitting : minimal, " + dataset + ", " + model_type + ", " + features_combination)
                for i in range(minimal_start_idx, minimal_start_idx + minimal_iter):
                    for j in range(trial_start_idx, trial_start_idx + trial_iter):
                        for test_size in test_sizes:
                            minimal_dir = os.path.join("features_minimal", dataset + "_" + model_type, features_combination, "minimal_" + str(i))
                            dest_dir = os.path.join(minimal_dir, f"split_{int(test_size*100)}_" + str(j))
                            split_dataset(os.path.join(minimal_dir, "dataset_minimal_idx.pkl"), dest_dir, test_size)
                        
    
def run_pca_minimal_datasets(datasets, model_types, features_combinations, test_sizes, minimal_start_idx=0, minimal_iter=10, trial_start_idx=0, trial_iter=10, pca_arguments = [60], data_types = [np.float16]):
    for dataset in tqdm(datasets, desc="datasets", position=0):
        for model_type in tqdm(model_types, desc="model_types", position=1):
            for features_combination in tqdm(features_combinations, desc="features_combinations", position=2):
                print("\nPCA : minimal, " + dataset + ", " + model_type + ", " + features_combination)
                for i in tqdm(range(minimal_start_idx, minimal_start_idx + minimal_iter), desc="minimal", position=3):
                    for j in range(trial_start_idx, trial_start_idx + trial_iter):
                        for test_size in test_sizes:
                            suffix_dir = os.path.join(dataset + "_" + model_type, features_combination, "minimal_" + str(i), f"split_{int(test_size*100)}_" + str(j))
                            minimal_dir = os.path.join("features_minimal", suffix_dir)
                            suffix_dir = os.path.join("pca_minimal", suffix_dir)
                            for pca_argument in pca_arguments:
                                for data_type in data_types:
                                    if pca_argument is not None:
                                        dest_dir = os.path.join(suffix_dir, f'pca_{int(pca_argument*100)}', data_type.__name__)
                                    else:
                                        dest_dir = os.path.join(suffix_dir, 'pca_None', data_type.__name__)
                                    full_dir = os.path.join("features", dataset + "_" + model_type, features_combination, data_type.__name__)
                                    make_pca(full_dir, minimal_dir, dest_dir, pca_argument)
        
        
def run_pca_classification_minimal_datasets(datasets, model_types, features_combinations, test_sizes, minimal_start_idx=0, minimal_iter=10, trial_start_idx=0, trial_iter=10, pca_arguments = [359], data_types = [np.float16], save_model=False):
    for dataset in datasets:
        
        result_dir = os.path.join("classification_results", "classification_pca_minimal", dataset)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir, "results.csv")
        if not os.path.exists(result_path):
            header = ["dataset", "model_type", "features_combination", "test_size", "minimal_iter", "trail_iter", "pca_argument", "data_type", "accuracy"]
            with open(result_path, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                
        data_params = []
    
        for model_type in model_types:
            for features_combination in features_combinations:
                for i in range(minimal_start_idx, minimal_start_idx + minimal_iter):
                    for j in range(trial_start_idx, trial_start_idx + trial_iter):
                        for test_size in test_sizes:
                            suffix_dir = os.path.join(dataset + "_" + model_type, features_combination, "minimal_" + str(i), f"split_{int(test_size*100)}_" + str(j))
                            minimal_dir = os.path.join("features_minimal", suffix_dir)
                            for pca_argument in pca_arguments:
                                for data_type in data_types:
                                    if pca_argument is not None:
                                        pca_dir = os.path.join(suffix_dir, f'pca_{int(pca_argument*100)}', data_type.__name__)
                                    else:
                                        pca_dir = os.path.join(suffix_dir, 'pca_None', data_type.__name__)   
                                    
                                    classification_dir = os.path.join("classification_pca_minimal", pca_dir, "default_linear")
                                    pca_dir = os.path.join("pca_minimal", pca_dir)
                                    full_dir = os.path.join("features", dataset + "_" + model_type, features_combination, data_type.__name__)
                                    
                                    data_params.append(([apply_pca_on_X, [pca_dir], full_dir, minimal_dir, classification_dir, save_model], [dataset, model_type, features_combination, test_size, i, j, pca_argument, data_type.__name__]))
        n = len(data_params)
        print("Dataset: " + dataset)
        print("SVM to train: " + str(n))
        multithread_task(data_params, result_path, n, linear_svm_classification_datasets)
        
     
def run_features_classification_minimal_datasets(datasets, model_types, features_combinations, test_sizes, minimal_start_idx=0, minimal_iter=10, trial_start_idx=0, trial_iter=10, data_types = [np.float16], save_model=False):
    for dataset in datasets:
        
        result_dir = os.path.join("classification_results", "classification_features_minimal", dataset)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir, "results.csv")
        if not os.path.exists(result_path):
            header = ["dataset", "model_type", "features_combination", "test_size", "minimal_iter", "trail_iter", "data_type", "accuracy"]
            with open(result_path, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                
        data_params = []
    
        for model_type in model_types:
            for features_combination in features_combinations:
                for i in range(minimal_start_idx, minimal_start_idx + minimal_iter):
                    for j in range(trial_start_idx, trial_start_idx + trial_iter):
                        for test_size in test_sizes:
                            suffix_dir = os.path.join(dataset + "_" + model_type, features_combination, "minimal_" + str(i), f"split_{int(test_size*100)}_" + str(j))
                            minimal_dir = os.path.join("features_minimal", suffix_dir)
                            for data_type in data_types:
                                classification_dir = os.path.join("classification_features_minimal", suffix_dir, data_type.__name__, "default_linear")
                                full_dir = os.path.join("features", dataset + "_" + model_type, features_combination, data_type.__name__)
                                
                                data_params.append(([apply_features_on_X, [], full_dir, minimal_dir, classification_dir, save_model], [dataset, model_type, features_combination, test_size, i, j, data_type.__name__]))
        n = len(data_params)
        print("Dataset: " + dataset)
        print("SVM to train: " + str(n))
        multithread_task(data_params, result_path, n, linear_svm_classification_datasets)
              

def run_subsampled_classification_minimal_datasets(datasets, model_types, features_combinations, test_sizes, minimal_start_idx=0, minimal_iter=10, trial_start_idx=0, trial_iter=10, every_ns = [10], data_types = [np.float16], save_model=False):
    for dataset in datasets:
        
        result_dir = os.path.join("classification_results", "classification_subsampled_minimal", dataset)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir, "results.csv")
        if not os.path.exists(result_path):
            header = ["dataset", "model_type", "features_combination", "test_size", "minimal_iter", "trail_iter", "every_n", "data_type", "accuracy"]
            with open(result_path, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                
        data_params = []
    
        for model_type in model_types:
            for features_combination in features_combinations:
                for i in range(minimal_start_idx, minimal_start_idx + minimal_iter):
                    for j in range(trial_start_idx, trial_start_idx + trial_iter):
                        for test_size in test_sizes:
                            suffix_dir = os.path.join(dataset + "_" + model_type, features_combination, "minimal_" + str(i), f"split_{int(test_size*100)}_" + str(j))
                            minimal_dir = os.path.join("features_minimal", suffix_dir)
                            for every_n in every_ns:
                                for data_type in data_types:
                                    classification_dir = os.path.join("classification_subsampled_minimal", suffix_dir, f'every_{every_n}', data_type.__name__, "default_linear")
                                    full_dir = os.path.join("features", dataset + "_" + model_type, features_combination, data_type.__name__)
                                    
                                    data_params.append(([apply_subsampled_on_X, [every_n], full_dir, minimal_dir, classification_dir, save_model], [dataset, model_type, features_combination, test_size, i, j, every_n, data_type.__name__]))
        n = len(data_params)
        print("Dataset: " + dataset)
        print("SVM to train: " + str(n))
        multithread_task(data_params, result_path, n, linear_svm_classification_datasets)
        
        
def run_random_subsampled_classification_minimal_datasets(datasets, model_types, features_combinations, test_sizes, minimal_start_idx=0, minimal_iter=10, trial_start_idx=0, trial_iter=10, random_subsamples = [359], data_types = [np.float16], save_model=False):
    for dataset in datasets:
        
        result_dir = os.path.join("classification_results", "classification_random_subsampled_minimal", dataset)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir, "results.csv")
        if not os.path.exists(result_path):
            header = ["dataset", "model_type", "features_combination", "test_size", "minimal_iter", "trail_iter", "random_subsample", "data_type", "accuracy"]
            with open(result_path, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                
        data_params = []
    
        for model_type in model_types:
            for features_combination in features_combinations:
                for i in range(minimal_start_idx, minimal_start_idx + minimal_iter):
                    for j in range(trial_start_idx, trial_start_idx + trial_iter):
                        for test_size in test_sizes:
                            suffix_dir = os.path.join(dataset + "_" + model_type, features_combination, "minimal_" + str(i), f"split_{int(test_size*100)}_" + str(j))
                            minimal_dir = os.path.join("features_minimal", suffix_dir)
                            for random_subsample in random_subsamples:
                                for data_type in data_types:
                                    classification_dir = os.path.join("classification_random_subsampled_minimal", suffix_dir, f'random_subsample_{random_subsample}', data_type.__name__, "default_linear")
                                    full_dir = os.path.join("features", dataset + "_" + model_type, features_combination, data_type.__name__)
                                    
                                    data_params.append(([apply_random_subsampled_on_X, [random_subsample, classification_dir], full_dir, minimal_dir, classification_dir, save_model], [dataset, model_type, features_combination, test_size, i, j, random_subsample, data_type.__name__]))
        n = len(data_params)
        print("Dataset: " + dataset)
        print("SVM to train: " + str(n))
        multithread_task(data_params, result_path, n, linear_svm_classification_datasets)
        
        
def run_zfp_pca_classification_minimal_datasets(datasets, model_types, features_combinations, test_sizes, minimal_start_idx=0, minimal_iter=10, trial_start_idx=0, trial_iter=10, pca_arguments = [60], data_types = [np.float32], tolerance_list=[1], parallel=True):
    for dataset in datasets:
        
        result_dir = os.path.join("zfp_classification_results", "zfp_classification_pca_minimal", dataset)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir, "results.csv")
        if not os.path.exists(result_path):
            header = ["dataset", "model_type", "features_combination", "test_size", "minimal_iter", "trail_iter", "pca_argument", "data_type", "length","accuracy", "zfp_tolerance", "zfp_ratio", "original_size", "compressed_size", "Frobenius_norm", "mean_abs_err", "max_abs_err"]
            with open(result_path, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                
        data_params = []
    
        for model_type in model_types:
            for features_combination in features_combinations:
                for i in range(minimal_start_idx, minimal_start_idx + minimal_iter):
                    for j in range(trial_start_idx, trial_start_idx + trial_iter):
                        for test_size in test_sizes:
                            suffix_dir = os.path.join(dataset + "_" + model_type, features_combination, "minimal_" + str(i), f"split_{int(test_size*100)}_" + str(j))
                            minimal_dir = os.path.join("features_minimal", suffix_dir)
                            for pca_argument in pca_arguments:
                                for data_type in data_types:
                                    if pca_argument is not None:
                                        pca_dir = os.path.join(suffix_dir, f'pca_{int(pca_argument*100)}', data_type.__name__)
                                    else:
                                        pca_dir = os.path.join(suffix_dir, 'pca_None', data_type.__name__)   
                                    
                                    pca_dir = os.path.join("pca_minimal", pca_dir)
                                    full_dir = os.path.join("features", dataset + "_" + model_type, features_combination, data_type.__name__)
                                    
                                    data_params.append(([apply_pca_on_X, [pca_dir], full_dir, minimal_dir, tolerance_list, parallel], [dataset, model_type, features_combination, test_size, i, j, pca_argument, data_type.__name__]))
        n = len(data_params)
        print("Dataset: " + dataset)
        print("SVM to train: " + str(n))
        multithread_task(data_params, result_path, n, linear_svm_classification_zfp_datasets)
        
                   
def run_save_minimal_image_datasets(datasets):
    for dataset in datasets:      
        dataset_info_dir=os.path.join("features", f"{dataset}_imagenet")
        minimal_split_dir=os.path.join("features_minimal", f"{dataset}_imagenet", "l2_l3_l4", "minimal_0", "split_20_0")
        dest_dir="data_minimal_dataset"
        save_minimal_image_datasets(dataset_info_dir, minimal_split_dir, dest_dir)
        

def run_get_size(datasets):
    for dataset in datasets: 
        print(dataset, get_size(os.path.join("data_minimal_dataset", f"{dataset}")), "MB")
        
        
def run_cbir_caltech256():
    random_state_iter = {
        "_0": 97,
        "_1": 7,
        "_2": 23,
        "_3": 13,
        "_4": 17,
        "_5": 29,
        "_6": 31,
        "_7": 59,
        "_8": 67,
        "_9": 79
        }

    for dataset_type in ["minimal", "full"]:
        X_path = os.path.join("features", "Caltech256_imagenet", "l2_l3_l4", "float16", "X.npy")
        y_path = os.path.join("features", "Caltech256_imagenet", "l2_l3_l4", "float16", "y.npy")
        dataset_info_path = os.path.join("features", "Caltech256_imagenet", "dataset_info.pkl")
        run_multiple_caltech256_experiments(X_path, y_path, dataset_info_path, dataset_type ="minimal", 
                                            random_state_iter=random_state_iter, pca_argument=60, n=20, npy_type=np.float16)
        compute_cbir_caltech256_iterations_mean(dataset_type= "minimal", num_of_iter = 10)
        run_multiple_caltech256_experiments_top_n(dataset_info_path, dataset_type = "minimal", random_state_iter=random_state_iter)
        compute_cbir_caltech256_iterations_mean_top_n(dataset_info_path= dataset_info_path, dataset_type= "minimal", num_of_iter = 10)
        compute_cbir_caltech256_iterations_mean_mAP(dataset_type= "minimal", num_of_iter = 10)

        
