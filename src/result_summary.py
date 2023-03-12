import pandas as pd
import os 
import matplotlib.pyplot as plt
import numpy as np

variants_to_args_names = {
    "pca": "pca_argument",
    "subsampled": "every_n",
    "random_subsampled": "random_subsample"
    }

benchmark_datasets = {
    "Flowers": "flowers",
    "Caltech101": "caltech101",
    "Caltech256": "caltech256",
    "Catordog": "catordog",
    "EuroSAT": "eurosat",
    "MLC2008": "mlc2008",
    }


benchmark_datasets_full_names = {
    "Flowers": "Oxford Flowers",
    "Caltech101": "Caltech-101",
    "Caltech256": "Caltech-256",
    "Catordog": "Dog vs Cat",
    "EuroSAT": "EuroSAT",
    "MLC2008": "MLC2008",
    }


components_to_n ={
    359: 10,
    180: 20,
    120: 30,
    90: 40,
    72: 50,
    60: 60,
    52: 70,
    45: 80,
    40: 90, 
    36: 100
    }


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

def save_variants_results_minimal(datasets, variants_to_args_names, src_dir='classification_results', dest_dir='summary_results'):
    for dataset in datasets:
        for variant in variants_to_args_names:
            df = pd.read_csv(os.path.join(src_dir, 'classification_'+variant+'_minimal', dataset, 'results.csv'))
            
            grouped_mean = df.groupby([variants_to_args_names[variant], 'features_combination', 'data_type']).mean()
            grouped_mean = grouped_mean.drop(columns=['test_size', 'minimal_iter', 'trail_iter'])
            grouped_mean.rename(columns={'accuracy': 'accuracy_mean'}, inplace=True)
            
            grouped_std = df.groupby([variants_to_args_names[variant], 'features_combination', 'data_type']).std()
            grouped_std = grouped_std.drop(columns=['test_size', 'minimal_iter', 'trail_iter'])
            grouped_std.rename(columns={'accuracy': 'accuracy_std'}, inplace=True)
            
            result = pd.concat([grouped_mean, grouped_std], axis=1, join="inner")
            
            save_dir = os.path.join(dest_dir, dataset)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            with open(os.path.join(save_dir, f"{variant}_mean_std_acc_summary.txt"),'w') as fd:
                fd.write(result.to_string())
        
        
def save_features_results_minimal(datasets, src_dir='classification_results', dest_dir='summary_results'):
    for dataset in datasets:
        df = pd.read_csv(os.path.join(src_dir, 'classification_features_minimal', dataset, 'results.csv'))
        
        grouped_mean = df.groupby(['features_combination', 'data_type']).mean()
        grouped_mean = grouped_mean.drop(columns=['test_size', 'minimal_iter', 'trail_iter'])
        grouped_mean.rename(columns={'accuracy': 'accuracy_mean'}, inplace=True)
        
        grouped_std = df.groupby(['features_combination', 'data_type']).std()
        grouped_std = grouped_std.drop(columns=['test_size', 'minimal_iter', 'trail_iter'])
        grouped_std.rename(columns={'accuracy': 'accuracy_std'}, inplace=True)
        
        result = pd.concat([grouped_mean, grouped_std], axis=1, join="inner")
        
        save_dir = os.path.join(dest_dir, dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        with open(os.path.join(save_dir, "features_mean_std_acc_summary.txt"),'w') as fd:
            fd.write(result.to_string())
        
        
def plot_all_variants_summary(datasets, variants_to_args_names, n__to_components, src_dir='classification_results', dest_dir=os.path.join("plots", "all_variants_summary")):
    # Variants
    variants = ['features', 'pca',]
    markers = ["X", "o"]
    # Features_combinations
    features_combinations = ["l2_l3", "out", "l2_l4", "l3_l4", "l2_l3_l4"]
    colors = ["k", "b", "g", "r", "c"]
    
    for dataset in datasets:
        
        plt.figure(figsize=(8,6))
        x = [1]+sorted(n__to_components.keys())
        y = np.full((len(variants), len(features_combinations), len(x)), np.nan)
    
        for k in range(len(variants)):
            df = pd.read_csv(os.path.join(src_dir, 'classification_'+variants[k]+'_minimal', dataset, 'results.csv'))
            if variants[k]=='features':
                grouped_mean = df.groupby(['features_combination', 'data_type']).mean()
            else:
                grouped_mean = df.groupby([variants_to_args_names[variants[k]], 'features_combination', 'data_type']).mean()
            grouped_mean = grouped_mean.drop(columns=['test_size', 'minimal_iter', 'trail_iter'])
            grouped_mean.rename(columns={'accuracy': 'accuracy_mean'}, inplace=True)
            grouped_mean1 = grouped_mean.to_dict()["accuracy_mean"]
            
            for j in range(len(features_combinations)):
                if variants[k] == "features":
                    y[0, j, 0] = grouped_mean1[(features_combinations[j], 'float16')]
                else:
                    for i in range(1,len(x)):
                        y[k, j, i] = grouped_mean1[(x[i]if variants[k] == "subsampled" else n__to_components[x[i]]  , features_combinations[j], 'float16')]
                    
                plt.scatter(x, y[k, j, :], color = colors[j], marker = markers[k], label = str(variants[k]+"_"+features_combinations[j]))
            
        
        plt.title(f"Comparison of PCA variants results on {benchmark_datasets_full_names[dataset]}")                
        plt.xlabel(r'Reduction of the feature vector $\it{t}$')
        plt.ylabel('Classification accuracy')
        plt.fill_between(range(-5, 105), max(y[0,1,0]-0.05, 0), min(y[0,1,0]+0.05, 1), alpha=0.2)
        plt.ylim(0.63, 1.01)
        plt.xlim(-4, 104)
        if dataset=="MLC2008":
            plt.legend()
        plt.xticks(x)
        plt.grid(True)
        save_path = dest_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        plt.savefig(os.path.join(save_path, dataset+ "_pca.png"))
                                                 
            
def plot_all_methods_summary(datasets, variants_to_args_names, n__to_components, src_dir='classification_results', dest_dir=os.path.join("plots", "all_variants_summary")):
    # Variants
    variants = ['features', 'pca', 'subsampled', 'random_subsampled']
    colors = ["k", "b", "g", "r"]
    # Features_combinations
    features_combinations = ["l2_l3", "out", "l2_l3_l4"]
    markers = ["X", "o", "+"]
    
    for dataset in datasets:
        plt.figure(figsize=(8,6))
        x = [1]+sorted(n__to_components.keys())
        y = np.full((len(variants), len(features_combinations), len(x)), np.nan)
    
        for k in range(len(variants)):
            df = pd.read_csv(os.path.join(src_dir, 'classification_'+variants[k]+'_minimal', dataset, 'results.csv'))
            if variants[k]=='features':
                grouped_mean = df.groupby(['features_combination', 'data_type']).mean()
            else:
                grouped_mean = df.groupby([variants_to_args_names[variants[k]], 'features_combination', 'data_type']).mean()
            grouped_mean = grouped_mean.drop(columns=['test_size', 'minimal_iter', 'trail_iter'])
            grouped_mean.rename(columns={'accuracy': 'accuracy_mean'}, inplace=True)
            grouped_mean1 = grouped_mean.to_dict()["accuracy_mean"]

            for j in range(len(features_combinations)):
                if variants[k] == "features":
                    y[0, j, 0] = grouped_mean1[(features_combinations[j], 'float16')]
                else:
                    for i in range(1,len(x)):
                        y[k, j, i] = grouped_mean1[(x[i]if variants[k] == "subsampled" else n__to_components[x[i]]  , features_combinations[j], 'float16')]
                    
                plt.scatter(x, y[k, j, :], color = colors[k], marker = markers[j], label = str(variants[k]+"_"+features_combinations[j]))
            
        plt.title(f"Comparison of reduction methods results on {benchmark_datasets_full_names[dataset]}")    
        plt.xlabel(r'Reduction of the feature vector $\it{t}$')
        plt.ylabel('Classification accuracy')
        plt.fill_between(range(-5, 105), max(y[0,1,0]-0.05, 0), min(y[0,1,0]+0.05, 1), alpha=0.2)
        plt.ylim(0.63, 1.01)
        plt.xlim(-4, 104)
        if dataset=="MLC2008":
            plt.legend()
        plt.xticks(x)
        plt.grid(True)
        save_path = dest_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        plt.savefig(os.path.join(save_path, dataset+ "_methods.png"))
                                                 
                       
def save_zfp_pca_results_minimal(datasets, src_dir='zfp_classification_results', dest_dir='summary_results', save_std=False):
    for dataset in datasets:
        df = pd.read_csv(os.path.join(src_dir, 'zfp_classification_pca_minimal', dataset, 'results.csv'))
        
        grouped_mean = df.groupby(['features_combination', 'pca_argument', 'length', 'data_type', 'zfp_tolerance']).mean()
        grouped_mean = grouped_mean.drop(columns=['test_size', 'minimal_iter', 'trail_iter', 'original_size', 'compressed_size'])
        grouped_mean.rename(columns={'accuracy': 'accuracy_mean', 'zfp_ratio': 'zfp_ratio_mean', 'Frobenius_norm': 'Frobenius_norm_mean', 'mean_abs_err': 'mean_abs_err_mean', 'max_abs_err': 'max_abs_err_mean'}, inplace=True)

        result = grouped_mean[['accuracy_mean', 'zfp_ratio_mean', 'Frobenius_norm_mean', 'mean_abs_err_mean', 'max_abs_err_mean']]

        save_name = "zfp_pca_mean_acc_summary"

        if save_std:
            grouped_std = df.groupby(['features_combination', 'pca_argument', 'length', 'data_type', 'zfp_tolerance']).std()
            grouped_std = grouped_std.drop(columns=['test_size', 'minimal_iter', 'trail_iter', 'original_size', 'compressed_size'])
            grouped_std.rename(columns={'accuracy': 'accuracy_std', 'zfp_ratio': 'zfp_ratio_std', 'Frobenius_norm': 'Frobenius_norm_std', 'mean_abs_err': 'mean_abs_err_std', 'max_abs_err': 'max_abs_err_std'}, inplace=True)
            
            result = pd.concat([grouped_mean, grouped_std], axis=1, join="inner")
            result = result[['accuracy_mean', 'accuracy_std', 'zfp_ratio_mean', 'zfp_ratio_std', 'Frobenius_norm_mean', 'Frobenius_norm_std', 'mean_abs_err_mean', 'mean_abs_err_std', 'max_abs_err_mean', 'max_abs_err_std']]
        
            save_name+="_with_std"
             
        save_dir = os.path.join(dest_dir, dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        with open(os.path.join(save_dir, save_name + ".txt"),'w') as fd:
            fd.write(result.to_string())
            

def plot_zfp_per_resfeats_combination(datasets, src_dir='zfp_classification_results', plot_dir=os.path.join('plots', 'zfp')):
    results = {}
    for dataset in datasets:
        df = pd.read_csv(os.path.join(src_dir, 'zfp_classification_pca_minimal', dataset, 'results.csv'))
        
        grouped_mean = df.groupby(['features_combination', 'pca_argument', 'length', 'data_type', 'zfp_tolerance']).mean()
        grouped_mean = grouped_mean.drop(columns=['test_size', 'minimal_iter', 'trail_iter', 'original_size', 'compressed_size', 'zfp_ratio', 'Frobenius_norm', 'mean_abs_err', 'max_abs_err'])
        grouped_mean.rename(columns={'accuracy': 'accuracy_mean'}, inplace=True)

        grouped_std = df.groupby(['features_combination', 'pca_argument', 'length', 'data_type', 'zfp_tolerance']).std()
        grouped_std = grouped_std.drop(columns=['test_size', 'minimal_iter', 'trail_iter', 'original_size', 'compressed_size', 'zfp_ratio', 'Frobenius_norm', 'mean_abs_err', 'max_abs_err'])
        grouped_std.rename(columns={'accuracy': 'accuracy_std'}, inplace=True)
        
        result = pd.concat([grouped_mean, grouped_std], axis=1, join="inner")
        result = result[['accuracy_mean', 'accuracy_std']]

        results[dataset] = result
        
    dataset_names = ("Catordog", "Flowers", "Caltech256", "Caltech101", "EuroSAT", "MLC2008")
    features_combinations = ["out", "l2_l3_l4", "l2_l3", "l2_l4", "l3_l4"]
    pca_argument = 60
    length = 60
    data_type = "float32"
    zfp_tolerance = 10.0
    
    dataset_names_means = {'pca60_'+features_combination:tuple(map(lambda x: np.round(100*results[x]["accuracy_mean"].loc[(features_combination, pca_argument, length, data_type, zfp_tolerance)], 2), dataset_names)) for features_combination in features_combinations}
    
    dataset_names_std = {'pca60_'+features_combination:tuple(map(lambda x: np.round(100*results[x]["accuracy_std"].loc[(features_combination, pca_argument, length, data_type, zfp_tolerance)], 2), dataset_names)) for features_combination in features_combinations}
    
    dataset_names_means_original = {'pca60_'+features_combination:tuple(map(lambda x: np.round(100*results[x]["accuracy_mean"].loc[(features_combination, pca_argument, length, data_type, 0.0)], 2), dataset_names)) for features_combination in features_combinations}

    x = np.arange(len(dataset_names))*2  # The label locations
    width = 0.3  # The width of the bars
    multiplier = 0
    
    fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 8.5))
    plt.grid(axis='x')
    
    for attribute, measurement in dataset_names_means.items():
        offset = width * multiplier
        rects = ax.barh(x + offset, measurement, width, xerr=dataset_names_std[attribute] , label=attribute)
        ax.bar_label(rects, labels=[str(x)+" ("+str(y)+")" for x,y in zip(dataset_names_means[attribute], dataset_names_means_original[attribute])], padding=3)
        multiplier += 1
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Accuracy [%]')
    ax.set_title('Accuracy on decompressed zfp files by datasets')
    ax.set_yticks(x + width +0.3, [benchmark_datasets_full_names[dataset_name] for dataset_name in dataset_names])
    ax.legend(loc='upper right')
    ax.set_xlim(55, 106)
    ax.set_xticks(list(range(55,105,5)))
    ax.set_axisbelow(True)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, "zfp_pca60_variants_all_datasets_with_original.png"))
    
    
            
plot_zfp_per_resfeats_combination(benchmark_datasets_full_names, src_dir='zfp_classification_results', plot_dir=os.path.join("plots", "zfp"))

# PLOTS
plot_all_methods_summary(benchmark_datasets_full_names, variants_to_args_names, n__to_components, src_dir='classification_results', dest_dir=os.path.join("plots", "all_methods_summary_final"))
            
plot_all_variants_summary(benchmark_datasets_full_names, variants_to_args_names, n__to_components, src_dir='classification_results', dest_dir=os.path.join("plots", "all_variants_summary_final"))    
        

# SUMMARY RESULTS
save_variants_results_minimal(benchmark_datasets, variants_to_args_names, src_dir='classification_results', dest_dir='summary_results')
save_features_results_minimal(benchmark_datasets, src_dir='classification_results', dest_dir='summary_results')


# ZFP results
save_zfp_pca_results_minimal(benchmark_datasets, src_dir='zfp_classification_results', dest_dir='summary_results', save_std=False)
save_zfp_pca_results_minimal(benchmark_datasets, src_dir='zfp_classification_results', dest_dir='summary_results', save_std=True)