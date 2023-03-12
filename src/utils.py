import torch
import os
import numpy as np
import shutil
import pickle
import pandas as pd
import random


def combine_layers_outputs(src_dir, dest_dir, layers_combination):
    os.makedirs(dest_dir, exist_ok=True)
    features = torch.hstack([torch.load(
        os.path.join(src_dir, layer + ".pt")) for layer in layers_combination])
    
    torch.save(features, 
               os.path.join(dest_dir, "X.pt"))
    
    shutil.copy2(os.path.join(src_dir, "y.pt"), os.path.join(dest_dir, "y.pt"))
    

def convert_to_npy(torch_path, npy_path, npy_type):
    # npy_type: np.int16, np.float16
    dir_npy = os.path.dirname(npy_path)
    if not os.path.exists(dir_npy):
        os.makedirs(dir_npy)
    with open(npy_path,'wb') as npy_file:
        np.save(npy_file, torch.load(torch_path).numpy().astype(npy_type))
        

def prepare_minimal_dataset_folder(src_dir, dest_dir):
    y = torch.load(os.path.join(src_dir, "y.pt")).numpy().astype(np.int16)
        
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    num_of_classes = len(np.unique(y))
    occurrences = dict()
    minimal = dict()
    for i in range(num_of_classes):
        occurrences[i]=[]
    
    for i in range(len(y)):
        occurrences[y[i][0]].append(i)
        
    min_cardinality = min([len(occurrences[key]) for key in occurrences.keys()])
    
    for i in range(num_of_classes):
        minimal[i] = random.sample(occurrences[i], min_cardinality)
        
    with open(os.path.join(dest_dir, "dataset_minimal_idx.pkl"), 'wb') as pickle_file:
        pickle.dump(minimal, pickle_file)
        
    
def split_dataset(dataset_idx_dict_path, dest_dir, test_size = 0.2):
    with open(dataset_idx_dict_path, 'rb') as pickle_file:
        dataset_idx_dict =  pickle.load(pickle_file)
        
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    train_idx = []
    test_idx = []
    for key in dataset_idx_dict:
        sampled_indices = random.sample(dataset_idx_dict[key], int(test_size*len(dataset_idx_dict[key])))
        test_idx += sampled_indices
        train_idx += list(set(dataset_idx_dict[key]).symmetric_difference(set(sampled_indices)))
    
    with open(os.path.join(dest_dir, "test_idx_list.pkl"), 'wb') as pickle_file:
        pickle.dump(test_idx, pickle_file)
        
    with open(os.path.join(dest_dir, "train_idx_list.pkl"), 'wb') as pickle_file:
        pickle.dump(train_idx, pickle_file)
    


def save_minimal_image_datasets(dataset_info_dir, minimal_split_dir, dest_dir="data_minimal_dataset"):     
    with open(os.path.join(minimal_split_dir, "train_idx_list.pkl"), 'rb') as pickle_file:
        train_idx = pickle.load(pickle_file)

    with open(os.path.join(dataset_info_dir, "dataset_info.pkl"), 'rb') as pickle_file:
        dataset_info = pickle.load(pickle_file)
 
    for idx in train_idx:
        old_path = dataset_info["samples"][idx][0]
        new_path = os.path.join(dest_dir, '/'.join(old_path.split("/")[1:]))
        new_path_dir = os.path.dirname(new_path)
        if not os.path.exists(new_path_dir):
            os.makedirs(new_path_dir)
        shutil.copy2(old_path, new_path)
        

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return round(total_size/1000/1000, 3)
    
    
def compute_cbir_caltech256_iterations_mean(dataset_type= "minimal", num_of_iter = 10):
    path = f'cbir/caltech256/{dataset_type}/pca60_l2_l3_l4_i'
    AP_list=[]
    for i in range(num_of_iter):
        full_path=path+str(i)+'/AP_summary.csv'
        summary= pd.read_csv(full_path)
        AP_list.append(summary["AP"])
    mean = sum(AP_list)/num_of_iter
    summary= pd.read_csv(f'cbir/caltech256/{dataset_type}/pca60_l2_l3_l4_i0/AP_summary.csv')
    summary["AP"] = mean
    save_path = f'cbir/caltech256/{dataset_type}/pca60_l2_l3_l4_summary'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    summary.to_csv(os.path.join(save_path, 'AP_summary.csv'), index=False)   
    
    
def compute_cbir_caltech256_iterations_mean_top_n(dataset_info_path, dataset_type= "minimal", num_of_iter = 10):
    path = f'cbir/caltech256/{dataset_type}/pca60_l2_l3_l4_i'
    tmp_dict = dict()
    all_top_k = pickle.load(open(os.path.join(path+"0", 'all_top_k.pkl'), 'rb'))
    top_k_list = all_top_k[0].keys()
    for key in all_top_k:
        tmp_dict[key] = []
    for i in range(num_of_iter):
        full_path=os.path.join(path+str(i), 'all_top_k.pkl')
        all_top_k = pickle.load(open(full_path, 'rb'))
        for key in all_top_k:
            tmp_dict[key].append(list(map(lambda x: x[2], all_top_k[key].values())))

    mean=[]
    std=[]
    for key in tmp_dict:
        mean.append(np.mean(np.array(tmp_dict[key]), axis=0))
        std.append(np.std(np.array(tmp_dict[key]), axis=0))
        
    dataset_info =  pickle.load(open(dataset_info_path, 'rb'))
    class_names = dataset_info["classes"]+["MAP over class", "MAP overall"]
    summary_dict=dict()
    for key_class in tmp_dict:
        summary_dict[(key_class, class_names[key_class])] = dict()
        for key_top_k in top_k_list:
            summary_dict[(key_class, class_names[key_class])]["AP@"+str(key_top_k)] = dict()
            summary_dict[(key_class, class_names[key_class])]["AP@"+str(key_top_k)]["mean"] = mean[key_class][key_top_k-1]
            summary_dict[(key_class, class_names[key_class])]["AP@"+str(key_top_k)]["std"] = std[key_class][key_top_k-1]
        
    save_path = f'cbir/caltech256/{dataset_type}/pca60_l2_l3_l4_summary'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    pickle.dump(summary_dict, open(os.path.join(save_path, 'all_top_k.pkl'), 'wb'))


def compute_cbir_caltech256_iterations_mean_mAP(dataset_type= "minimal", num_of_iter = 10):
    path = f'cbir/caltech256/{dataset_type}/pca60_l2_l3_l4_i'
    mAP_list=[]
    for i in range(num_of_iter):
        full_path=os.path.join(path+str(i),'mAP.txt')
        f = open(full_path, "r")
        mAP= float(f.readline())
        f.close()
        mAP_list.append(mAP)
         
    save_path = f'cbir/caltech256/{dataset_type}/pca60_l2_l3_l4_summary'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f = open(os.path.join(save_path, "mAP.txt"), "w")
    f.write(f'Mean mAP: {np.mean(mAP_list)}\n')
    f.write(f'Std mAP: {np.std(mAP_list)}')
    f.close()
    
        
    