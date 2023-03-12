import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm


def make_cbir_caltech256_top_n(src_dest_dir, dataset_info_path):
    with open(os.path.join(src_dest_dir, 'X_train_pca.npy'), 'rb') as f:
        X_train = np.load(f)
        
    with open(os.path.join(src_dest_dir, 'y_train.npy'), 'rb') as f:
        y_train = np.load(f)
        
    with open(os.path.join(src_dest_dir, 'X_test_pca.npy'), 'rb') as f:
        X_test = np.load(f)
        
    with open(os.path.join(src_dest_dir, 'train_idx.npy'), 'rb') as f:
        train_idx = np.load(f)
        
    with open(os.path.join(src_dest_dir, 'test_idx.npy'), 'rb') as f:
        test_idx = np.load(f)

    dataset_info =  pickle.load(open(dataset_info_path, 'rb'))
        
    train_class_num  = np.bincount(y_train.squeeze())

    top_n=dict()
        
    for idx, query in enumerate(tqdm(X_test)):
            
        dists = np.linalg.norm(X_train - query, axis=1)
        
        query_class = dataset_info["samples"][test_idx[idx]][1]
                
        # Extract n images that have lowest distance
        ids = np.argsort(dists)[:train_class_num[query_class]]
        scores = [(dists[i], *dataset_info["samples"][train_idx[i]]) for i in ids]
        top_n[(idx, test_idx[idx], *dataset_info["samples"][test_idx[idx]])] = scores
           
    pickle.dump(top_n, open(os.path.join(src_dest_dir, 'top_n.pkl'), 'wb'))
                                         

def get_three_sum_formula_mAP(src_dest_dir):
    top_n =  pickle.load(open(os.path.join(src_dest_dir, 'top_n.pkl'), 'rb'))
    keys = list(top_n.keys())
    mAP=0
    Q=len(keys)
    for i in tqdm(range(Q)):
        query_class= keys[i][3]
        m=len(top_n[keys[i]])
        example_classes = list(map(lambda x: x[2], top_n[keys[i]]))
        after_second_sum = 0
        for k in range(1,m):
            precision_at_k = (example_classes[:k].count(query_class))/k
            after_second_sum += precision_at_k
        mAP += after_second_sum/m
    mAP /= Q
    
    return mAP
        

def count_top_n(src_dest_dir, num_classes=256):
    top_n =  pickle.load(open(os.path.join(src_dest_dir, 'top_n.pkl'), 'rb'))
    keys = list(top_n.keys())
    num_examples_per_query_list = list(map(lambda x: len(x), top_n.values()))
    max_n = min(num_examples_per_query_list)
    
    responses_per_classes = dict()
    for i in range(num_classes):
        responses_per_classes[i]=[]
    
    for i in range(len(keys)):
        query_class = keys[i][3]
        right_example_class = list(map(lambda x: 1 if x[2]==query_class else 0, top_n[keys[i]]))
        responses_per_classes[query_class].append(right_example_class)
    
    AP_at_k_per_classes=dict()
    for i in range(num_classes):
        AP_per_k=dict()
        for k in range(1, max_n+1):
            sum_=sum(list(map(lambda x: sum(x[:k]), responses_per_classes[i])))
            len_=k*len(responses_per_classes[i])
            AP_per_k[k] = (sum_, len_, sum_/len_)
        AP_at_k_per_classes[i] = AP_per_k
        
    AP_overall=dict()
    AP_over_class=dict()
    for k in range(1, max_n+1):
        sum_=0
        len_=0
        sum_AP=0
        for i in range(num_classes):
            sum_+=AP_at_k_per_classes[i][k][0]
            len_+=AP_at_k_per_classes[i][k][1]
            sum_AP+=AP_at_k_per_classes[i][k][2]
        AP_overall[k]=(sum_, len_, sum_/len_)
        AP_over_class[k]=(sum_AP, num_classes, sum_AP/num_classes)
        
    AP_at_k_per_classes[num_classes]=AP_over_class
    AP_at_k_per_classes[num_classes+1]=AP_overall
    
    pickle.dump(AP_at_k_per_classes, open(os.path.join(src_dest_dir, 'all_top_k.pkl'), 'wb'))
    
    

# unused
def save_plot_AP_per_k(src_dest_dir, dataset_info_path, id_class=257):
    dataset_info =  pickle.load(open(dataset_info_path, 'rb'))
    AP_at_k_per_classes =  pickle.load(open(os.path.join(src_dest_dir, 'all_top_k.pkl'), 'rb'))
    x = list(AP_at_k_per_classes[id_class].keys())
    y = list(map(lambda x: x[2], AP_at_k_per_classes[id_class].values()))
    if id_class==256:
        title = "MAP over class"
    elif id_class==257:
        title = "MAP overall"
    else:
        title = f'AP for {dataset_info["classes"][id_class]}'
        
    plt.clf() 
    plt.plot(x, y, 'ro')
    plt.axis([0, max(x)+1, 0, 1])
    plt.xlabel("k")
    plt.ylabel("Metric value @k")
    plt.title(title)
    plt.savefig(os.path.join(src_dest_dir, f'{"_".join(title.split(" "))}.png'))
    plt.clf() 


def run_multiple_caltech256_experiments_top_n(dataset_info_path, dataset_type, random_state_iter):
    
    dataset_name = "caltech256"    
    pca_argument=60
    
    for suffix in tqdm(random_state_iter.keys()):
        src_dest_dir = f'cbir/{dataset_name}/{dataset_type}/pca{pca_argument}_l2_l3_l4_i{suffix[1:]}'
        
        make_cbir_caltech256_top_n(src_dest_dir, dataset_info_path)
        
        f = open(os.path.join(src_dest_dir, "mAP.txt"), "w")
        f.write(str(get_three_sum_formula_mAP(src_dest_dir)))
        f.close()
        
        count_top_n(src_dest_dir, num_classes=256)
        
    
def save_plot_MAP_per_k(src_dest_dir, dataset_info_path, id_class=257):
    src_dest_dir= 'cbir/caltech256'
    dataset_info_path="cbir_caltech256_data/dataset_info.pkl"
    id_class=257
    dataset_info =  pickle.load(open(dataset_info_path, 'rb'))
    AP_at_k_per_classes_full =  pickle.load(open(os.path.join('cbir/caltech256/full/pca60_l2_l3_l4_summary', 'all_top_k.pkl'), 'rb'))
    AP_at_k_per_classes_minimal =  pickle.load(open(os.path.join('cbir/caltech256/minimal/pca60_l2_l3_l4_summary', 'all_top_k.pkl'), 'rb'))
    class_dict_id = list(AP_at_k_per_classes_full.keys())[id_class]
    class_value_list = list(AP_at_k_per_classes_full[class_dict_id])
    x = list(map(lambda x: int(x[3:]), class_value_list))
    y_full = list(map(lambda x: x["mean"], AP_at_k_per_classes_full[class_dict_id].values()))
    y_minimal = list(map(lambda x: x["mean"], AP_at_k_per_classes_minimal[class_dict_id].values()))
    if id_class==256:
        title = "MAP over class"
    elif id_class==257:
        title = "MAP overall"
    else:
        title = f'AP for {dataset_info["classes"][id_class]}'
        
    plt.clf() 
    plt.plot(x, y_full, 'ro', color="red")
    plt.plot(x, y_minimal, 'ro', color="blue")
    plt.xlim(0, int(x[-1])+1)
    plt.ylim(0, 1)
    plt.xlabel("k")
    plt.ylabel("Metric value @k")
    plt.title(title)
    plt.xticks([1]+list(range(4, 50, 4)))
    plt.yticks([i/10 for i in range(10)])
    plt.grid(axis='both')
    plt.legend(["full", "minimal"])
    plt.savefig(os.path.join(src_dest_dir, f'{id_class}_{"_".join(title.split(" "))}.png'))
    plt.clf() 
