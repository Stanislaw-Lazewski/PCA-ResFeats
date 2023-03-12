import os
import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


def apply_pca_on_caltech256(X_path, y_path, dest_dir, minimal=False, 
                            pca_argument=60, random_state=97, npy_type = np.float16):
    with open(X_path, 'rb') as f:
        X = np.load(f)
        
    with open(y_path, 'rb') as f:
        y = np.load(f)

    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    if minimal:
        # get 80 random sample for each class and 
        # split train test with 0.4 test_split
        
        num_of_classes = len(np.unique(y))
        occurrences = dict()
        for i in range(num_of_classes):
            occurrences[i]=[]
        
        for i in range(len(y)):
            occurrences[y[i][0]].append(i)
        
        train_idx = []
        test_idx = []
        for i in range(num_of_classes):
            random.seed(random_state)
            sampled_indices = random.sample(occurrences[i], 80)
            train_idx += sampled_indices[:int(80*0.6)]
            test_idx += sampled_indices[int(80*0.6):]
        
    else:
    
        # split train test with 0.4 test_split
        train_idx, test_idx, _, _ = train_test_split(list(range(len(X))), y, random_state=random_state, test_size=0.4, stratify=y)
        
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]


    with open(os.path.join(dest_dir, 'X_train.npy'),'wb') as npy_file:
        np.save(npy_file, X_train.astype(npy_type))
        
    with open(os.path.join(dest_dir, 'X_test.npy'),'wb') as npy_file:
        np.save(npy_file, X_test.astype(npy_type))
        
    with open(os.path.join(dest_dir, 'y_train.npy'),'wb') as npy_file:
        np.save(npy_file, y_train.astype(np.int16))
        
    with open(os.path.join(dest_dir, 'y_test.npy'),'wb') as npy_file:
        np.save(npy_file, y_test.astype(np.int16))

    with open(os.path.join(dest_dir, 'test_idx.npy'),'wb') as npy_file:
        np.save(npy_file, np.array(test_idx, dtype=np.int16))
        
    with open(os.path.join(dest_dir, 'train_idx.npy'),'wb') as npy_file:
        np.save(npy_file, np.array(train_idx, dtype=np.int16))
        

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    
    pickle.dump(sc, open(os.path.join(dest_dir, 'standard_scaler.sav'), 'wb'))
    

    if pca_argument is not None:
        if pca_argument < 1:
            pca = PCA(pca_argument)
        else:
            pca = PCA(min(pca_argument, X_train.shape[0], X_train.shape[1]))
    else:
        pca = PCA()
    X_train = pca.fit_transform(X_train)
    
    
    pickle.dump(pca, open(os.path.join(dest_dir, 'pca_model.sav'), 'wb'))
    
    X_test = sc.transform(X_test)
    X_test = pca.transform(X_test)
    
    
    with open(os.path.join(dest_dir, 'X_train_pca.npy'),'wb') as npy_file:
        np.save(npy_file, X_train.astype(npy_type))
        
    with open(os.path.join(dest_dir, 'X_test_pca.npy'),'wb') as npy_file:
        np.save(npy_file, X_test.astype(npy_type))
    
    save_path = os.path.join(dest_dir, "pca_report.txt")
    with open(save_path,'a') as fd:
        fd.write(f'Explained variance: {sum(pca.explained_variance_ratio_)}\n')
        fd.write(f'Number of components: {pca.n_components_}\n')
        fd.write(f'Number of features: {pca.n_features_}\n')
        fd.write(f'Number of samples: {pca.n_samples_}\n\n\n')
        


def make_cbir_caltech256(src_dest_dir, dataset_info_path, n=20):
    with open(os.path.join(src_dest_dir, 'X_train_pca.npy'), 'rb') as f:
        X_train = np.load(f)
        
    with open(os.path.join(src_dest_dir, 'X_test_pca.npy'), 'rb') as f:
        X_test = np.load(f)
        
    with open(os.path.join(src_dest_dir, 'train_idx.npy'), 'rb') as f:
        train_idx = np.load(f)
        
    with open(os.path.join(src_dest_dir, 'test_idx.npy'), 'rb') as f:
        test_idx = np.load(f)
    
   
    dataset_info =  pickle.load(open(dataset_info_path, 'rb'))
    # Image indexes to save
    to_save = [100, 500, 1000, 1500]

    true_pred = dict()
    for i in range(256):
        true_pred[i]=[]
        
    for idx, query in enumerate(X_test):
            
        dists = np.linalg.norm(X_train - query, axis=1)
                
        # Extract n images that have lowest distance
        ids = np.argsort(dists)[:n]
        scores = [(dists[i], *dataset_info["samples"][train_idx[i]]) for i in ids]
        
        classes_id = list(map(lambda x: x[2], scores))
        
        query_id = dataset_info["samples"][test_idx[idx]][1]
        
        true_pred[query_id].append((idx, classes_id.count(query_id)))
        
        if idx in to_save:
            # Visualize the result
            axes=[]
            fig=plt.figure(figsize=(8,8))
            for a in range(n):
                score = scores[a]
                axes.append(fig.add_subplot(int(n/5), 5, a+1))
                subplot_title=str(round(score[0], 2))+" ["+dataset_info["classes"][score[2]]+"]"
                axes[-1].set_title(subplot_title)  
                plt.axis('off')
                plt.imshow(Image.open(score[1]))
            fig.tight_layout()
            save_path = os.path.join(src_dest_dir, "cbir")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, 'cbir_' + "-".join(dataset_info["samples"][test_idx[idx]][0].split("/")))) 
        
    pickle.dump(true_pred, open(os.path.join(src_dest_dir, 'true_pred.pkl'), 'wb'))
                        


def save_AP_summary(src_dest_dir, dataset_info_path, n=20):
    
    true_pred =  pickle.load(open(os.path.join(src_dest_dir, 'true_pred.pkl'), 'rb'))
    dataset_info =  pickle.load(open(dataset_info_path, 'rb'))
    sum_per_class = []
    len_per_class = []
    ap_per_class = []
    mean_list = [("class_id", "class_name", "AP")]
    for k in true_pred.keys():
        sum_ = sum(list(map(lambda x: int(x[1])/n, true_pred[k])))
        len_ = len(true_pred[k])
        ap_ = sum_ / len_
        sum_per_class.append(sum_)
        len_per_class.append(len_)
        ap_per_class.append(ap_)
        
        mean_list.append((k, dataset_info["classes"][k], ap_))
        
    mean_list.append(("X", "MAP over class", np.mean(ap_per_class)))
    mean_list.append(("X", "MAP overall", sum(sum_per_class)/sum(len_per_class)))
    
    np.savetxt(os.path.join(src_dest_dir, "AP_summary.csv"), mean_list, delimiter=",", fmt='%s')
    


def run_multiple_caltech256_experiments(X_path, y_path, dataset_info_path, dataset_type, random_state_iter, pca_argument=60, n=20, 
                                         npy_type=np.float16):
    
    dataset_name = "caltech256"    
    
    if dataset_type == "minimal":
        minimal = True
    if dataset_type == "full":
        minimal = False
    for suffix in tqdm(random_state_iter.keys()):
        src_dest_dir = f'cbir/{dataset_name}/{dataset_type}/pca{pca_argument}_l2_l3_l4_i{suffix[1:]}'
        if not os.path.exists(src_dest_dir):
            os.makedirs(src_dest_dir)
        random_state = random_state_iter[suffix]

        apply_pca_on_caltech256(X_path, y_path, src_dest_dir, minimal=minimal, pca_argument=pca_argument,
                                random_state=random_state, npy_type = npy_type)
        make_cbir_caltech256(src_dest_dir, dataset_info_path, n)
        save_AP_summary(src_dest_dir, dataset_info_path, n)
        
        