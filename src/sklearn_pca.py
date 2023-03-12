import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

def make_pca(full_dir, minimal_dir, dest_dir, pca_argument=0.99):
    with open(os.path.join(full_dir, "X.npy"), 'rb') as f:
        X = np.load(f)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
           
    with open(os.path.join(minimal_dir, "train_idx_list.pkl"), 'rb') as pickle_file:
        train_idx = pickle.load(pickle_file)
    
    X_train = X[(train_idx)]

    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
 
    with open(os.path.join(dest_dir, 'standard_scaler.sav'), 'wb') as pickle_file:
        pickle.dump(sc, pickle_file)
    
    if pca_argument is not None:
        if pca_argument < 1:
            pca = PCA(pca_argument)
        else:
            pca = PCA(min(pca_argument, X.shape[0], X.shape[1]))
    else:
        pca = PCA()
    pca.fit(X_train)

    with open(os.path.join(dest_dir, 'pca_model.sav'), 'wb') as pickle_file:
        pickle.dump(pca, pickle_file)

    save_path = os.path.join(dest_dir, "pca_report.txt")
    with open(save_path,'a') as fd:
        fd.write(f'Explained variance: {sum(pca.explained_variance_ratio_)}\n')
        fd.write(f'Number of components: {pca.n_components_}\n')
        fd.write(f'Number of features: {pca.n_features_}\n')
        fd.write(f'Number of samples: {pca.n_samples_}\n\n\n')
