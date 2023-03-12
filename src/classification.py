import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pickle
from random import sample
from pyzfp import compress, decompress


def apply_pca_on_X(X_train, X_test, pca_dir):
    X_type = X_train.dtype.name
    with open(os.path.join(pca_dir, 'standard_scaler.sav'), 'rb') as pickle_file:
        sc = pickle.load(pickle_file)
        
    X_train = sc.transform(X_train).astype(X_type)
    X_test = sc.transform(X_test).astype(X_type)
        
    with open(os.path.join(pca_dir, 'pca_model.sav'), 'rb') as pickle_file:
        pca = pickle.load(pickle_file)
    
    X_train = pca.transform(X_train).astype(X_type)
    X_test = pca.transform(X_test).astype(X_type)
    
    return X_train, X_test



def apply_features_on_X(X_train, X_test):
    return X_train, X_test


def apply_subsampled_on_X(X_train, X_test, n):
    return X_train[:, ::n], X_test[:, ::n]


def apply_random_subsampled_on_X(X_train, X_test, n, classification_dir):
    to_sample = range(0, X_train.shape[1])
    sampled = sample(to_sample, n)
    with open(os.path.join(classification_dir, "subsampled_coord_list.pkl"), 'wb') as pickle_file:
        pickle.dump(sampled, pickle_file)
    return X_train[:, sampled], X_test[:, sampled]


def linear_svm_classification_datasets(apply_transformations_on_X, transformation_args, full_dir, minimal_dir, classification_dir, save_model=False):
    with open(os.path.join(full_dir, "X.npy"), 'rb') as f:
        X = np.load(f)
      
    with open(os.path.join(full_dir, "y.npy"), 'rb') as f:
        y = np.load(f)

    with open(os.path.join(minimal_dir, "test_idx_list.pkl"), 'rb') as pickle_file:
        test_idx = pickle.load(pickle_file)
        
    with open(os.path.join(minimal_dir, "train_idx_list.pkl"), 'rb') as pickle_file:
        train_idx = pickle.load(pickle_file)
    
    X_train = X[(train_idx)]
    X_test = X[(test_idx)]
    
    y_train = y[(train_idx)]
    y_test = y[(test_idx)]
    
    save_path = os.path.join(classification_dir, "classification_report.txt")
    if not os.path.exists(classification_dir):
        os.makedirs(classification_dir)
    
    X_train, X_test = apply_transformations_on_X(X_train, X_test, *transformation_args)
    
    clf = svm.SVC(kernel="linear", verbose=False)
    clf.fit(X_train, y_train.ravel())
    
    if save_model:
        with open(os.path.join(classification_dir, 'svm_model.sav'), 'wb') as pickle_file:
            pickle.dump(clf, pickle_file)
    
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    with open(save_path,'a') as fd:
        fd.write(f'Accuracy: {accuracy}\n')
        fd.write(f'Classification report: \n{classification_report(y_test, y_pred)}\n')
        fd.write(f'Parameters: \n{clf.get_params()}\n\n\n')
        
    return accuracy


def linear_svm_classification_zfp_datasets(apply_transformations_on_X, transformation_args, full_dir, minimal_dir, tolerance_list, parallel=True):
    with open(os.path.join(full_dir, "X.npy"), 'rb') as f:
        X = np.load(f)
      
    with open(os.path.join(full_dir, "y.npy"), 'rb') as f:
        y = np.load(f)

    with open(os.path.join(minimal_dir, "test_idx_list.pkl"), 'rb') as pickle_file:
        test_idx = pickle.load(pickle_file)
        
    with open(os.path.join(minimal_dir, "train_idx_list.pkl"), 'rb') as pickle_file:
        train_idx = pickle.load(pickle_file)
    
    X_train = X[(train_idx)]
    X_test = X[(test_idx)]
    
    y_train = y[(train_idx)]
    y_test = y[(test_idx)]
    
    X_train, X_test = apply_transformations_on_X(X_train, X_test, *transformation_args)
    
    original_size = len(X_train.tobytes())
    results_list = []
    
    clf = svm.SVC(kernel="linear", verbose=False)
    clf.fit(X_train, y_train.ravel()) 
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # Results without zfp compression
    results_list.append([X_train.shape[1], accuracy, 0, 1.0, original_size, original_size, 0.0, 0.0, 0.0])
    
    for tolerance in tolerance_list:
        compressed = compress(X_train, parallel=parallel, tolerance=tolerance)
        recovered = decompress(compressed, X_train.shape, X_train.dtype, tolerance=tolerance)
        
        original_size = len(X_train.tobytes())
        compressed_size = len(compressed)
        compression_ratio = original_size / compressed_size
        # space_saving = (1 - compressed_size / original_size)
        
        # clf = svm.SVC(kernel="linear", max_iter=10000000, verbose=False)
        clf = svm.SVC(kernel="linear", verbose=False)
        clf.fit(recovered, y_train.ravel())
        
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results_list.append([recovered.shape[1], accuracy, tolerance, compression_ratio, original_size, compressed_size, np.linalg.norm(recovered - X_train), np.absolute(recovered - X_train).mean(), np.absolute(recovered - X_train).max()])
            
    return results_list

    