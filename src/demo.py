import os
import torch
from torchvision import transforms
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report



from extract_resfeats import extract_features


# Dataset available at: https://www.kaggle.com/datasets/vencerlanz09/shells-or-pebbles-an-image-classification-dataset




def main():
    # Paths to data and results
    src_dir = os.path.join("data_demo", "Shells_and_Pebbles")
    resfeats_dir = os.path.join("resfeats", "Shells_and_Pebbles")
    pca_resfeats_dir = os.path.join("pca-resfeats", "Shells_and_Pebbles")
    
    if not os.path.exists(resfeats_dir):
        os.makedirs(resfeats_dir)
    if not os.path.exists(pca_resfeats_dir):
        os.makedirs(pca_resfeats_dir)

    
    # Features type: np.float16 or np.float32
    npy_type = np.float16
    
    # List of layers "l2", "l3", "l4", combination e.g. ["l2", "l4"]
    layers_combination = ["l2", "l3", "l4"]
    
    test_size = 0.2
    
    # Number of principal components in PCA
    pca_argument = 60
    
    # If with classification
    classify = True
    
    preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    device = torch.device('cpu')
    model_type = "imagenet"
    
    
    # Features extraction from ResNet-50
    extract_features(src_dir, resfeats_dir, preprocess, device, model_type)
  
    
    X = torch.hstack([torch.load(
        os.path.join(resfeats_dir, layer + ".pt")) for layer in layers_combination]).numpy().astype(npy_type)
    
    y = torch.load(os.path.join(resfeats_dir, "y.pt")).numpy().astype(np.int16)
    
    train_idx, test_idx, _, _ = train_test_split(list(range(len(X))), y, random_state=97, test_size=test_size, stratify=y)
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    
    pickle.dump(sc, open(os.path.join(pca_resfeats_dir, 'standard_scaler.sav'), 'wb'))
    
    # PCA
    if pca_argument is not None:
        if pca_argument < 1:
            pca = PCA(pca_argument)
        else:
            pca = PCA(min(pca_argument, X_train.shape[0], X_train.shape[1]))
    else:
        pca = PCA()
    X_train = pca.fit_transform(X_train)
    
    
    pickle.dump(pca, open(os.path.join(pca_resfeats_dir, 'pca_model.sav'), 'wb'))
    
    X_test = sc.transform(X_test)
    X_test = pca.transform(X_test)
    
    
    with open(os.path.join(pca_resfeats_dir, 'X_train_pca-resfeats.npy'),'wb') as npy_file:
        np.save(npy_file, X_train.astype(npy_type))
        
    with open(os.path.join(pca_resfeats_dir, 'X_test_pca-resfeats.npy'),'wb') as npy_file:
        np.save(npy_file, X_test.astype(npy_type))
    
    
    with open(os.path.join(pca_resfeats_dir, "pca_report.txt"),'a') as fd:
        fd.write(f'Explained variance: {sum(pca.explained_variance_ratio_)}\n')
        fd.write(f'Number of components: {pca.n_components_}\n')
        fd.write(f'Number of features: {pca.n_features_}\n')
        fd.write(f'Number of samples: {pca.n_samples_}\n\n\n')
        
    # Classification
    if classify:
        clf = svm.SVC(kernel="linear", verbose=False)
        clf.fit(X_train, y_train.ravel())
        
        with open(os.path.join(pca_resfeats_dir, 'svm_model.sav'), 'wb') as pickle_file:
            pickle.dump(clf, pickle_file)
        
        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        with open(os.path.join(pca_resfeats_dir, "classification_report.txt"),'a') as fd:
            fd.write(f'Accuracy: {accuracy}\n')
            fd.write(f'Classification report: \n{classification_report(y_test, y_pred)}\n')
            fd.write(f'Parameters: \n{clf.get_params()}\n\n\n')


if __name__ == "__main__":
    main()

    
    
        
    
    
