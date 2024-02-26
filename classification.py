import os
import time

import numpy as np
from sklearn import svm
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, recall_score, balanced_accuracy_score,
    precision_score, f1_score
    )


def classify_with_raport(X_train, y_train, X_test, y_test, features_dir, idx_to_class, t, save_pred=True):
    for kernel in ["linear", "rbf"]:
        start_time = time.time()
        clf = svm.SVC(kernel=kernel, max_iter=10_000_000, verbose=False)
        clf.fit(X_train, y_train.ravel())
        end_time = time.time()
        t["train"][f"svm_{kernel}_fit"] = end_time - start_time
        # with open(os.path.join(features_dir, f'svm_{kernel}_model.sav'), 'wb') as pickle_file:
        #     pickle.dump(clf, pickle_file)
        start_time = time.time()
        y_pred = clf.predict(X_test)
        end_time = time.time()
        t["test"][f"svm_{kernel}_predict"] = end_time - start_time
        if save_pred:
            with open(os.path.join(features_dir, 'y_pred.npy'), 'wb') as npy_file:
                np.save(npy_file, y_pred)

        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average=None, zero_division=np.nan)
        precision = precision_score(y_test, y_pred, average=None, zero_division=np.nan)
        f1 = f1_score(y_test, y_pred, average=None, zero_division=np.nan)
        micro_recall = recall_score(y_test, y_pred, average="micro", zero_division=np.nan)
        micro_precision = precision_score(y_test, y_pred, average="micro", zero_division=np.nan)
        micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=np.nan)
        macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=np.nan)
        macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=np.nan)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=np.nan)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, zero_division=np.nan)

        with open(os.path.join(features_dir, f"classification_report_svm_{kernel}.txt"), 'a') as fd:
            fd.write(f'Accuracy: {accuracy}\n\n')
            fd.write(f'Balanced_accuracy: {balanced_accuracy}\n\n')
            fd.write(f'F1: {f1}\n\n')
            fd.write(f'Precision: {precision}\n\n')
            fd.write(f'Recall: {recall}\n\n')
            fd.write(f'Micro F1: {micro_f1}\n\n')
            fd.write(f'Micro Precision: {micro_precision}\n\n')
            fd.write(f'Micro Recall: {micro_recall}\n\n')
            fd.write(f'Macro F1: {macro_f1}\n\n')
            fd.write(f'Macro Precision: {macro_precision}\n\n')
            fd.write(f'Macro Recall: {macro_recall}\n\n')
            fd.write(f'Features number: {X_train.shape[-1]}\n\n')
            fd.write(f'Class names: \n{idx_to_class}\n\n')
            fd.write(f'Classification report: \n{cr}\n')
            fd.write(f'Parameters: \n{clf.get_params()}\n\n')
            fd.write(f'Confusion matrix: \n{cm}\n\n\n')
    return t
