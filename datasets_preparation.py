import os
import pickle
import warnings
from typing import Tuple, List

import torch
import torch.multiprocessing
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

from utils import (
    BREAKHIS_CLASS_SUBDIRS,
    BREAKHIS_MAGS,
    DATASET_INFO_ROOT,
    DATASETS_ROOT,
    N_SPLITS,
    RANDOM_STATE,
    SPLIT_STRATEGY,
)


class DatasetInfo:
    def __init__(self, name, dataset_dir, paths, classes, all_classes, skipped_classes=[], labels=None, encoded_labels=None):
        if labels is None and encoded_labels is None:
            msg = f"Missing labels or encoded_labels for {name} dataset"
            raise ValueError(msg)
        self.name = name
        self.dataset_dir = dataset_dir
        self.paths = np.array(paths)
        self.classes = classes
        self.all_classes = all_classes
        self.skipped_classes = skipped_classes
        encoded_classes = list(range(len(self.classes)))
        self.class_to_idx = dict(zip(self.classes, encoded_classes))
        self.idx_to_class = dict(zip(encoded_classes, self.classes))
        if encoded_labels is None:
            self.encoded_labels = np.array(list(map(lambda x: self.class_to_idx[x], labels)))
        else:
            assert sorted(range(len(classes))) == sorted(set(encoded_labels))
            self.encoded_labels = encoded_labels

        assert len(self.paths) == len(self.encoded_labels), "Image labels are incompatible with image paths."


class ImageDataset(Dataset):
    def __init__(self, data: DatasetInfo, transform=transforms.ToTensor()) -> None:
        self.name = data.name
        self.dataset_dir = data.dataset_dir
        self.paths = data.paths
        self.encoded_labels = data.encoded_labels
        self.classes = data.classes
        self.all_classes = data.all_classes
        self.skipped_classes = data.skipped_classes
        self.class_to_idx = data.class_to_idx
        self.idx_to_class = data.idx_to_class

        self.transform = transform

        self.image_extensions = set(map(lambda x: os.path.splitext(x)[1], self.paths))
        if len(self.image_extensions) > 1:
            msg = f"Multiple file formats {self.image_extensions} in {self.name} dataset."
            warnings.warn(msg)
        if sorted(list(map(lambda x: self.class_to_idx[x], self.classes))) != sorted(set(self.encoded_labels)):
            msg = f"Classes are not correct: try {sorted(set(map(lambda x: self.idx_to_class[x], self.encoded_labels)))} instead of {sorted(self.classes)}"
            raise ValueError(msg)

        assert len(self.paths) == len(self.encoded_labels), "Image labels are incompatible with image paths."

        # If error with jpg or tif format, reinstall pillow
        _exts = Image.registered_extensions()
        pil_supported_ext = {ex for ex, f in _exts.items() if f in Image.OPEN}
        self.is_pil_supported_ext = (self.image_extensions <= pil_supported_ext)
        self.special_ext = {".npy"}
        if not self.is_pil_supported_ext and not (self.image_extensions <= self.special_ext):
            msg = f"File format not supported: {self.image_extensions}. PIL supported extensions: {pil_supported_ext}"
            raise NotImplementedError(msg)

    def load_image(self, index: int):
        "Opens an image via a path and returns it."
        image_path = self.paths[index]

        if self.is_pil_supported_ext:
            img = Image.open(image_path)
        elif self.image_extensions == {".npy"}:
            img_array = np.load(image_path)
            img = Image.fromarray(img_array)
        else:
            msg = f"File format not supported: {self.image_extensions}."
            raise NotImplementedError(msg)
        return img.convert('RGB'), image_path

    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        "Returns one sample of data, data, label and image_path (X, y, image_path)."
        img, image_path = self.load_image(index)
        class_idx = self.encoded_labels[index]

        if self.transform:
            return self.transform(img), class_idx, image_path  # return data, label, image_path (X, y, image_path)
        else:
            return img, class_idx, image_path  # return data, label, image_path (X, y, image_path)


# TODO: IMPLEMENTATION MISSING
def has_image_folder_structure(dataset_dir):
    return True


def prepare_data(dataset_name, dataset_dir, n_classes=None, skipped_classes=[]) -> List[DatasetInfo]:
    dataset_info_list = []
    if dataset_name == "BreaKHis":
        if n_classes == 2:
            class_coord = 0
        elif n_classes == 8:
            class_coord = 2
        else:
            msg = f"{n_classes} not equal 2 or 8 for {dataset_name} dataset!"
            raise ValueError(msg)
        all_classes = list(set(map(lambda x: x[class_coord], BREAKHIS_CLASS_SUBDIRS)))

        # Dataset with all magnifications
        name = f"{dataset_name}_C{n_classes}"
        paths = []
        labels = []
        classes = set()
        for class_subdir in BREAKHIS_CLASS_SUBDIRS:
            if class_subdir[class_coord] in skipped_classes:
                continue
            classes.add(class_subdir[class_coord])
            class_dir = os.path.join(dataset_dir, *class_subdir)
            for sub in os.listdir(class_dir):
                for mag in BREAKHIS_MAGS:
                    img_dir = os.path.join(class_dir, sub, mag)
                    for file in os.listdir(img_dir):
                        paths.append(os.path.join(img_dir, file))
                        labels.append(class_subdir[class_coord])
        skipped_classes = sorted(set(all_classes).difference(classes))
        dataset_info_list.append(DatasetInfo(name=name,
                                             dataset_dir=dataset_dir,
                                             paths=paths,
                                             labels=labels,
                                             classes=sorted(classes),
                                             all_classes=all_classes,
                                             skipped_classes=skipped_classes))
        if len(classes) != len(all_classes):
            msg = f"""{len(classes)} classes out of {len(all_classes)} available were used.
            {skipped_classes} was skipped."""
            warnings.warn(msg)

        # Split by magnifications
        for mag in BREAKHIS_MAGS:
            name = f"{dataset_name}_{mag}_C{n_classes}"
            paths = []
            labels = []
            classes = set()
            for class_subdir in BREAKHIS_CLASS_SUBDIRS:
                if class_subdir[class_coord] in skipped_classes:
                    continue
                classes.add(class_subdir[class_coord])
                class_dir = os.path.join(dataset_dir, *class_subdir)
                for sub in os.listdir(class_dir):
                    img_dir = os.path.join(class_dir, sub, mag)
                    for file in os.listdir(img_dir):
                        paths.append(os.path.join(img_dir, file))
                        labels.append(class_subdir[class_coord])
            skipped_classes = sorted(set(all_classes).difference(classes))
            dataset_info_list.append(DatasetInfo(name=name,
                                                 dataset_dir=dataset_dir,
                                                 paths=paths,
                                                 labels=labels,
                                                 classes=sorted(classes),
                                                 all_classes=all_classes,
                                                 skipped_classes=skipped_classes))
            if len(classes) != len(all_classes):
                msg = f"""{len(classes)} classes out of {len(all_classes)} available were used.
                {skipped_classes} was skipped."""
                warnings.warn(msg)

    elif has_image_folder_structure(dataset_dir):
        if n_classes is not None:
            msg = f"Use 'skipped_classes' parameter instead n_classes={n_classes} or implement preparing your own dataset"
            raise ValueError(msg)
        name = dataset_name
        paths = []
        labels = []
        classes = set()
        all_classes = sorted(os.listdir(dataset_dir))
        for class_name in all_classes:
            if class_name in skipped_classes:
                continue
            classes.add(class_name)
            class_dir = os.path.join(dataset_dir, class_name)
            for file in os.listdir(class_dir):
                if os.path.splitext(file)[1] == "":
                    continue
                paths.append(os.path.join(class_dir, file))
                labels.append(class_name)
        skipped_classes = sorted(set(all_classes).difference(classes))
        dataset_info_list.append(DatasetInfo(name=name,
                                             dataset_dir=dataset_dir,
                                             paths=paths,
                                             labels=labels,
                                             classes=sorted(classes),
                                             all_classes=all_classes,
                                             skipped_classes=skipped_classes))
        if len(classes) != len(all_classes):
            msg = f"""{len(classes)} classes out of {len(all_classes)} available were used.
            {skipped_classes} was skipped."""
            warnings.warn(msg)

    else:
        msg = f"{dataset_name} not implemented yet!"
        raise NotImplementedError(msg)

    return dataset_info_list


def split_dataset(dataset_info: DatasetInfo, strategy: str, random_state=97, n_splits=5):
    dataset_info_dir = os.path.join(DATASET_INFO_ROOT, f"{dataset_info.name}")
    if not os.path.exists(dataset_info_dir):
        msg = f"{dataset_info_dir} directory for {dataset_info.name} dataset does not exist."
        raise FileNotFoundError(msg)
    if os.path.exists(os.path.join(dataset_info_dir, strategy)):
        msg = f"Strategy {strategy} for {dataset_info.name} dataset already exists."
        raise FileExistsError(msg)

    if strategy == "cross-validation":
        X = list(range(len(dataset_info.encoded_labels)))
        y = dataset_info.encoded_labels
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for i, (train_idx, test_idx) in tqdm(enumerate(skf.split(X, y)), total=skf.get_n_splits()):
            fold = i + 1
            fold_dir = os.path.join(dataset_info_dir, strategy, f"fold_{fold}_{n_splits}")
            os.makedirs(fold_dir, exist_ok=False)

            train_idx = shuffle(train_idx, random_state=random_state)
            test_idx = shuffle(test_idx, random_state=random_state)

            with open(os.path.join(fold_dir, 'test_idx.npy'), 'wb') as npy_file:
                np.save(npy_file, np.array(test_idx, dtype=np.int32))

            with open(os.path.join(fold_dir, 'train_idx.npy'), 'wb') as npy_file:
                np.save(npy_file, np.array(train_idx, dtype=np.int32))

            train_dataset_info = DatasetInfo(
                name=dataset_info.name,
                dataset_dir=dataset_info.dataset_dir,
                paths=dataset_info.paths[train_idx],
                encoded_labels=dataset_info.encoded_labels[train_idx],
                classes=dataset_info.classes,
                all_classes=dataset_info.all_classes,
                skipped_classes=dataset_info.skipped_classes
                )
            test_dataset_info = DatasetInfo(
                name=dataset_info.name,
                dataset_dir=dataset_info.dataset_dir,
                paths=dataset_info.paths[test_idx],
                encoded_labels=dataset_info.encoded_labels[test_idx],
                classes=dataset_info.classes,
                all_classes=dataset_info.all_classes,
                skipped_classes=dataset_info.skipped_classes
                )
            with open(os.path.join(fold_dir, "train_dataset_info.pkl"), 'wb') as pickle_file:
                pickle.dump(train_dataset_info, pickle_file)
            with open(os.path.join(fold_dir, "test_dataset_info.pkl"), 'wb') as pickle_file:
                pickle.dump(test_dataset_info, pickle_file)

    elif strategy == "predetermined_split":
        if "BreaKHis" in dataset_info.name:
            for fold in tqdm(range(1, 6)):
                fold_filename = os.path.join(DATASETS_ROOT, "BreaKHis", f"dsfold{fold}.txt")
                f = open(fold_filename, "r")
                fold_dir = os.path.join(dataset_info_dir, strategy, f"fold_{fold}_5")
                os.makedirs(fold_dir, exist_ok=False)
                filename_list = []
                train_test_list = []
                train_idx = []
                test_idx = []
                for row in f.readlines():
                    columns = row.split('|')
                    filename_list.append(columns[0])
                    train_test_list.append(columns[3].strip())
                assert len(set(filename_list)) == len(filename_list)
                assert len(train_test_list) == len(filename_list)
                for i, path in enumerate(dataset_info.paths):
                    filename = os.path.basename(path)
                    if filename in filename_list:
                        train_test_str = train_test_list[filename_list.index(filename)]
                        if train_test_str == "train":
                            train_idx.append(i)
                        elif train_test_str == "test":
                            test_idx.append(i)
                        else:
                            msg = f"Check file {fold_filename}. Unexpected {train_test_str} value."
                            raise ValueError(msg)

                with open(os.path.join(fold_dir, 'test_idx.npy'), 'wb') as npy_file:
                    np.save(npy_file, np.array(test_idx, dtype=np.int32))

                with open(os.path.join(fold_dir, 'train_idx.npy'), 'wb') as npy_file:
                    np.save(npy_file, np.array(train_idx, dtype=np.int32))

                train_dataset_info = DatasetInfo(
                    name=dataset_info.name,
                    dataset_dir=dataset_info.dataset_dir,
                    paths=dataset_info.paths[train_idx],
                    encoded_labels=dataset_info.encoded_labels[train_idx],
                    classes=dataset_info.classes,
                    all_classes=dataset_info.all_classes,
                    skipped_classes=dataset_info.skipped_classes
                    )
                test_dataset_info = DatasetInfo(
                    name=dataset_info.name,
                    dataset_dir=dataset_info.dataset_dir,
                    paths=dataset_info.paths[test_idx],
                    encoded_labels=dataset_info.encoded_labels[test_idx],
                    classes=dataset_info.classes,
                    all_classes=dataset_info.all_classes,
                    skipped_classes=dataset_info.skipped_classes
                    )
                with open(os.path.join(fold_dir, "train_dataset_info.pkl"), 'wb') as pickle_file:
                    pickle.dump(train_dataset_info, pickle_file)
                with open(os.path.join(fold_dir, "test_dataset_info.pkl"), 'wb') as pickle_file:
                    pickle.dump(test_dataset_info, pickle_file)


def get_strategy(dataset_name):
    if "BreaKHis" in dataset_name:
        return SPLIT_STRATEGY["BreaKHis"]
    else:
        return SPLIT_STRATEGY[dataset_name]


def prepare_dataset(dataset_name, dataset_dir, n_classes=None, skipped_classes=[]):
    dataset_info_list = prepare_data(dataset_name=dataset_name,
                                     dataset_dir=dataset_dir,
                                     n_classes=n_classes,
                                     skipped_classes=skipped_classes)
    for dataset_info in dataset_info_list:
        dataset_info_dir = os.path.join(DATASET_INFO_ROOT, f"{dataset_info.name}")
        os.makedirs(dataset_info_dir, exist_ok=False)

        with open(os.path.join(dataset_info_dir, "dataset_info.pkl"), 'wb') as pickle_file:
            pickle.dump(dataset_info, pickle_file)

        split_dataset(dataset_info=dataset_info,
                      strategy=get_strategy(dataset_info.name),
                      random_state=RANDOM_STATE,
                      n_splits=N_SPLITS)
