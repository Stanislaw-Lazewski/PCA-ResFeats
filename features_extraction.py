import os
import pickle
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets_preparation import DatasetInfo, ImageDataset
from resnet_impl import resnet50, resnet101, resnet152
from utils import (
    DATASET_INFO_ROOT,
    MODELS_ROOT,
    PREPROCESS_DICT,
    RAW_FEATURES_ROOT,
)


def get_model_name(backbone_model, with_ext: bool = True) -> str:
    model_name = f"{backbone_model.model_arch}_{backbone_model.model_trained_on}"
    if backbone_model.suffix:
        model_name = model_name + f"_{backbone_model.suffix}"
    if with_ext:
        model_name = model_name + ".pth"
    return model_name


def get_model(device, backbone_model):
    if backbone_model.model_trained_on == "imagenet":
        model = eval(f'{backbone_model.model_arch}(pretrained=True)')
        return model.to(device)
    if backbone_model.model_trained_on == "caltech256":
        # TODO: CHECK NUM_CLASSES
        model = eval(f'{backbone_model.model_arch}(num_classes = 256)')
    elif backbone_model.model_trained_on == "kather":
        model = eval(f'{backbone_model.model_arch}(num_classes = 8)')
    else:
        msg = f"{backbone_model.model_arch} trained on {backbone_model.model_trained_on} not implemented yet!"
        raise NotImplementedError(msg)

    model_name = get_model_name(backbone_model)
    model_path = os.path.join(MODELS_ROOT, model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)


def extract_features(dataset_name, device, preprocess_id, backbone_model):
    torch.multiprocessing.set_sharing_strategy('file_system')

    dataset_info_dir = os.path.join(DATASET_INFO_ROOT, dataset_name)
    dataset_info = pickle.load(open(os.path.join(dataset_info_dir, "dataset_info.pkl"), 'rb'))

    dataset = ImageDataset(dataset_info, PREPROCESS_DICT[preprocess_id])

    model_name = get_model_name(backbone_model,
                                with_ext=False)
    raw_features_dir = os.path.join(RAW_FEATURES_ROOT, dataset_info.name, f"preprocess_{preprocess_id}", model_name)
    os.makedirs(raw_features_dir, exist_ok=False)

    with open(os.path.join(raw_features_dir, "dataset.pkl"), 'wb') as pickle_file:
        pickle.dump(dataset, pickle_file)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    with torch.no_grad():
        model = get_model(device, backbone_model)
        model.eval()

        # a dict to store the activations
        activation = {}

        def getActivation(name):
            # the hook signature
            def hook(model, _input, output):
                activation[name] = output.detach()
            return hook

        if backbone_model.model_arch.startswith("resnet"):
            # register forward hooks on the layers of choice
            h1 = model.layer1.register_forward_hook(getActivation('l1'))
            h2 = model.layer2.register_forward_hook(getActivation('l2'))
            h3 = model.layer3.register_forward_hook(getActivation('l3'))
            h4 = model.layer4.register_forward_hook(getActivation('l4'))
            ha = model.avgpool.register_forward_hook(getActivation('l4a'))

            for layer in ['l1', 'l2', 'l3', 'l4', 'l4a']:
                os.makedirs(os.path.join(raw_features_dir, layer), exist_ok=False)

            for i, (X, y, image_path) in enumerate(tqdm(dataloader, total=len(dataloader))):
                # forward pass -- getting the outputs
                X = X.to(device)
                start_time = time.time()
                out = model(X)
                end_time = time.time()
                with open(os.path.join(raw_features_dir, "time.csv"), 'a') as fd:
                    fd.write(f'{end_time-start_time}\n')

                # save the activations
                torch.save(activation['l1'], os.path.join(raw_features_dir, "l1", f"{i}.pt"))
                torch.save(activation['l2'], os.path.join(raw_features_dir, "l2", f"{i}.pt"))
                torch.save(activation['l3'], os.path.join(raw_features_dir, "l3", f"{i}.pt"))
                torch.save(activation['l4'], os.path.join(raw_features_dir, "l4", f"{i}.pt"))
                torch.save(activation['l4a'], os.path.join(raw_features_dir, "l4a", f"{i}.pt"))

            h1.remove()
            h2.remove()
            h3.remove()
            h4.remove()
            ha.remove()

        elif backbone_model.model_arch.startswith("densenet"):
            raise NotImplementedError()
        else:
            raise NotImplementedError()
