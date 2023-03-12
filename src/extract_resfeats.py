import os
import pickle
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from resnet_impl import resnet50
import torch.multiprocessing
from tqdm import tqdm


def check_file(path):
    return (os.path.splitext(path)[1] in [".png", ".jpg"])
    

def get_model(device, model_type):
    if model_type == "imagenet":
        model = resnet50(pretrained=True)
        return model.to(device)
    if model_type == "caltech256":
        model = resnet50(num_classes = 257)
        model.load_state_dict(torch.load("model_weights.pth", map_location=device))
        return model.to(device)
    

def extract_features(src_dir, dest_dir, preprocess, device, model_type = "imagenet"):
    torch.multiprocessing.set_sharing_strategy('file_system') 
    
    os.makedirs(dest_dir, exist_ok=True)
    
    dataset = ImageFolder(src_dir, preprocess, is_valid_file=check_file)
    
    with open(os.path.join(dest_dir, "dataset_info.pkl"), 'wb') as pickle_file:
        pickle.dump(dataset.__dict__, pickle_file)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    with torch.no_grad():
        model = get_model(device, model_type)
        model.eval()
        
        # a dict to store the activations
        activation = {}
        
        def getActivation(name):
          # the hook signature
          def hook(model, input, output):
            activation[name] = output.detach()
          return hook
        
        # register forward hooks on the layers of choice
        h2 = model.layer2.register_forward_hook(getActivation('l2'))
        h3 = model.layer3.register_forward_hook(getActivation('l3'))
        h4 = model.layer4.register_forward_hook(getActivation('l4'))
        ha = model.avgpool.register_forward_hook(getActivation('la'))
 
        l2=[]
        l3=[]
        l4=[]
        ly=[]
        la=[]
        
        for X, y in tqdm(dataloader, total=len(dataloader)):
            # forward pass -- getting the outputs
            out = model(X)
            # collect the activations in the correct list
            l2.append(torch.amax(activation['l2'], dim = (0,2,3)))
            l3.append(torch.amax(activation['l3'], dim = (0,2,3)))
            l4.append(torch.amax(activation['l4'], dim = (0,2,3)))
            ly.append(y)
            la.append(torch.squeeze(activation['la']))
       
       
        torch.save(torch.stack(l2), os.path.join(dest_dir, "l2.pt"))
        torch.save(torch.stack(l3), os.path.join(dest_dir, "l3.pt"))
        torch.save(torch.stack(l4), os.path.join(dest_dir, "l4.pt"))
        torch.save(torch.stack(ly), os.path.join(dest_dir, "y.pt"))
        torch.save(torch.stack(la), os.path.join(dest_dir, "out.pt"))
        
        del l2
        del l3
        del l4
        del ly
        del la
        
        
        h2.remove()
        h3.remove()
        h4.remove()
        ha.remove()
        


