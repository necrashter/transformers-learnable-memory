import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import transformers
from transformers import ViTModel, ViTConfig, ViTForImageClassification
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
from copy import deepcopy
import random
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from vit_train import validate, to_rgb, pretrained_model, create_dataset
from vit import MemoryCapableViT
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision.transforms.functional import to_pil_image, to_grayscale


device = "cuda:0"
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Directories for cache and datasets
home_dir = "/hdd/ege"
cache_dir = os.path.join(home_dir, "ceng502")
datasets_dir = os.path.join(home_dir, "datasets")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Adding optional argument
    
    parser.add_argument('-m', '--models-list', nargs='+', default=[], choices=["CIFAR100","INaturalist","Places", "Sun"], help = "List for models and datasets", required = True)
    parser.add_argument("-b", "--batch_size", type = int, help="Number of bathces", default = 64)
    parser.add_argument("-t", "--number_of_memory_tokens", type = int, help="Number of memory tokens", default = 1)

    args = parser.parse_args()
    model_list = list(args.models_list)
    batch_size = int(args.batch_size)
    memory_token = int(args.number_of_memory_tokens)

    valid_dataset_list = []
    print("Model for first dataset is loading...")
    base_model = pretrained_model(cache_dir = cache_dir)
    
    model = MemoryCapableViT(deepcopy(base_model))
    
    base_dataset = model_list[0]
    train_loader_base, valid_loader_base, number_of_class_base = create_dataset(base_dataset, datasets_dir, batch_size)
    
    valid_dataset_list.append(valid_loader_base)
    
    new_parameters = model.add_head(memory_tokens=memory_token, num_classes=number_of_class_base)

    model.load_state_dict(torch.load(f"models/{base_dataset}_model.pt"))
    
    print("Model for first dataset is loaded")
    
    for i in args.models_list[1:]:
        print("Next dataset is loading...")
        model_t = pretrained_model(cache_dir = cache_dir)
        model_t = MemoryCapableViT(deepcopy(model_t))
        train_loader_t, valid_loader_t, number_of_class_t = create_dataset(i, datasets_dir, batch_size)
        valid_dataset_list.append(valid_loader_t)
        model_t.add_head(memory_tokens=memory_token, num_classes=number_of_class_t)
        model_t.load_state_dict(torch.load(f"models/{i}_model.pt"))
        model.concatenate(model_t)
        
    print("All memory tokens and models have been loaded.")
    
    for index, i in enumerate(args.models_list):
        print(f"Validation for the Dataset {i}: {validate(model, valid_dataset_list[index], output_head = index + 1)}")
        
        
    
    

    

