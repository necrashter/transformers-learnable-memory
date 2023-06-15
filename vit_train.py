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
from vit import MemoryCapableViT
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision.transforms.functional import to_pil_image, to_grayscale


# Define a transformation that converts images to RGB
def to_rgb(image):
    image = to_pil_image(image)
    return ToTensor()(image.convert('RGB'))


def create_dataset(dataset, datasets_dir, batch_size):
    if dataset == "CIFAR100":
        # Load and preprocess the dataset
        transform = transforms.Compose([
            transforms.RandomResizedCrop((224)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # CIFAR100
        train_dataset = datasets.CIFAR100(root=datasets_dir, train=True, transform=transform, download=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        validation_dataset = datasets.CIFAR100(root=datasets_dir, train=False, transform=transform, download=True)
        validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
        number_of_classes = 100

    elif dataset == "Places":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        try:
            train_dataset = datasets.Places365(root=datasets_dir,small=True, split="train-standard", transform=transform, download=True)

        except:
            train_dataset = datasets.Places365(root=datasets_dir,small=True, split="train-standard", transform=transform)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        try:
            validation_dataset = datasets.Places365(root=datasets_dir,small=True, split="val", transform=transform, download=True)

        except:
            validation_dataset = datasets.Places365(root=datasets_dir,small=True, split="val", transform=transform)

        validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

        number_of_classes = 365

    elif dataset == "INaturalist":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            Lambda(to_rgb)
        ])
        try:
            naturalist_dataset = datasets.INaturalist(root=datasets_dir, version="2017", transform=transform, download=True)
        except:
            naturalist_dataset = datasets.INaturalist(root=datasets_dir, version="2017", transform=transform)

        # Split the dataset into training and testing subsets
        train_size = int(0.8 * len(naturalist_dataset))
        test_size = len(naturalist_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(naturalist_dataset, [train_size, test_size])

        # Create data loaders for the training and test subsets
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

        number_of_classes = 5089

    else:
        raise ValueError(f"Unknown dataset name: {dataset}")

    return train_loader, validation_loader, number_of_classes


def warmup_linear(step):
    """
    Linear warmup.
    """
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return 1.0


def train(model, parameters,
          dataloader, valid_dataloader,
          output_head=None,
          total_steps = 20):
    model.train()

    # SGD with Momentum optimizer
    optimizer = optim.SGD(parameters, lr=0.1, momentum=0.9)
    # Cosine learning rate schedule
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    warmup_scheduler = LambdaLR(optimizer, warmup_linear)

    train_losses, valid_accuracy = [], []
    for step in tqdm(range(total_steps), leave=False):
        train_loss = 0.0
        for batch_idx, (data, targets) in enumerate(tqdm(dataloader, leave=False)):
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(data)
            if output_head is not None:
                outputs = outputs[output_head]
            loss = criterion(outputs.logits, targets)
            loss_val = loss.detach().cpu().item()
            train_loss += loss_val

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            clip_grad_norm_(parameters, max_norm=1.0)

            optimizer.step()

            # Update learning rate
            if step < warmup_steps:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()

        epoch_loss = train_loss/len(dataloader)
        print(f"step {step} loss is {epoch_loss:.4f}")
        train_losses.append(epoch_loss)

        valid_acc = validate(model, valid_dataloader, output_head)
        print(f"step {step} valid acc is {valid_acc:.2f}")
        valid_accuracy.append(valid_acc)

        #print(f"Epoch {epoch + 1}/{num_epochs} loss: {train_loss}")
    return train_losses, valid_accuracy


def validate(model, dataloader, output_head=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data, targets in tqdm(dataloader, leave=False):
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            if output_head is not None:
                outputs = outputs[output_head]
            _, predicted = torch.max(outputs.logits, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return correct / total


def pretrained_model(
        cache_dir,
        model_name = 'google/vit-base-patch32-224-in21k',
        num_classes = 100
        ):
    config = ViTConfig.from_pretrained(model_name, num_labels=num_classes, cache_dir=cache_dir)
    model = ViTForImageClassification.from_pretrained(model_name, config=config, cache_dir=cache_dir)
    model = model.to(device)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=["CIFAR100","INaturalist","Places"], help="Dataset for training", required=True)
    # Adding optional arguments
    parser.add_argument("-r", "--directory", help="Directory for home_dir", default = os.path.expanduser('~'))
    parser.add_argument("-e", "--epochs", type = int, help="Number of epochs", default = 20)
    parser.add_argument("-b", "--batch_size", type = int, help="Number of elements in a batch", default = 64)
    parser.add_argument("-m", "--number_of_memory_tokens", type = int, help="Number of memory tokens", default = 1)
    args = parser.parse_args()

    home_dir = str(args.directory)
    cache_dir = os.path.join(home_dir, "ceng502")
    datasets_dir = os.path.join(home_dir, "datasets")
    dataset = str(args.dataset)
    batch_size = int(args.batch_size)

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed is set as {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, validation_loader, number_of_classes = create_dataset(dataset, datasets_dir, batch_size)
    criterion = nn.CrossEntropyLoss()
    # Define number of steps and warmup steps
    total_steps = int(args.epochs)
    warmup_steps = 5

    model = pretrained_model(cache_dir)
    parameters = [model.vit.embeddings.cls_token] + list(model.classifier.parameters())
    base_model = model
    model = MemoryCapableViT(deepcopy(base_model))
    new_parameters = model.add_head(memory_tokens=int(args.number_of_memory_tokens), num_classes=number_of_classes)

    memory_train, memory_val = train(model, new_parameters, train_loader,validation_loader, output_head=1, total_steps = total_steps)

    print(f"Validation Accuracy: {validate(model, validation_loader, output_head=1)}")

    torch.save(model.state_dict(), f"models/{dataset}_model.pt")
