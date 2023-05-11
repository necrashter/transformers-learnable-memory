"""
Unit tests for memory capable ViT model.
Run with pytest.
"""
import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTModel, ViTConfig, ViTForImageClassification

from vit import MemoryCapableViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

home_dir = os.path.expanduser('~')
cache_dir = os.path.join(home_dir, "ceng502")
datasets_dir = os.path.join(home_dir, "datasets")


def test_memory_capable_vit():
    model_name = 'google/vit-base-patch32-224-in21k'
    config = ViTConfig.from_pretrained(model_name, num_labels=10, cache_dir=cache_dir)
    model = ViTForImageClassification.from_pretrained(model_name, config=config, cache_dir=cache_dir)
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.CIFAR10(root=datasets_dir, train=False, transform=transform, download=True)
    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=False)
    data, _ = next(iter(dataloader))
    data = data.to(device)

    original_output = model(data)

    # MemoryCapableViT must be equivalent to the default model if no new head/memory is added.
    model = MemoryCapableViT(model)
    new_output = model(data)[0]
    assert torch.allclose(original_output.logits, new_output.logits, atol=1e-5, rtol=1e-5)
