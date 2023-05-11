"""
Unit tests for memory capable ViT model.
Run with pytest.
"""
import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTModel, ViTConfig, ViTForImageClassification
from copy import deepcopy

from vit import MemoryCapableViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

home_dir = os.path.expanduser('~')
cache_dir = os.path.join(home_dir, "ceng502")
datasets_dir = os.path.join(home_dir, "datasets")


def test_memory_capable_vit():
    model_name = 'google/vit-base-patch32-224-in21k'
    config = ViTConfig.from_pretrained(model_name, num_labels=10, cache_dir=cache_dir)
    base_model = ViTForImageClassification.from_pretrained(model_name, config=config, cache_dir=cache_dir)
    base_model = base_model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.CIFAR10(root=datasets_dir, train=False, transform=transform, download=True)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)
    data, _ = next(iter(dataloader))
    data = data.to(device)

    original_output = base_model(data).logits

    # MemoryCapableViT must be equivalent to the default model if no new head/memory is added.
    model = MemoryCapableViT(deepcopy(base_model))
    new_output = [i.logits for i in model(data)]
    assert torch.allclose(original_output, new_output[0], atol=1e-5, rtol=1e-5)

    old_parameters = list(model.parameters())
    # Try adding new memory
    parameters = model.add_head(4)
    # 1 class token, 12 memory parameters (1 for each self-attention layer),
    # weight and bias for the classifier head.
    assert len(parameters) == 15
    # There shouldn't be any old parameters in newly returned parameters
    for parameter in parameters:
        for old_parameter in old_parameters:
            assert not (parameter is old_parameter)
    # Newly returned parameters should be in all parameters
    all_parameters = list(model.parameters())
    for parameter in parameters:
        for model_parameter in all_parameters:
            if parameter is model_parameter:
                break
        else:  # Not found
            assert False

    new_output = [i.logits for i in model(data)]
    # Must return 2 outputs now
    assert len(new_output) == 2
    # First output should not change
    assert torch.allclose(original_output, new_output[0], atol=1e-5, rtol=1e-5)
    del new_output

    # Try once more
    old_parameters = all_parameters
    parameters = model.add_head(4)
    assert len(parameters) == 15
    for parameter in parameters:
        for old_parameter in old_parameters:
            assert not (parameter is old_parameter)
    # Newly returned parameters should be in all parameters
    all_parameters = list(model.parameters())
    for parameter in parameters:
        for model_parameter in all_parameters:
            if parameter is model_parameter:
                break
        else:  # Not found
            assert False

    model_output = [i.logits for i in model(data)]
    # Must return 3 outputs now
    assert len(model_output) == 3
    # First output should not change
    assert torch.allclose(original_output, model_output[0], atol=1e-5, rtol=1e-5)

    # Create another model
    model2 = MemoryCapableViT(base_model)
    # To avoid out of memory errors
    del base_model
    model2.add_head(2)
    model2.add_head(3)
    model2_output = [i.logits for i in model2(data)]
    assert len(model2_output) == 3
    assert torch.allclose(original_output, model2_output[0], atol=1e-5, rtol=1e-5)

    # Concatenate models
    model.concatenate(model2)
    del model2
    concat_output = [i.logits for i in model(data)]
    assert len(concat_output) == 5
    assert torch.allclose( original_output, concat_output[0], atol=1e-5, rtol=1e-5)
    assert torch.allclose( model_output[1], concat_output[1], atol=1e-5, rtol=1e-5)
    assert torch.allclose( model_output[2], concat_output[2], atol=1e-5, rtol=1e-5)
    assert torch.allclose(model2_output[1], concat_output[3], atol=1e-5, rtol=1e-5)
    assert torch.allclose(model2_output[2], concat_output[4], atol=1e-5, rtol=1e-5)
