#!/bin/bash

echo Creating and entering the models directory...
mkdir models
cd models

wget https://huggingface.co/necrashter/transformers-learnable-memory/resolve/main/CIFAR100_model.pt -q --show-progress
wget https://huggingface.co/necrashter/transformers-learnable-memory/resolve/main/INaturalist_model.pt -q --show-progress
wget https://huggingface.co/necrashter/transformers-learnable-memory/resolve/main/Places_model.pt -q --show-progress
wget https://huggingface.co/necrashter/transformers-learnable-memory/resolve/main/Sun_model.pt -q --show-progress

cd ..
echo Done!
echo The models are available in ./models directory.
