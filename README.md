# Paper title [@TODO: Change]

This readme file is an outcome of the [CENG502 (Spring 2023)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 20223) Project List](https://github.com/CENG502-Projects/CENG502-Spring2023) for a complete list of all paper reproduction projects.

# 1. Introduction

@TODO: Introduce the paper (inc. where it is published) and describe your goal (reproducibility).

## 1.1. Paper summary

@TODO: Summarize the paper, the method & its contributions in relation with the existing literature.

# 2. The method and my interpretation

## 2.1. The original method

@TODO: Explain the original method.

## 2.2. Our interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

The original paper uses ViT-B/32 base model for finetuning with learnable memory. They have also examined ViT-B/16 and ViT-L/32 models for learnable memory token. The paper conducted their experiments on 4 different datasets which are CIFAR-100, i-Naturalist, Places-365, SUN-397. They have used accuracy value of the models on the validation sets as their performance metric. These 4 datasets can be easily found on the Internet. In addition to that, these datasets are also implemented in pytorch's dataset package. However, some implementation details about these datasets are not stated in the paper. For example, i-Naturalist and SUN-397 datasets do not have splits as training and validation sets, and the paper is not explained how they split the dataset into train and validation so we just randomly 0.8-0.2 splitted the datasets for these two dataset. Moreover, Places365 and i-Naturalist datasets have several different versions in the pytorch dataset package, and the versions are also not stated in the paper. We have used standart version for Places-365 dataset and 2017 version for i-Naturalist dataset. They have used SGD with Momentum with gradient clipping and 5-step linear
rate warmup for all their finetuning experiments, we also implemented these hyperparameters and have also used these settings in our experiments. In the paper, they have followed standard inception preprocessing for all datasets except CIFAR-100, where they used random clipping. We also followed the same preprocessing steps for all datasets. We also initialized the memory tokens from a distribution of N(0, 0.02) as in the paper. The main differences between the paper and our experimentation setups are the number of batch sizes and the number of finetuning steps. They have used 512 batch size and run for 20000 steps. However, due to limited resources in terms of memory and time constraints that we have, we were able to use 64 as our batch size and the number of steps are 20 steps for CIFAR-100, 10 steps i-Naturalist and SUN-397, 3 steps for Places365 as our finetuning steps. The numbers are much smaller than the paper's number of step, but the datasets are huge. For example, one epoch time duration is more than 13 hours for Places365 dataset. However, they have stated shorter runs generally reached slightly worse results but preserved the relative relationship, and we also found out our results are comparable with the paper's result. We only tested 1 cell memory token for our 4 dataset and compared with the result of the paper's 1 cell memory token. In the paper, they have also shared the full finetuning and head+class token only finetuning. In our implementation, we did not add them because these are not the main contributions of the paper. However, you can examine these type of finetuning options in our sample notebook.

## 3.2. Running the code

```
Directory structure:
	├── models *
	│   ├── CIFAR100_model.pt
	│   ├── Places_model.pt
	│   ├── INaturalist_model.pt
	│   └── Sun_model.pt
	├── images
	├── mnist
	│
	├── vit_train.py *
	├── vit_validation.py *
	├── vit.py *
	├── vit.ipynb ~
	├── large-vit.ipynb ~
	├── model_concatenate.ipynb ~
	└── requirements.txt *

Dataset Directory Structure:
	├── ceng502
	│   └── models--google--vit-base-patch32-224-in21k
	│
	└── datasets
```
- Folders and files with * on their right are the folders and files that should be downloaded.
- Dataset Directory is created on a given directory which should be declared as an argument while training or validation phase. We have created this directory at our HDD because the datasets are huge. The datasets will be downloaded if the code cannot find them at the given directory.
- models directory contains finetuned learnable memory models for each dataset, and you have to download them if you want to directly use the finetuned model without waiting for training.
- vit_train.py is the script for training and there are some arguments should be given.
- vit_validation.py is the script for validation and there are some arguments should be given.
- vit.py is the script for the memory token model which is essential part of our implementation.
- requirements.txt should be downloaded and the packages should be installed before running the scripts. You can install the required packages by the given command.

```bash
  pip install -r requirements.txt
```







## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
