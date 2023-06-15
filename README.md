# [Fine-tuning Image Transformers using Learnable Memory](https://arxiv.org/abs/2203.15243)

This README file is an outcome of the [CENG502 (Spring 2023)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 20223) Project List](https://github.com/CENG502-Projects/CENG502-Spring2023) for a complete list of all paper reproduction projects.

# 1. Introduction

<!-- @TODO: Introduce the paper (inc. where it is published) and describe your goal (reproducibility). -->

[Fine-tuning Image Transformers using Learnable Memory](https://arxiv.org/abs/2203.15243) is a paper published in [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Sandler_Fine-Tuning_Image_Transformers_Using_Learnable_Memory_CVPR_2022_paper.html).
The proposed method introduces learnable memory tokens in each self-attention layer of Vision Transformer models, enabling non-destructive fine-tuning and preserving performance on previous tasks while adapting to new ones.

In this repository, we implement this paper in PyTorch and aim to reproduce the results with our limited computational resources.
- The main implementation in [`vit.py`](vit.py) is based on the ViT implementation in HuggingFace Transformers library.
  - Unit tests are available in [`test_vit.py`](test_vit.py). Run them using `pytest`.
- In [the mnist directory](./mnist), there's a minimal ViT implementation in PyTorch from scratch with support for learnable memory tokens.
  - This implementation is simpler and may be easier to understand.

## 1.1. Paper summary

<!-- @TODO: Summarize the paper, the method & its contributions in relation with the existing literature. -->

The main idea in the proposed method is to introduce learnable memory tokens in each self-attention layer.
These tokens don't attend to other tokens and they are discarded after the self-attention, but the other tokens attend to these tokens.
Furthermore, the performance of the model on the previous dataset is preserved thanks to the proposed attention masking strategy.
Thus, this method increases the capacity of a pre-trained model in a non-destructive manner while avoiding the catastrophic forgetting problem that plagues the fine-tuning approaches.
Finally, the attention masking allows us to concatenate separately fine-tuned models into a single model which enables the reuse of computation while running these models.

# 2. The method and our interpretation

## 2.1. The original method

@TODO: Explain the original method.

## 2.2. Our interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
