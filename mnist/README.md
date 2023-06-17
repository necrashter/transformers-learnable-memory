# Minimal implementation from scratch

Before implementing the full model, we created a minimal ViT in PyTorch from scratch and added support for learnable memory tokens as a stepping stone.

- [`model.py`](./model.py) contains the full implementation as a library. It's not a standalone script.
  - This implementation is simpler compared to the main implementation that is based on the HuggingFace library, and hence it may be easier to understand.
  - Some functions are almost the same in both implementations, e.g., `build_attention_mask` method.
- [The notebook](./502 Project MNIST.ipynb) demonstrates this implementation on MNIST.
  - Since this is not the main implementation, the documentation is minimal. But we believe the code in the notebook is easy to understand due to its simplicity.

This implementation does not aim to reproduce the results given in the paper. It's designed to be basic and intended for educational purposes.
