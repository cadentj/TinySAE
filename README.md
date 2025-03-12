# Intro

An absolutely minimal implementation of an SAE. Primarily intended for making hacky changes quickly.

Built off of https://github.com/EleutherAI/sparsify. Currently trains about 2 times slower than that implementation and achieves about the same FVU. I think this is primarily due to the fast triton kernels; I might add these at some point.

The following should install all necessary packages:
```
pip3 install torch wandb tqdm bitsandbytes datasets transformers dotenv accelerate einops
```