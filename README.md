# Intro

An absolutely minimal implementation of an SAE. Primarily intended for making hacky changes quickly.

Based on the code from https://github.com/EleutherAI/sparsify. Currently trains about as fast as that implementation and achieves about the same FVU.

The following should install all necessary packages:
```
pip3 install torch wandb tqdm bitsandbytes datasets transformers dotenv accelerate einops
```