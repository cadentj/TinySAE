# %%

try:
    from IPython import get_ipython  # type: ignore

    ipython = get_ipython()
    assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

    is_notebook = True
except:
    is_notebook = False

import dotenv

dotenv.load_dotenv()

# %%

import os

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tiny_sae import Sae, SaeConfig, train_sae, TrainConfig


MODEL = "openai-community/gpt2"
dataset = load_dataset(
    "togethercomputer/RedPajama-Data-1T-Sample",
    split="train",
    trust_remote_code=True,
)
dataset = dataset.shuffle(seed=43)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

context_len = 1024


class DatasetIterator:
    def __init__(self):
        self.length = len(dataset)

    def __iter__(self):
        for i in range(len(dataset)):
            tokens = tokenizer(dataset[i]["text"], return_tensors="pt")["input_ids"][0]
            if len(tokens) > context_len:
                tokens = tokens[:context_len]
            if len(tokens) < context_len:
                continue
            yield tokens

    def __len__(self):
        return self.length


def dataset_iterator():
    return DatasetIterator()


device = "cuda:0"
gpt = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
)
# %%

sae_cfg = SaeConfig(
    d_in=768,
    num_latents=2**14,
    k=64,
    hookpoint="transformer.h.8",
)

sae = Sae(sae_cfg, device=device)

# %%
cfg = TrainConfig(
    wandb_project="tiny-sae",
    wandb_name="test",
    save_every_n_tokens=10_000_000,
    optimize_every_n_tokens=8192,
    model_batch_size=16,
)
train_sae(sae, gpt, dataset_iterator(), train_cfg=cfg)
# %%
