import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tiny_sae import Sae, SaeConfig, train_sae, TrainConfig

def load_model_and_tokenizer(model_name, device):
    """Load the model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    return model, tokenizer

def prepare_dataset(dataset_name, tokenizer, context_len):
    """Load and prepare the dataset"""
    dataset = load_dataset(
        dataset_name,
        split="train",
        trust_remote_code=True,
    )
    dataset = dataset.shuffle(seed=42)

    def _tokenize_fn(x: dict[str, list]):
        output = tokenizer(
            x["text"], 
            max_length=context_len, 
            return_attention_mask=False, 
            truncation=True
        )
        return output

    # Tokenize the dataset
    data = dataset.map(
        _tokenize_fn, 
        batched=True, 
        batch_size=32, 
        num_proc=16, 
        load_from_cache_file=True
    )

    # Filter sequences to match context_len
    data = data.filter(
        lambda x: len(x["input_ids"]) == context_len,
        load_from_cache_file=True,
        num_proc=16,
    )

    return data

def main():
    # Configuration
    MODEL = "google/gemma-3-4b-pt"
    DATASET_NAME = "togethercomputer/RedPajama-Data-1T-Sample"
    CONTEXT_LEN = 1024
    DEVICE = "cuda:0"

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL, DEVICE)
    
    # Prepare dataset
    data = prepare_dataset(DATASET_NAME, tokenizer, CONTEXT_LEN)

    # Configure SAE
    sae_cfg = SaeConfig(
        d_in=2560,
        num_latents=2560*8,
        k=128,
        hookpoint="language_model.model.layers.27",
    )
    sae = Sae(sae_cfg, device=DEVICE)

    # Training configuration
    train_cfg = TrainConfig(
        wandb_project="tiny-sae",
        wandb_name="test",
        save_every_n_tokens=100_000_000,
        optimize_every_n_tokens=8192,
        model_batch_size=16,
        mask_first_n_tokens=1,
    )

    # Train SAE
    train_sae(
        sae=sae, 
        model=model, 
        token_iterator=data, 
        train_cfg=train_cfg, 
        use_wandb=True
    )

if __name__ == "__main__":
    main()
