from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Iterable

import einops
import torch as t
from safetensors.torch import load_model, save_model
from tqdm import tqdm
import wandb
from transformers import (
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)


class EarlyExit(Exception):
    pass


@dataclass
class SaeConfig:
    d_in: int
    num_latents: int
    hookpoint: str
    k: int


class Sae(t.nn.Module):
    def __init__(
        self,
        cfg: SaeConfig,
        device: str | t.device | None = None,
        dtype: t.dtype | None = None,
    ):
        super().__init__()

        self.cfg = cfg

        self.encoder = t.nn.Linear(
            self.cfg.d_in, self.cfg.num_latents, device=device, dtype=dtype
        )
        self.encoder.bias.data.zero_()

        self.W_dec = t.nn.Parameter(self.encoder.weight.data.clone())
        self.set_decoder_norm_to_unit_norm()

        self.b_dec = t.nn.Parameter(
            t.zeros(self.cfg.d_in, dtype=dtype, device=device)
        )

    @staticmethod
    def load_from_disk(
        path: Path | str, device: str | t.device = "cpu"
    ) -> "Sae":
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            cfg = SaeConfig(**cfg_dict)

        sae = Sae(cfg, device=device)
        load_model(
            model=sae,
            filename=str(path / "sae.safetensors"),
            device=str(device),
        )
        return sae

    def save_to_disk(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_model(self, str(path / "sae.safetensors"))
        with open(path / "cfg.json", "w") as f:
            json.dump(asdict(self.cfg), f)

    @property
    def device(self):
        return self.encoder.weight.device

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    def encode(self, x: t.Tensor) -> t.Tensor:
        forward = self.encoder(x - self.b_dec)
        top_acts, top_indices = forward.topk(self.cfg.k, dim=-1, sorted=False)
        return top_acts, top_indices

    def decode(
        self,
        top_acts: t.Tensor,
        top_indices: t.Tensor,
    ) -> t.Tensor:
        batch_size = top_indices.shape[0]
        top_acts = top_acts.flatten(end_dim=1)
        top_indices = top_indices.flatten(end_dim=1)
        res = t.nn.functional.embedding_bag(
            top_indices, self.W_dec, per_sample_weights=top_acts, mode="sum"
        )
        res = einops.rearrange(res, "(b n) d -> b n d", b=batch_size)
        return res + self.b_dec, top_indices

    def forward(self, x: t.Tensor, return_indices: bool = False) -> t.Tensor:
        x_hat, top_indices = self.decode(*self.encode(x))
        if return_indices:
            return x_hat, top_indices
        return x_hat

    @t.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        norm = t.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + 1e-5


@dataclass
class TrainConfig:
    wandb_project: str
    wandb_name: str
    mask_first_n_tokens: int
    model_batch_size: int = 8
    save_every_n_tokens: int = 10_000_000
    optimize_every_n_tokens: int = 8192


def train_sae(
    sae: Sae,
    model: PreTrainedModel,
    token_iterator: Iterable[t.Tensor],
    train_cfg: TrainConfig,
    use_wandb: bool = True,
):
    if use_wandb:
        wandb.init(
            name=train_cfg.wandb_name,
            project=train_cfg.wandb_project,
            config={
                "sae_config": asdict(sae.cfg),
                "train_config": asdict(train_cfg),
            },
            save_code=True,
        )

    # Auto-select LR using 1 / sqrt(d) scaling law from Fig 3 of the paper
    lr = 2e-4 / (sae.cfg.num_latents / (2**14)) ** 0.5
    optimizer = t.optim.Adam(sae.parameters(), lr=lr)

    num_batches = len(token_iterator) // train_cfg.model_batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer, 1000, num_batches)

    x = None

    def hook(module: t.nn.Module, inputs, outputs):
        nonlocal x
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        x = outputs

        raise EarlyExit("Stop here")

    hookpoint = model.get_submodule(sae.cfg.hookpoint)
    handle = hookpoint.register_forward_hook(hook)

    global_step = 0
    num_tokens_since_fired = t.zeros(sae.cfg.num_latents, device="cuda")

    def _update_tokens_since_fired(top_indices: t.Tensor, num_tokens: int):
        nonlocal num_tokens_since_fired
        top_indices = top_indices.flatten(end_dim=1).unique()

        num_tokens_since_fired.add_(num_tokens)
        num_tokens_since_fired.index_fill_(0, top_indices, 0)

        dead_count = (num_tokens_since_fired > 5_000_000).sum()
        wandb.log({"dead_count": dead_count / sae.cfg.num_latents}, step=global_step)

    try:
        tokens_seen_since_last_step = 0
        tokens_seen_since_last_save = 0
        pbar = tqdm(token_iterator)
        batch = []
        for tokens in pbar:
            if len(batch) < train_cfg.model_batch_size:
                batch.append(t.tensor(tokens["input_ids"]))
                continue

            batch = t.stack(batch).to(model.device)
            n_tokens = batch.numel()
            tokens_seen_since_last_step += n_tokens
            tokens_seen_since_last_save += n_tokens

            with t.no_grad():
                try:
                    model(batch)
                except EarlyExit:
                    pass

            x = x.to(sae.dtype).to(sae.device)
            x = x[:, train_cfg.mask_first_n_tokens :]

            x_hat, top_indices = sae(x, return_indices=True)
            _update_tokens_since_fired(top_indices, n_tokens)
            error = x_hat - x
            loss = error.pow(2).sum()
            loss = loss / (x_hat - x_hat.mean(1, keepdim=True)).pow(2).sum()
            loss.backward()

            if tokens_seen_since_last_step >= train_cfg.optimize_every_n_tokens:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                sae.set_decoder_norm_to_unit_norm()
                tokens_seen_since_last_step = 0
                if use_wandb:
                    wandb.log({"fvu": loss.item()}, step=global_step)
                global_step += 1

            if tokens_seen_since_last_save >= train_cfg.save_every_n_tokens:
                sae.save_to_disk(f"sae-ckpts/{train_cfg.wandb_name}")
                tokens_seen_since_last_save = 0

            pbar.set_postfix(loss=loss.item())
            batch = []
    finally:
        handle.remove()
        if use_wandb:
            wandb.finish()
