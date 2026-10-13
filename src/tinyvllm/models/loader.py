from huggingface_hub import hf_hub_download
import os
import torch
from transformers import GPT2LMHeadModel
from tinyvllm.models.gpt2 import GPT2Config, GPT2Model


def load_hf_gpt2(model_name: str = "gpt2") -> GPT2Model:
    print(f"Fetching '{model_name}' from HuggingFace...")
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    hf_dict = hf_model.state_dict()

    config = GPT2Config(
        vocab_size=50257,
        max_position_embeddings=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
    )
    model = GPT2Model(config)

    new_state_dict = {}
    for k, v in hf_dict.items():
        new_k = k
        if new_k.startswith("transformer."):
            new_k = k.removeprefix("transformer.")

        new_k = new_k.replace(".ln_1.", ".ln1.").replace(".ln_2.", ".ln2.")

        if any(
            proj in new_k for proj in ["c_attn.weight", "c_proj.weight", "c_fc.weight"]
        ):
            v = v.t()

        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict, strict=False)
    return model


def load_tinystories_checkpoint(
    repo_or_path: str = "~/projects/py/tinyvllm/weights/model.pt",
) -> GPT2Model:
    if os.path.exists(repo_or_path):
        ckpt_path = repo_or_path
    else:
        print(f"Downloading 'model.pt' from HuggingFace repo '{repo_or_path}'...")
        ckpt_path = hf_hub_download(repo_id=repo_or_path, filename="model.pt")

    print(f"Loading TinyStories-51M from '{ckpt_path}'...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = ckpt["config"]
    config = GPT2Config(
        vocab_size=cfg.get("vocab_size", 50304),
        max_position_embeddings=cfg.get("block_size", 512),
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
    )
    model = GPT2Model(config)

    raw_state_dict = ckpt["model"]
    cleaned_dict = {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in raw_state_dict.items()
    }

    model.load_state_dict(cleaned_dict, strict=False)
    return model
