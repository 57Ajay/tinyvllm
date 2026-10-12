from tinyvllm.attention.naive_flat import flat_sdpa_attention
from tinyvllm.attention.metadata import AttentionMetadata
import torch
from torch import nn
from dataclasses import dataclass
import torch.nn.functional as F


@dataclass(slots=True)
class GPT2Config:
    vocab_size: int = 50257
    max_position_embeddings: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(
        self,
        x: torch.Tensor,  # [N, n_embd]
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:  # [N, n_embd]
        N, _ = x.size()
        q, k, v = self.c_attn(x).split(self.config.n_embd, -1)
        head_dim = self.config.n_embd // self.config.n_head

        q = q.view(N, self.config.n_head, head_dim)
        k = k.view(N, self.config.n_head, head_dim)
        v = v.view(N, self.config.n_head, head_dim)

        out = flat_sdpa_attention(q, k, v, attn_metadata)
        out = out.reshape(N, self.config.n_embd)
        return self.c_proj(out)


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        out = self.c_proj(x)
        return out


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.attn = GPT2Attention(config)
        self.mlp = GPT2MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(
        self, x: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_metadata)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.n_embd)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "GPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,  # [N]
        positions: torch.Tensor,  # [N]
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:  # [num_requests, vocab_size]

        x = self.wte(input_ids) + self.wpe(positions)

        for block in self.h:
            x = block(x, attn_metadata)

        x = self.ln_f(x)
        if attn_metadata.logits_indices is not None:
            x = x[attn_metadata.logits_indices]

        return self.lm_head(x)
