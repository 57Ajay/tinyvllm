from tinyvllm.attention.metadata import AttentionMetadata
import torch
import torch.nn.functional as F


def flat_sdpa_attention(
    q: torch.Tensor,  # [N, num_heads, head_dim]
    k: torch.Tensor,  # [N, num_kv_heads, head_dim]
    v: torch.Tensor,  # [N, num_kv_heads, head_dim]
    att_metadata: AttentionMetadata,
) -> torch.Tensor:  # [N, num_heads, head_dim]
    out = torch.empty_like(q)

    for i in range(len(att_metadata.seq_lens)):
        start = att_metadata.query_start_loc[i]
        end = att_metadata.query_start_loc[i + 1]
        q_i = q[start:end].transpose(0, 1).unsqueeze(0)
        k_i = k[start:end].transpose(0, 1).unsqueeze(0)
        v_i = v[start:end].transpose(0, 1).unsqueeze(0)

        out_i = F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=True)
        out[start:end] = out_i.squeeze(0).transpose(0, 1)

    return out
