import torch
import torch.nn.functional as F
from tinyvllm.attention.metadata import AttentionMetadata
from tinyvllm.attention.naive_flat import flat_sdpa_attention


def test_flat_sdpa_parity_with_separate_forwards():
    torch.manual_seed(42)
    num_heads = 4
    head_dim = 16

    # 3 requests with lengths 5, 3, and 7 tokens
    lens = [5, 3, 7]

    # Generate random Q, K, V for each request
    req_tensors = []
    for l in lens:
        q_req = torch.randn(l, num_heads, head_dim)
        k_req = torch.randn(l, num_heads, head_dim)
        v_req = torch.randn(l, num_heads, head_dim)
        req_tensors.append((q_req, k_req, v_req))

    # 1. Run each request individually with standard SDPA
    individual_outs = []
    for q_req, k_req, v_req in req_tensors:
        # standard SDPA expects [batch, heads, seq_len, head_dim]
        q_in = q_req.transpose(0, 1).unsqueeze(0)
        k_in = k_req.transpose(0, 1).unsqueeze(0)
        v_in = v_req.transpose(0, 1).unsqueeze(0)
        out_i = F.scaled_dot_product_attention(q_in, k_in, v_in, is_causal=True)
        # revert back to [seq_len, heads, head_dim]
        individual_outs.append(out_i.squeeze(0).transpose(0, 1))

    # 2. Concatenate into a single flat packed batch [N, num_heads, head_dim]
    q_flat = torch.cat([t[0] for t in req_tensors], dim=0)
    k_flat = torch.cat([t[1] for t in req_tensors], dim=0)
    v_flat = torch.cat([t[2] for t in req_tensors], dim=0)

    query_start_loc = torch.tensor([0, 5, 8, 15], dtype=torch.int32)
    seq_lens = torch.tensor([5, 3, 7], dtype=torch.int32)
    attn_metadata = AttentionMetadata(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
    )

    # Run packed forward
    flat_out = flat_sdpa_attention(q_flat, k_flat, v_flat, attn_metadata)

    # Verify bit-for-bit parity
    expected_out = torch.cat(individual_outs, dim=0)
    assert torch.allclose(flat_out, expected_out, atol=1e-6)
    print("Flat-batch SDPA parity verified successfully!")
