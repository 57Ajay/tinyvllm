import torch
from tinyvllm.attention.metadata import AttentionMetadata
from tinyvllm.models.gpt2 import GPT2Config, GPT2Model


def test_gpt2_flat_batch_forward():
    torch.manual_seed(42)
    config = GPT2Config(
        vocab_size=1000,
        max_position_embeddings=128,
        n_layer=2,
        n_head=4,
        n_embd=64,
    )
    model = GPT2Model(config)
    model.eval()

    # 3 requests with lengths 4, 2, and 3 tokens
    # Req 0: tokens at positions [0, 1, 2, 3]
    # Req 1: tokens at positions [10, 11] (e.g. continuing an ongoing request)
    # Req 2: tokens at positions [0, 1, 2]
    input_ids = torch.randint(0, config.vocab_size, (9,), dtype=torch.long)
    positions = torch.tensor([0, 1, 2, 3, 10, 11, 0, 1, 2], dtype=torch.long)

    query_start_loc = torch.tensor([0, 4, 6, 9], dtype=torch.int32)
    seq_lens = torch.tensor([4, 12, 3], dtype=torch.int32)
    # Last token of each request: index 3 for req 0, index 5 for req 1, index 8 for req 2
    logits_indices = torch.tensor([3, 5, 8], dtype=torch.long)

    metadata = AttentionMetadata(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        logits_indices=logits_indices,
    )

    with torch.no_grad():
        logits = model(input_ids, positions, metadata)

    # We have 3 requests, so we must get exactly 3 logits rows!
    assert logits.shape == (3, config.vocab_size), (
        f"Expected (3, {config.vocab_size}), got {logits.shape}"
    )
    assert not torch.isnan(logits).any(), "Logits contain NaNs!"
    print("GPT-2 flat-batch forward passed successfully!")
