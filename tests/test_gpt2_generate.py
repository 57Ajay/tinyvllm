import os
import pytest
import tiktoken
import torch
from tinyvllm.attention.metadata import AttentionMetadata
from tinyvllm.models.loader import load_hf_gpt2, load_tinystories_checkpoint


def test_hf_gpt2_generates_text():
    print("\nLoading HuggingFace GPT-2 weights...")
    model = load_hf_gpt2("gpt2")
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    prompt = "Alan Turing was a"
    prompt_tokens = enc.encode(prompt)

    # here i am generating 15 tokens autoregressively using flat-batch forward pass
    current_tokens = list(prompt_tokens)

    with torch.no_grad():
        for _step in range(15):
            N = len(current_tokens)
            input_ids = torch.tensor(current_tokens, dtype=torch.long)
            positions = torch.arange(0, N, dtype=torch.long)

            # 1 request in the batch:
            query_start_loc = torch.tensor([0, N], dtype=torch.int32)
            seq_lens = torch.tensor([N], dtype=torch.int32)
            # Only predict the last token:
            logits_indices = torch.tensor([N - 1], dtype=torch.long)

            metadata = AttentionMetadata(
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                logits_indices=logits_indices,
            )

            logits = model(input_ids, positions, metadata)  # [1, 50257]
            next_token = torch.argmax(logits[0], dim=-1).item()
            current_tokens.append(int(next_token))

    output_text = enc.decode(current_tokens)
    print(f"\nPrompt: '{prompt}'")
    print(f"Generated text: '{output_text}'")

    assert len(current_tokens) == len(prompt_tokens) + 15
    assert (
        "computer" in output_text.lower()
        or "mathematician" in output_text.lower()
        or "british" in output_text.lower()
    )
    print("Coherent generation test passed!")


def test_tinystories_51m_generates_text():
    weights_path = "weights/model.pt"
    if not os.path.exists(weights_path):
        pytest.skip(f"Checkpoint {weights_path} not found.")

    print(f"\nLoading TinyStories-51M weights from {weights_path}...")
    model = load_tinystories_checkpoint(weights_path)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    prompt = "Once upon a time, there was a little girl named Lily"
    prompt_tokens = [enc.eot_token] + enc.encode_ordinary(prompt)
    current_tokens = list(prompt_tokens)

    with torch.no_grad():
        for _step in range(25):
            N = len(current_tokens)
            input_ids = torch.tensor(current_tokens, dtype=torch.long)
            positions = torch.arange(0, N, dtype=torch.long)

            query_start_loc = torch.tensor([0, N], dtype=torch.int32)
            seq_lens = torch.tensor([N], dtype=torch.int32)
            logits_indices = torch.tensor([N - 1], dtype=torch.long)

            metadata = AttentionMetadata(
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                logits_indices=logits_indices,
            )

            logits = model(input_ids, positions, metadata)
            if logits.shape[-1] > 50257:
                logits[..., 50257:] = float("-inf")

            next_token = torch.argmax(logits[0], dim=-1).item()
            current_tokens.append(int(next_token))

    output_text = enc.decode(current_tokens[1:])
    print(f"\nPrompt: '{prompt}'")
    print(f"Generated Story:\n{output_text}\n")

    assert len(current_tokens) == len(prompt_tokens) + 25
    print("TinyStories-51M generation test passed!")
