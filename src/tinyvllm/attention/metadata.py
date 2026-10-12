import torch
from dataclasses import dataclass


@dataclass(slots=True)
class AttentionMetadata:
    query_start_loc: torch.Tensor
    """
    shape: [number_of_requests + 1]

    what it represents: How many tokens of
    this request are we computing in this exact
    forward pass?
    """

    seq_lens: torch.Tensor
    """
    shape: [number_of_requests] 

    what it represents: How many total
    tokens does this request have in its
    entire conversation history up to this moment?
    """

    logits_indices: torch.Tensor | None = None
    """
    shape: [number_of_requests]

    what it represents: the vector of last token
    indice which we need to generate next token 
    of each request
    """
