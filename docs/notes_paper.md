# PagedAttention Paper Notes: Kwon et al., SOSP 2023

## 1. The Core Bottleneck
In autoregressive LLM decoding, memory bandwidth to load the Key-Value (KV) cache from High
Bandwidth Memory (HBM) to SRAM is the physical bottleneck. Generation throughput is constrained
by how many requests can fit in GPU memory simultaneously.

## 2. The Three Forms of KV Cache Memory Waste
Prior systems required KV caches to be stored in contiguous virtual memory:
1. **Reservation Fragmentation:** Pre-allocating contiguous buffers for the maximum sequence
length (`max_model_len`). Slots for future tokens sit idle for seconds/minutes.
2. **Internal Fragmentation:** Memory reserved for maximum length that is never utilized
because requests terminate early upon generating EOS.
3. **External Fragmentation:** Dynamic variable-sized contiguous allocations create scattered
memory holes that cause premature CUDA OOM errors.

**Empirical Reality:** In existing systems, only 20.4%–38.2% of allocated KV cache memory
held actual token representations. 60%–80% of VRAM was wasted.

## 3. The PagedAttention Mechanism
- Slices the KV cache into fixed-size physical blocks (default block size $B = 16$).
- Allocates blocks dynamically on-demand as tokens are generated.
- Maintains a per-request **Block Table** mapping logical token indices to physical block IDs.
- Eliminates external fragmentation entirely (all blocks are identical size).
- Bounds internal fragmentation to at most $B - 1$ tokens per sequence ($< 4\%$).

## 4. Key Architectural Features
- **Copy-on-Write (CoW):** Enables parallel sampling ($n > 1$) and beam search by sharing
prompt blocks across branches with reference counters, cloning physical blocks only when
branches diverge.
- **Prefix Caching:** Enables caching prompt prefixes across independent requests using
content hash chains.
- **Preemption Handling:** When VRAM is exhausted, requests are preempted via Recompute or
Swap-to-CPU.
