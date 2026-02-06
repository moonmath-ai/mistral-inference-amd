"""
Minimal test: one attention forward with list-based decode mask vs tensor decode mask.
Compare output to see if the backend produces the same result.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from xformers.ops.fmha import memory_efficient_attention
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalWithOffsetPaddedKeysMask

from mistral_inference.cache import _decode_mask_tensor

def main():
    if not torch.cuda.is_available():
        print("No CUDA, skip")
        return
    dev = torch.device("cuda")
    dtype = torch.bfloat16
    B = 2
    cache_size = 32
    n_heads = 4
    head_dim = 8
    kv_seqlen_list = [5, 10]  # valid keys per batch

    # Same Q, K, V for both runs
    torch.manual_seed(42)
    q = torch.randn(1, B, n_heads, head_dim, device=dev, dtype=dtype)
    k = torch.randn(1, B * cache_size, n_heads, head_dim, device=dev, dtype=dtype)
    v = torch.randn(1, B * cache_size, n_heads, head_dim, device=dev, dtype=dtype)

    # 1) List-based mask (BlockDiagonalCausalWithOffsetPaddedKeysMask)
    mask_list = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
        q_seqlen=[1] * B,
        kv_padding=cache_size,
        kv_seqlen=kv_seqlen_list,
        device=dev,
    )
    out_list = memory_efficient_attention(q, k, v, mask_list)

    # 2) Tensor mask (our _decode_mask_tensor)
    kv_seqlen_tensor = torch.tensor(kv_seqlen_list, device=dev, dtype=torch.long)
    mask_tensor = _decode_mask_tensor(dev, B, cache_size, kv_seqlen_tensor, dtype)
    mask_tensor = mask_tensor.expand(1, n_heads, -1, -1).contiguous()
    out_tensor = memory_efficient_attention(q, k, v, mask_tensor)

    diff = (out_list - out_tensor).abs().max().item()
    print(f"Max diff list vs tensor mask: {diff}")
    if diff < 1e-2:
        print("OK: outputs match")
    else:
        print("MISMATCH: tensor mask path gives different attention output")
        print("out_list sample:", out_list[0, 0, 0, :4].tolist())
        print("out_tensor sample:", out_tensor[0, 0, 0, :4].tolist())

if __name__ == "__main__":
    main()
