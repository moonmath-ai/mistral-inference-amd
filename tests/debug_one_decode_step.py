"""
Run one decode step with list mask vs tensor mask and compare logits.
Same prefill twice (same seed), then one decode each with list vs tensor mask.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from mistral_inference.cache import BufferCache, DecodeMetadataBuffers
from mistral_inference.transformer import Transformer

def main():
    if not torch.cuda.is_available():
        print("No CUDA, skip")
        return
    model_path = os.path.expanduser("~/models/7b_instruct_v.3")
    if not os.path.isdir(model_path):
        print("No model, skip")
        return

    model = Transformer.from_folder(model_path)
    model = model.eval()
    B = 1
    prompt = [1, 2, 3]
    device = model.device
    cache_window = max(prompt) + 32
    V = model.args.vocab_size

    def prefill_and_get_next(cache):
        cache.reset()
        # forward expects 1D input_ids (num_toks,) and seqlens
        input_ids = torch.tensor(prompt, device=device)
        with torch.inference_mode():
            logits = model.forward(input_ids, seqlens=[len(prompt)], cache=cache)
        next_token = logits[-1:].argmax(dim=-1)
        return next_token

    def one_decode_step(cache, use_tensor_mask, next_tok):
        decode_meta = cache.get_input_metadata_decode(B, use_tensor_mask=use_tensor_mask)
        decode_buffers = DecodeMetadataBuffers(model.n_local_layers, model.args.max_batch_size, device)
        decode_buffers.update_from_metadata(decode_meta, B)
        meta = decode_buffers.get_metadata_list(B, decode_meta, [1])
        with torch.inference_mode():
            logits = model.forward(next_tok, seqlens=[1], cache=cache, input_metadata=meta)
        return logits

    cache = BufferCache(
        model.n_local_layers, model.args.max_batch_size, cache_window,
        model.args.n_kv_heads, model.args.head_dim, model.args.sliding_window,
    )
    cache.to(device=device, dtype=model.dtype)

    torch.manual_seed(123)
    next_token = prefill_and_get_next(cache)
    logits_list = one_decode_step(cache, use_tensor_mask=False, next_tok=next_token)

    torch.manual_seed(123)
    next_token2 = prefill_and_get_next(cache)
    assert next_token.item() == next_token2.item(), "same seed should give same next token"
    logits_tensor = one_decode_step(cache, use_tensor_mask=True, next_tok=next_token2)

    diff = (logits_list - logits_tensor).abs().max().item()
    print(f"Logits max diff: {diff}")
    print("argmax list:  ", logits_list.argmax(dim=-1).item())
    print("argmax tensor:", logits_tensor.argmax(dim=-1).item())
    if diff < 0.1:
        print("OK")
    else:
        print("MISMATCH")

if __name__ == "__main__":
    main()
