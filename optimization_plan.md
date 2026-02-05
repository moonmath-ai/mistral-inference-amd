# Optimization plan: reduce host–device copies (AMD MI300X, 7B / 8×7B)

## Reference benchmarks (before optimization)

Baseline runs on single AMD MI300X, native (Mistral) path, 11 prompts per model. Use these numbers to compare after implementing the plan.

| Model | Command | TPS (tokens/s) | Notes |
|-------|---------|----------------|-------|
| Mistral-7B-Instruct-v0.3 | `python bench.py --model 7b_instruct_v.3` | **26.83 ± 5.33** | 11 prompts |
| Mixtral-8x7B-Instruct-v0.1 | `python bench.py --model 8x7b_instruct_v.1` | **14.16 ± 0.23** | 11 prompts |

Summary files: `output/benchmarks/<model_name_short>/native_summary_single_gpu.txt`. Re-run the same commands after changes and compare TPS to confirm improvement.

---

## Goal

Improve inference throughput for Mistral 7B and 8×7B on single AMD MI300X by keeping tensors on device and removing redundant host–device copies and CPU syncs. Profiling showed ~71% of GPU time in HtoD copies and related overhead; GEMMs are batch-1 (expected in decode) and matrix sizes are fine.

**Scope:** `mistral_inference` in this repo; single-GPU, smaller models first. Multi-GPU and 8×22B later.

---

## Root causes (from profiling)

1. **Cache metadata built on CPU then copied to device**  
   Every forward calls `cache.get_input_metadata(seqlens)`, which:
   - Syncs `kv_seqlens` to CPU (`.tolist()`).
   - Builds `to_cache_mask`, `cached_elements`, `positions`, `batch_idx` from Python lists via `torch.tensor(..., device=self.device)` (effectively HtoD).
   - Uses `.item()` and `.tolist()` for mask construction (GPU→CPU syncs).
   This runs once per forward and touches metadata for every layer.

2. **Positions and small tensors created on CPU**  
   `torch.arange(...)` with no device is CPU; then `.to(device)` causes HtoD. Same for `torch.tensor([...], device=...)` when the source is a Python list.

3. **Decode-loop syncs in `generate.py`**  
   - `is_finished = is_finished | (next_token == eos_id).cpu()`: GPU→CPU every token.
   - Multiple `.item()` calls per token for logprobs: each syncs a scalar.

4. **One-time or rare**  
   - `BufferCache.to(device, dtype)` and `Transformer.from_folder` / `freqs_cis.to(device)` are acceptable (startup or first use).
   - `cache.update_seqlens(seqlens)` uses `torch.tensor(seqlens, device=self.device)` every forward (small HtoD; can be improved once metadata is device-native).

---

## Proposed changes by file

### 1. `src/mistral_inference/cache.py`

**1.1 Avoid `kv_seqlens.tolist()` in `get_input_metadata`**

- Today: `seqpos = self.kv_seqlens.tolist()` then Python loop over `seqpos` and `seqlens`.
- Change: Keep `seqpos` on device. In `_get_input_metadata_layer`, compute positions and masks with tensor ops on device. Only call `.tolist()` or `.item()` where an external API (e.g. xformers mask) strictly requires a Python list.

**1.2 Build positions on device**

- Replace `torch.cat([torch.arange(pos, pos + seqlen) for pos, seqlen in zip(seqpos, seqlens)]).to(device=self.device, dtype=torch.long)` with device-side construction:
  - Option A: `torch.cat([torch.arange(pos, pos + seqlen, device=self.device, dtype=torch.long) for ...])` (no `.to()`).
  - Option B: Pre-allocate a positions buffer and fill with a single kernel or vectorized op to avoid many small `arange` calls (better if we have a fixed max length).

**1.3 Build `to_cache_mask` and `cached_elements` on device**

- Today: Python list comprehensions → `torch.tensor(..., device=self.device)`.
- Change: Compute on GPU from `seqlens` (as a tensor) and `seqpos` / `cache_size`:
  - e.g. cumulative offsets from `seqlens`, then `positions >= (seqlen - cache_size)` style logic in tensor form, so no Python list of bools and no HtoD for this tensor.

**1.4 Avoid `.item()` and `.tolist()` in mask construction**

- In `_get_input_metadata_layer`, lines ~246 and ~253 use `cached_s.clamp(max=cache_size).item()` and `(self.kv_seqlens + cached_elements).clamp(max=cache_size).tolist()` for xformers mask APIs.
- Change: Check whether xformers (e.g. `BlockDiagonalMask.from_seqlens`, `BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens`) can accept device tensors or a single batched descriptor. If not, keep a single minimal sync (e.g. one `.tolist()` per forward for the mask) instead of per-layer or per-element.

**1.5 `update_seqlens`**

- Today: `self.kv_seqlens += torch.tensor(seqlens, device=self.device, dtype=torch.long)`.
- Change: If `seqlens` is already a tensor (e.g. from the caller), add in place. If the API must stay list-based, this is a small tensor; still consider a small pre-allocated buffer and copy to avoid repeated list→tensor HtoD if we see it in profiles.

---

### 2. `src/mistral_inference/generate.py`

**2.1 Keep `is_finished` on device**

- Today: `is_finished = is_finished | (next_token == eos_id).cpu()`.
- Change: `is_finished = is_finished | (next_token == eos_id)` and keep `is_finished` on the same device as `next_token`. For the loop exit, use `if is_finished.all(): break` (no need to pull to CPU every token). Only sync when we must (e.g. for logging or early exit condition that cannot be expressed on device).

**2.2 Reduce logprob syncs**

- Today: Multiple `.item()` per token to append to Python list `logprobs[i]`.
- Options:
  - If logprobs are not needed for the API: add a flag to skip logprob collection in the hot path.
  - If needed: accumulate logprobs in a tensor on device and do a single `.tolist()` (or batch transfer) per step or per request, instead of one `.item()` per token per sequence.

**2.3 Input tensors**

- `torch.tensor(sum(prompt_chunks, []), device=model.device, dtype=torch.long)` and similar are already on device; keep this pattern. Ensure no unintended CPU materialization (e.g. via `.numpy()` or list indexing that forces sync).

---

### 3. `src/mistral_inference/transformer.py`

**3.1 `SimpleInputMetadata.from_seqlens`**

- Today: `torch.cat([torch.arange(0, seqlen) for seqlen in seqlens]).to(device=device, dtype=torch.long)`.
- Change: Use `torch.arange(..., device=device, dtype=torch.long)` inside the loop (or a single device-side construction with cumulative lengths) so the result is created on device and no `.to()` is needed.

**3.2 `freqs_cis`**

- Already moved to device only when needed; keep as is. Ensure no repeated .to() in the hot path (current check `if self._precomputed_freqs_cis.device != self.device` is fine).

---

### 4. `src/mistral_inference/transformer_layers.py`

**4.1 RMSNorm**

- `x.float().type_as(x)` and similar are device-resident; no change required unless profiling shows otherwise.

**4.2 Attention / FFN**

- No direct `.to()` or `.cpu()` in the hot path; xformers and linear layers use model device. If xformers internally does extra copies (e.g. for bias), that will show in profiler; we can revisit after cache and generate optimizations.

---

### 5. Other files

- **`vision_encoder.py`**: `freqs_cis.to(self.device)` and `positions.to(self.device)` — same idea: create on device where possible; otherwise one-time or rare.
- **`lora.py`**: `lora_state_dict` .to(device) at load time is fine.
- **`moe.py`**: `.to(inputs.dtype)` is in-place dtype cast; no device copy.
- **`main.py`**: Server path; not in the single-request decode hot path for this plan.

---

## Implementation phases

### Phase 1 – Quick wins (cache metadata + generate)

1. **cache.py**
   - Build `positions` on device (arange with `device=self.device`).
   - Avoid `kv_seqlens.tolist()` by passing `kv_seqlens` as a tensor into metadata construction and computing indices on device where possible.
   - Reduce or consolidate `.item()` / `.tolist()` in mask creation (at least one sync per forward instead of per layer if API allows).
2. **generate.py**
   - Keep `is_finished` on device; remove `.cpu()` in the decode loop.
   - Optionally add a “no logprobs” path that skips all `.item()` in the loop for benchmarking.

**Validation:** Re-run `bench.py --profile torch` (and Python profiler if desired); compare HtoD and total time. Expect a visible drop in `hipMemcpyWithStream` and `aten::_to_copy` / `aten::to` in the summary.

### Phase 2 – Cache metadata fully on device

3. **cache.py**
   - Build `to_cache_mask` and `cached_elements` with tensor ops on device (no Python list → `torch.tensor(..., device=...)`).
   - Ensure xformers mask APIs are called with minimal CPU data (tensors or one list per forward if necessary).

**Validation:** Same as Phase 1; aim for no remaining per-forward or per-layer HtoD in cache metadata.

### Phase 3 – Logprobs and polish

4. **generate.py**
   - If logprobs are required: batch device-side logprob collection and do one transfer per step or per request.
5. **transformer.py**
   - `SimpleInputMetadata.from_seqlens`: create positions on device.
6. Any remaining `torch.tensor(..., device=...)` from Python lists in the hot path: replace with device-side creation or a single batched transfer.

**Validation:** Full benchmark suite; compare TPS and latency with baseline before Phase 1.

---

## How to validate

- **Throughput:** `python bench.py --model 7b_instruct_v.3` (and 8×7B if available). Compare TPS before/after.
- **Profiler:** `python bench.py --model 7b_instruct_v.3 --profile torch` with 1–2 prompts. Check:
  - `profile_torch_*_summary.txt`: “By self GPU time” and “By input shape”.
  - Expect lower share for `Memcpy HtoD`, `aten::copy_`, `aten::_to_copy`, `aten::to`.
- **Correctness:** Existing tests (e.g. `tests/test_generate.py`) and spot-check decoded output and logprobs (if enabled) for a few prompts.

---

## Risks and notes

- **xformers API:** Mask APIs may require Python lists for sequence lengths. If so, we minimize to one `.tolist()` per forward and keep the rest on device.
- **ROCm/PyTorch:** Behavior is the same in principle; if any op (e.g. `nonzero`) triggers extra syncs on ROCm, we address those in a follow-up.
- **Multi-GPU:** Not in scope here; when we add multi-GPU, we will need to ensure device placement and copy patterns respect the new topology.

---

## Why HtoD dropped but total time only slightly

After Phase 1–3, profiling showed **Memcpy HtoD** share of GPU time dropping from ~71% to ~48%, but TPS improved only modestly (~2%). Reason:

1. **HtoD is a share of GPU time, not wall time**  
   Wall time = CPU work + sync points + GPU work. The profiler’s “%” is of *self GPU time*. Reducing GPU copy time doesn’t reduce wall time 1:1 if the **critical path** is elsewhere.

2. **Copies can overlap; syncs cannot**  
   Much HtoD is asynchronous (copy and compute overlap). So we removed copy *work*, but the limiting factor for latency is often **synchronization**: every `.tolist()` or `.item()` blocks the CPU until the GPU has finished up to that point. We still do **32 `.tolist()` per forward** (one per layer for xformers mask APIs), so the pipeline keeps stalling on those syncs.

3. **Critical path**  
   Total time is dominated by **kernel execution + sync points**, not by the HtoD we removed. To reduce total time further, we need to **reduce or batch sync points**, not only copies.

---

## Next phase: reduce sync points (lower total time)

Goal: fewer GPU→CPU syncs in the hot path so the critical path shortens and TPS can increase.

### 4.1 cache.py: minimize `.tolist()` for masks — **done**

- **Implemented:** In `get_input_metadata`, we now build a single 2D tensor `(n_layers, B)` for `kv_seqlen` (subsequent_prefill or decode), call **one** `.tolist()` per forward, and pass `kv_seqlen_lists[layer_idx]` into `_get_input_metadata_layer`. So there is **one GPU→CPU sync per forward** for masks instead of 32. xformers APIs still receive Python lists per layer; the sync is batched.

### 4.2 cache.py: avoid redundant `.item()` / syncs

- We still have `total_len = seqlens_t.sum().item()` once per forward (acceptable).
- `first_prefill = (kv_seqlens_tensor[0] == 0).item()` — one `.item()` per forward; keep unless we can express the branch without a sync.
- No other per-layer `.item()` in the hot path after Phase 1–3.

### 4.3 generate.py: logprobs — **done (no-logprobs path)**

- **Implemented:** `generate(..., return_logprobs=True)` is the default. When `return_logprobs=False`, we skip all logprob `.item()` calls in the prefill chunk pass and in the decode loop. `Chat.__call__(..., return_logprobs=True)` passes through to `generate`; the benchmark uses `return_logprobs=False` for maximum TPS.
- If logprobs are required in the future: **batch** — accumulate on device and do one transfer per step or per request instead of one `.item()` per token per sequence.

### 4.4 Profile sync cost

- In the PyTorch trace (Chrome/Perfetto), look for **gaps** between GPU work and CPU work: those are sync stalls. Correlate with `.tolist()` / `.item()` call sites.
- Optional: add small timing blocks in code around `get_input_metadata` and mask construction to see how much wall time is spent in the sync path.

### Validation

- Re-run `bench.py --model 7b_instruct_v.3` (and 8×7B); compare TPS to reference and to Phase 1–3 results.
- Re-run `--profile torch` and confirm Memcpy HtoD share stays low and that no new heavy syncs appear in the trace.

---

## Summary

| Area            | Change                                              | Phase |
|-----------------|-----------------------------------------------------|-------|
| cache.py        | positions on device; avoid kv_seqlens.tolist()       | 1     |
| cache.py        | fewer .item()/.tolist() in mask creation            | 1     |
| generate.py     | is_finished on device; optional no-logprobs path    | 1     |
| cache.py        | to_cache_mask, cached_elements built on device      | 2     |
| generate.py     | batched logprob transfer if needed                  | 3     |
| transformer.py | SimpleInputMetadata positions on device             | 3     |
| cache.py        | seqlens_t/total_len once per forward (no per-layer) | done  |
| cache.py        | one .tolist() per forward for mask kv_seqlen        | 4 done |
| generate.py     | optional no-logprobs path (bench uses it)           | 4 done |
| generate.py     | batch logprob transfer if needed later             | 4     |
| —               | profile sync cost in trace                          | 4     |

Goal: keep tensors on device for 7B/8×7B on MI300X, remove redundant copies and CPU syncs, then reduce sync points so the critical path shortens and total time (TPS) improves. Measure with `bench.py` and the PyTorch profiler.
