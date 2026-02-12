# MoE Execution Deep Dive & All2All Hook Points

## 1. Call chain

```
Transformer.forward_partial(input_ids, seqlens, cache, images)
  → h = embed / recv  # (num_toks, dim)
  → for each layer: h = layer(h, freqs_cis, cache_view)
        TransformerBlock.forward(x, freqs_cis, cache)
          → r = attention(attention_norm(x), ...)
          → h = x + r
          → r = feed_forward(ffn_norm(h))   # ← MoE runs here
          → out = h + r
  → ...
```

- **MoE input**: `x` has shape `(num_toks, dim)`. Tokens are **packed**: all sequences in the batch concatenated along dimension 0. `seqlens` (e.g. `[5, 7, 2]`) gives per-sequence lengths; `num_toks = sum(seqlens)`.
- **Prefill**: `num_toks` can be large (e.g. hundreds or thousands).
- **Decode**: typically `num_toks = batch_size` (one token per sequence).

---

## 2. MoeLayer.forward step-by-step

**File**: `src/mistral_inference/moe.py`

### Inputs

- `inputs`: `(T, dim)` with `T = num_toks`, `dim = args.dim`.
- `MoeArgs`: `num_experts` (e.g. 8), `num_experts_per_tok` (e.g. 2).

### Step 1: Gate (routing)

```python
gate_logits = self.gate(inputs)   # (T, num_experts)
```

- `gate`: `nn.Linear(dim, num_experts, bias=False)`.
- One logit per expert per token. Routing is **local** (no cross-device yet).

### Step 2: Top‑k selection

```python
weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
# weights: (T, top_k),  selected_experts: (T, top_k),  values in [0, num_experts-1]
weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
```

- Each token gets exactly `num_experts_per_tok` experts and a weight vector (sum to 1 after softmax).

### Step 3: Expert loop (current: all on one device)

```python
results = torch.zeros_like(inputs)   # (T, dim)
for i, expert in enumerate(self.experts):
    batch_idx, nth_expert = torch.where(selected_experts == i)
    expert_out = expert(inputs[batch_idx])   # (len(batch_idx), dim)
    results[batch_idx] += weights[batch_idx, nth_expert, None] * expert_out
return results
```

- **Loop over experts** (not over tokens): for expert `i`, find all token indices `batch_idx` that selected it, and which of the top‑k slots `nth_expert` (0 or 1 for top‑2).
- Run `expert(inputs[batch_idx])` → shape `(len(batch_idx), dim)`.
- Scatter-add into `results[batch_idx]` with the corresponding weight. So each token accumulates `num_experts_per_tok` weighted contributions.

### Step 4: Output

- `results`: `(T, dim)`, same shape as `inputs`. Returned and then added as residual in `TransformerBlock`: `out = h + r`.

---

## 3. Expert (FeedForward) structure

Each expert is a **FeedForward** (SwiGLU):

- `w1`: `(dim → hidden_dim)`
- `w3`: `(dim → hidden_dim)`
- `w2`: `(hidden_dim → dim)`
- Forward: `w2(silu(w1(x)) * w3(x))`

So for expert parallelism you only need to move the **hidden states** (dim and hidden_dim); the gate stays on the “token” side (or replicated).

---

## 4. Where to hook an All2All

Expert parallelism usually splits **experts** across ranks; each rank owns a subset of experts. The all2all is used to:

1. **Dispatch**: Send each token’s hidden state to the ranks that own its selected experts (so each rank receives the tokens it must process).
2. **Expert compute**: Each rank runs only its local experts on the tokens it received.
3. **Combine**: Send expert outputs back so each rank has the contributions for “its” tokens, then reduce (sum) per token.

So the hook is **inside `MoeLayer.forward`**: replace the single-device expert loop with:

- (Optional) keep gate + topk on “current” rank or replicate gate everywhere.
- Build **dispatch**: for each token, list of `(expert_id, weight)`; optionally build per-rank send buffers (which token indices go to which rank).
- **All2all (dispatch)**: exchange so that rank `r` receives exactly the tokens that selected an expert on `r`, plus routing metadata (weight, which expert, original token index if needed).
- **Local expert compute**: on each rank, for each local expert `i`, run `expert(inputs_local)` on the tokens that selected it; scale by weight; prepare “partial results” keyed by original token index.
- **All2all (combine)**: send partial results back so that the rank that “owns” each token has all contributions for that token.
- **Reduce**: sum the weighted expert outputs per token; write into `results`.

The current code already gives you the exact interface:

- **Input**: `inputs (T, dim)`, and you need `selected_experts (T, top_k)`, `weights (T, top_k)`.
- **Output**: `results (T, dim)`.

So a clean hook is to **replace the body of `MoeLayer.forward`** (from after `weights = F.softmax(...)` to `return results`) with a backend that can be either:

- **Local backend**: current loop (for single rank / no expert parallelism).
- **All2all backend**: your implementation that does dispatch all2all → local experts → combine all2all → reduce.

---

## 5. Suggested hook interface

You can keep the gate and topk in `MoeLayer` and delegate “run experts and combine” to a backend:

```python
# Pseudocode for MoeLayer.forward with backend
gate_logits = self.gate(inputs)
weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)

# Hook point: backend does dispatch → expert compute → combine → reduce
results = self.moe_backend(
    inputs=inputs,
    weights=weights,
    selected_experts=selected_experts,
    experts=self.experts,          # or None on non-owner ranks if experts are sharded
    expert_to_rank=self.expert_to_rank,  # optional: expert_id -> rank
)
return results
```

- **Local backend**: implements the current for-loop over `self.experts` and returns `(T, dim)`.
- **All2all backend**: uses `inputs`, `weights`, `selected_experts`, and (if sharded) a subset of experts and `expert_to_rank`; performs two all2alls and returns `(T, dim)`.

You can also hook at a **lower level**: keep the loop but replace `expert(inputs[batch_idx])` with a call that either runs locally or sends to the expert rank and gets back the result; that usually implies custom send/recv or a single “all2all-style” layer that does both dispatch and combine in one go. The two-all2all pattern (dispatch then combine) is the standard and fits the “replace the loop with one backend call” hook above.

---

## 6. Summary

| Stage            | Current implementation                         | Tensor shapes / notes |
|------------------|------------------------------------------------|------------------------|
| Gate             | `gate(inputs)`                                 | (T, dim) → (T, num_experts) |
| Top‑k            | `topk(gate_logits, num_experts_per_tok)`       | (T, top_k) weights and expert ids |
| Expert loop      | For each expert, `where(selected_experts==i)`, `expert(inputs[batch_idx])`, scatter-add | (T, dim) → (T, dim) |
| Output           | `results`                                     | (T, dim) |

**Best hook for your all2all**: implement a **MoeBackend** (or equivalent) that takes `(inputs, weights, selected_experts)` and optional expert sharding info, and returns `(T, dim)`. Inside `MoeLayer.forward`, after computing `weights` and `selected_experts`, call this backend instead of the current for-loop. The backend can then use your all2all for dispatch and combine while keeping the rest of the transformer (and pipeline parallelism) unchanged.

---

## 7. Dispatch without a custom kernel (group tokens by expert)

There is **no single built-in PyTorch op** that does “dispatch” or “group tokens by expert” as a named API. You have two options.

### Option A: Pure PyTorch (no custom kernel)

You can build the **permutation** with existing ops and get one contiguous buffer whose rows are ordered by expert (all expert 0 assignments, then expert 1, …). No inter-GPU communication; all local.

- Flatten the routing: each of the `T * top_k` (token, expert) pairs has a token index and an expert id.
- **Argsort by expert**: sort so that all pairs for expert 0 come first, then expert 1, etc.
- Use the sorted order to form a **source index** (which token each position in the buffer comes from); then **index_select** to gather `inputs` into that order.

Result: one tensor of shape `(T * num_experts_per_tok, dim)` with rows grouped by expert. You can then compute per-expert sizes via a linear scan or `bincount` and run each expert on its slice.

```python
# inputs: (T, dim),  selected_experts: (T, top_k)
T, top_k = selected_experts.shape
num_experts = selected_experts.max().item() + 1

# Flatten: each of T*top_k positions has (token_idx, expert_id)
token_idx_flat = torch.arange(T, device=inputs.device).unsqueeze(1).expand(-1, top_k).flatten()  # (T*top_k,)
expert_flat = selected_experts.flatten()  # (T*top_k,)

# Sort by expert so that all expert-0 rows are first, then expert-1, etc.
order = expert_flat.argsort()  # (T*top_k,)
src_index = token_idx_flat[order]  # (T*top_k,) — which token to read for each position in the buffer

# One contiguous buffer: (T*top_k, dim), ordered by expert
dispatched = inputs[src_index]
```

To get **per-expert start/count** (for running each expert on its slice) you can use `torch.bincount(expert_flat, minlength=num_experts)` and then `cumsum` for offsets. So: **yes, you can do “sort tokens into sections by expert” in pure PyTorch** with `argsort` + `index_select` (+ optional bincount/cumsum). No custom kernel required.

Caveat: `argsort` is O(n log n). A single-pass custom kernel (e.g. histogram of counts per expert, then one pass writing `token_idx` into the right segment) can be faster and is a “trivial” kernel if you want to add it.

### Option B: Pre-existing fused MoE kernels (dispatch + expert + combine)

Libraries that implement **fused** MoE (dispatch + expert GEMM + combine) include:

- **Megablocks** (nomic-ai/megablocks): CUDA dispatch/combine/block-sparse kernels; there is a **ROCm/Megablocks** port for AMD (see AMD blog + ROCm compatibility docs). The dispatch step is inside the library, not exposed as a standalone “dispatch only” op.
- **PyTorch Triton MoE** (e.g. pytorch-labs/applied-ai col_major_moe_gemm, and PyTorch blog “Accelerating MoE’s with a Triton Persistent Cache-Aware Grouped GEMM Kernel”): fused or grouped-GEMM kernels that do the equivalent of dispatch + batched expert matmuls + combine. Again, not a separate “dispatch only” callable.

So: **there is no standalone “dispatch kernel” in stock PyTorch**, but (1) you can get the same layout with **argsort + index_select** in pure PyTorch, and (2) **fused MoE kernels** in Megablocks/Triton do include the dispatch step internally. If you only need the “sort into sections” part and want to avoid a custom kernel, use Option A; if you want maximum performance and are on AMD, consider Megablocks’ ROCm port or a small custom HIP kernel for the one-pass permutation.
