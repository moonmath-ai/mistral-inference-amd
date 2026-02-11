# Optimization Results Summary

## Initial vs Current Timing

| Metric | Initial (baseline) | Current | Delta |
|---|---:|---:|---:|
| Tokens Per Second | 11.41 | 18.19 | +6.78 (+59.4%) |
| Decode TPS | 11.55 | 14.09 | +2.54 (+22.0%) |
| Decode total (ms) | 23068.05 | 19021.65 | -4046.40 (-17.5%) |
| Prefill total (ms) | 106.16 | 111.24 | +5.08 (+4.8%) |
| TTFT (ms) | 106.20 | 111.27 | +5.07 (+4.8%) |
| Attention decode (ms) | 1059.14 | 1002.29 | -56.85 (-5.4%) |
| MoE dispatch+combine decode (ms) | 8411.73 | 6546.35 | -1865.38 (-22.2%) |
| MoE expert GEMM decode (ms) | 5157.18 | 3099.07 | -2058.11 (-39.9%) |

Notes:
- `Tokens Per Second` compares **non-instrumented** runs (without `--stage-timing`), i.e. production throughput.
- Decode-stage rows come from instrumented runs (`--stage-timing`) and are used for attribution, not absolute TPS.

## MoE Dispatch+Combine Work and Measurements

### What we targeted
- Decode bottleneck in Mixtral was MoE, especially dispatch+combine overhead.
- Goal: reduce routing/indexing/combining overhead while preserving expert correctness.

### Changes made
1. **First pass**
   - Flattened routing data and introduced `index_add_` for combine.
   - Added route sorting by expert.
2. **Second pass (current best)**
   - Removed global route sort (it added overhead in decode).
   - Kept flattened routing and `index_add_`.
   - Used per-expert route extraction from flattened assignments.
3. **Aggressive MI300X-focused pass (current best production TPS)**
   - Process only active experts per decode step (`torch.unique(flat_experts)`).
   - Cache flattened token index tensor to avoid rebuilding route index scaffolding each forward.
   - Keep ROCm-friendly `index_select` gather and in-place scale + `index_add_` combine.

### Timing progression (decode averages)
| Run | MoE dispatch+combine (ms) | MoE expert GEMM (ms) | Decode TPS | Notes |
|---|---:|---:|---:|---|
| Baseline | 8411.73 | 5157.18 | 11.55 | Starting point |
| After first pass | 9549.66 | 3192.83 | 12.06 | Dispatch worsened, expert compute improved |
| After second pass (current) | 6546.35 | 3099.07 | 14.09 | Dispatch improved strongly; best overall |

### Net outcome
- We achieved a significant decode speedup primarily by improving MoE path efficiency.
- Best measured production throughput (no stage instrumentation): **18.19 ± 0.32 TPS**.

## MoE Dispatch+Combine Speedup Analysis

### What the numbers say
- **Production throughput improved strongly**: 11.41 -> 18.19 TPS (**+59.4%**).
- Latest aggressive pass improved production TPS from 15.78 -> 18.19 (**+15.3%** incremental).
- On comparable coarse stage-timing runs, MoE dispatch+combine improved from:
  - **8411.73 ms -> 6546.35 ms** (**-22.2%**).
- This dispatch+combine reduction is the primary contributor to decode speedup.

### Important measurement caveat
- Fine-grained MoE sub-stage timing (`moe_gate`, `moe_topk`, `moe_gather`, etc.) introduces extra synchronization.
- Because of that instrumentation overhead, absolute dispatch+combine values in those runs are inflated and **not directly comparable** to coarse timing or production TPS.
- Use:
  - **no `--stage-timing`** for final TPS comparisons,
  - **`--stage-timing`** for bottleneck attribution only.
