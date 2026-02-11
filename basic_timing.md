
## Baseline

Tokens Per Second: 11.39 ± 0.49

## Stage Timing Report (Mixtral-8x7B, `--stage-timing`, averages)








| Metric | Value |
|---|---|
| Tokenization | 0.43 ms |
| Prefill total | 106.16 ms |
| Prefill TPS | 224.98 tok/s |
| TTFT | 106.20 ms |
| Decode total | 23068.05 ms |
| Decode TPS | 11.55 tok/s |
| Attention (prefill) | 4.11 ms |
| Attention (decode) | 1059.14 ms |
| MoE dispatch+combine (decode) | 8411.73 ms |
| MoE expert GEMM (decode) | 5157.18 ms |

| Component | Share of decode total |
|---|---|
| MoE dispatch+combine | 36.5% |
| MoE expert GEMM | 22.4% |
| Attention | 4.6% |
| Other decode work | 36.6% |

Recommendations:
- Optimize MoE dispatch+combine first.
- Then optimize MoE expert GEMM batching/grouping.
- Attention is lower priority right now.

---

Basic stage timing changes
==========================

Goal
----
Add stage-level timing for native benchmark runs in a non-breaking way.

Enabled by
----------
Use:
  python bench.py --model 8x7b_instruct_v.1 --stage-timing

Default behavior is unchanged when `--stage-timing` is not set.

What is timed
-------------
Per prompt (native path):
- Tokenization
- Prefill total + prefill TPS
- TTFT
- Decode total + decode TPS
- Attention kernel time (prefill)
- Attention kernel time (decode)
- MoE dispatch+combine time (decode)
- MoE expert GEMM time (decode)

Implementation details
----------------------
1) New timing module:
   - src/mistral_inference/timing.py
   - Adds `StageTiming` dataclass.
   - Adds context helpers to tag current phase (`prefill` or `decode`).
   - Adds GPU-event timing helper (CUDA events) with CPU fallback.

2) Chat-level tokenization timing:
   - chat.py
   - `Chat.__call__` now supports `return_timing=False` (default).
   - If `return_timing=True`, it records tokenization time and passes timing
     collector into `generate(...)`.
   - Return type:
     - unchanged default: `(response, nof_tokens)`
     - timed mode: `(response, nof_tokens, timing)`

3) Prefill/decode/TTFT timing:
   - src/mistral_inference/generate.py
   - Adds optional `timing: StageTiming | None` parameter (default None).
   - Records:
     - prefill total and prefill token count
     - TTFT (from prefill start to first emitted token)
     - decode total and decode token count
   - Wraps model forward calls in `timing_scope(..., phase="prefill|decode")`.

4) Attention kernel timing:
   - src/mistral_inference/transformer_layers.py
   - Times `memory_efficient_attention(...)` using GPU events when timing is enabled.
   - Attributes elapsed time to prefill or decode using current phase.

5) MoE timing split (decode):
   - src/mistral_inference/moe.py
   - During decode phase:
     - measures total MoE layer time
     - measures expert compute time
     - computes dispatch+combine as:
         dispatch+combine = total_moe_time - expert_gemm_time
   - Accumulates decode-only MoE metrics.

6) Benchmark output:
   - bench.py
   - Adds CLI flag `--stage-timing`.
   - When enabled in native path:
     - prints per-prompt stage timing line
     - writes stage timing lines to summary output
     - adds average stage timings at end.

Files changed
-------------
- bench.py
- chat.py
- src/mistral_inference/generate.py
- src/mistral_inference/transformer_layers.py
- src/mistral_inference/moe.py
- src/mistral_inference/timing.py (new)
- basic_timing.md (this file)
