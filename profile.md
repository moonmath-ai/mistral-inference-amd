# Profiling plan for bench.py

This document describes how to add profiling support to `bench.py` so we can measure where time is spent: in Python, in PyTorch ops, and on the GPU (ROCm/HIP). All profiling is triggered from the same entry point and applies only to the **native** (Mistral) benchmark path; the vLLM path is unchanged. All profiler outputs (including rocprof) are written under **`output/benchmarks/<model_name_short>/`**.

---

## Entry point and scope

- **Script:** `bench.py` (single entry point for benchmarking and profiling).
- **Profiled path:** Native run only (`run_mistral_benchmark`). When `--vllm` is set, profiling flags are ignored.
- **Output directory:** All profiler outputs go under `output/benchmarks/<model_name_short>/` (same as existing benchmark outputs), with consistent naming so we can implement modes one by one.

---

## CLI design

Add to `bench.py`:

- **`--profile`** (or **`--profiler`**): choose profiling mode.
  - Values: `none` | `python` | `torch` | `rocprof`
  - Default: `none` (current behaviour, no profiler).

Only one of `python`, `torch`, or `rocprof` is active per run. No combination of multiple in-process profilers in a single run (to keep overhead and interaction simple). For `rocprof`, bench.py re-executes itself in a subprocess under rocprof and then moves rocprof’s output into the same output tree (see Phase 3).

---

## Implementation order

We implement in three steps:

1. **Phase 1: Python profiler** — cProfile (built-in), optional pyinstrument later.
2. **Phase 2: PyTorch profiler** — `torch.profiler` with CPU + GPU activities and timeline trace.
3. **Phase 3: ROCm profiler** — when `--profile rocprof`, run bench.py under rocprof in a subprocess and move rocprof output to `output/benchmarks/<model_name_short>/`.

---

## Phase 1: Python profiler

### Goal

Identify where **CPU (Python)** time is spent: tokenizer, data handling, `generate`, etc.

### Mechanism

- Use **cProfile** (stdlib): no extra dependencies.
- When `--profile python`:
  - Run `run_mistral_benchmark(args.model, prompts)` under `cProfile.Profile()`.
  - No change to warmup or prompt list; same workload as a normal benchmark run.

### Outputs (under `output/benchmarks/<model_name_short>/`)

- `profile_python_<suffix>.prof` — binary profile for `pstats`.
- `profile_python_<suffix>.txt` — human-readable summary (e.g. `pstats.Stats(...).strip_dirs().sort_stats(...).print_stats()`).

Suffix can include `single_gpu` / `multigpu` to match existing summary naming.

### Usage

```bash
python bench.py --model 7b_instruct_v.3 --profile python
# Inspect:
python -m pstats output/benchmarks/Mistral-7B-Instruct-v0.3/profile_python_single_gpu.prof
```

### Optional later

- Add **pyinstrument** as an alternative when `--profile python` and pyinstrument is installed: write an HTML report to `profile_python_<suffix>.html` for easier browsing.

---

## Phase 2: PyTorch profiler

### Goal

See **Python + GPU** in one place: which ops/kernels run, how long they take, and optional memory usage. Works with PyTorch built for ROCm.

### Mechanism

- Use **`torch.profiler.profile`** (and related APIs).
- When `--profile torch`:
  - Enable both CPU and GPU activities (use the correct activity enum for ROCm if different from CUDA).
  - Wrap only the part that runs the model (e.g. the inner loop in `run_mistral_benchmark` that calls `chat(prompt)`, or the whole `run_mistral_benchmark` if preferred).
  - Use a simple schedule (e.g. one-shot) so we don’t need warmup steps inside the profiler; we already have a warmup before.
  - Export the timeline trace (Trace Event Format) and, optionally, a text summary table of top ops/kernels.

### Outputs (under `output/benchmarks/<model_name_short>/`)

- `profile_torch_<suffix>.json` — Trace Event Format (open in Chrome `chrome://tracing` or Perfetto UI https://ui.perfetto.dev).
- `profile_torch_<suffix>_summary.txt` — optional table of top ops/kernels.

### Implementation notes

- Check PyTorch ROCm docs for the exact activity type for GPU (e.g. `ProfilerActivity.CUDA` vs ROCm-specific).
- If GPU activity is not available on ROCm in the current PyTorch version, document that and still export CPU trace and summary.

### Usage

```bash
python bench.py --model 7b_instruct_v.3 --profile torch
# Open the exported .json in either:
#   - Chrome: chrome://tracing -> Load profile_torch_*.json
#   - Perfetto: https://ui.perfetto.dev -> Open trace file (same format)
```

---

## Phase 3: ROCm profiler (rocprof)

### Goal

Get **low-level GPU view**: HIP kernel names, runtimes, and memory copies. No Python context, but authoritative GPU-side data. Output is written to the **same directory** as other profilers: `output/benchmarks/<model_name_short>/`.

### Mechanism

- **rocprof** is an external process wrapper. When `--profile rocprof`:
  - bench.py re-executes itself in a subprocess under rocprof:
    - Build argv without `--profile` (and without `--profile rocprof`) so the child runs a normal benchmark.
    - Run: `rocprof [options] python bench.py --model ...` (same model, same prompts).
    - Run the subprocess with **cwd** set to `output/benchmarks/<model_name_short>/` so that rocprof writes its output there directly; or run from repo root and after the subprocess exits, **move** rocprof’s output (e.g. `results.stats.csv`, `results.hip_stats.csv`) from the current directory into `output/benchmarks/<model_name_short>/` with names like `profile_rocprof_<suffix>.csv` (or keep original names and document them).
  - Prefer running the subprocess with `cwd=output/benchmarks/<model_name_short>/` if rocprof always writes to cwd, so rocprof writes where other profiles write without a second move step.

### rocprof options to use

- Basic stats: `rocprof --stats ...` (kernel stats).
- Optional: `-t HIP_API,HIP_RUNTIME,OPS` for more detail (document in profile.md and README).

### Outputs (under `output/benchmarks/<model_name_short>/`)

- rocprof result files (e.g. `results.stats.csv`, `results.hip_stats.csv`, or after move: `profile_rocprof_<suffix>.csv`) in the **same directory** as `profile_python_*` and `profile_torch_*`.

### Usage

```bash
python bench.py --model 7b_instruct_v.3 --profile rocprof
# Results in output/benchmarks/<model_name_short>/ (same as other profilers)
```

---

## Summary table

| Phase | Profiler   | Flag               | What it measures        | Main outputs                                      |
|-------|------------|--------------------|--------------------------|---------------------------------------------------|
| 1     | Python     | `--profile python` | CPU (Python) time        | `.prof`, `.txt` (and optionally `.html`)          |
| 2     | PyTorch    | `--profile torch`  | CPU + GPU ops/kernels    | `.json` (trace), `_summary.txt`                   |
| 3     | ROCm       | `--profile rocprof`| GPU (HIP) kernels/copies | rocprof files in same dir (e.g. `.csv`)           |

---

## Documentation

- **profile.md** (this file): full plan and implementation order.
- **BENCHMARK_README.md**: add a short “Profiling” section that points to profile.md and lists:
  - `--profile python | torch | rocprof`
  - One example command per mode.
  - Where outputs are written: **`output/benchmarks/<model_name_short>/`** for all profilers.

---

## Next steps

1. Implement **Phase 1** (Python profiler) in `bench.py` and update BENCHMARK_README.
2. Implement **Phase 2** (PyTorch profiler) in `bench.py` and document ROCm-specific behaviour if any.
3. Implement **Phase 3** (rocprof subprocess + move/cwd so rocprof writes to `output/benchmarks/<model_name_short>/`) and finalise BENCHMARK_README profiling section.
