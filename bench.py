import argparse
import cProfile
import os
import pstats
import time

import numpy as np
import requests
import torch
from openai import OpenAI

from chat import Chat
from mistral_inference.timing import StageTiming

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="7b_instruct_v.3")
parser.add_argument("--vllm", action="store_true")
parser.add_argument("--multigpu", action="store_true")
parser.add_argument(
    "--profile",
    type=str,
    choices=("none", "python", "torch", "rocprof"),
    default="none",
    help="Profiling mode: none (default), python (cProfile), torch, or rocprof.",
)
parser.add_argument(
    "--stage-timing",
    action="store_true",
    help="Collect stage-level timing (tokenization/prefill/decode/attention/MoE) for native path.",
)
args = parser.parse_args()

MODEL_FULL_NAMES = {
    "7b_instruct_v.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "8x7b_instruct_v.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "8x22b_instruct_v.3": "mistralai/Mixtral-8x22B-Instruct-v0.1",
}

METRICS_URL = "http://localhost:8000/metrics"
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Ensure output directory exists
os.makedirs("output/benchmarks", exist_ok=True)

def get_vllm_internal_metrics():
    """Parses the Prometheus text format into a dictionary."""
    try:
        response = requests.get(METRICS_URL)
        metrics = {}
        for line in response.text.split('\n'):
            if line.startswith("vllm:") and not line.startswith("#"):
                # Format: metric_name{labels} value
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0].split('{')[0]
                    value = float(parts[-1])
                    metrics[name] = value
        return metrics
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return {}

prompts = [
    "Say 'Hello' and nothing else.",
    "Write a poem about coding, with three verses and a chorus.",
    "Explain quantum physics to a five-year-old in three paragraphs.",
    "Summarize the plot of Inception.",
    "Write a 200-word story about a space-faring armadillo.",
    "Explain the importance of the Navier-Stokes equations for fluid dynamics.",
# ]

# prompts = [
    "How many wood can a woodchuck protect if the industrial machine comes to the forest? Present startegies for optimal defense.",
    "If the entire world is a very narrow bridge, can we approximate it to be a one dimensional line? How would this affect the physics of the world?",
    "Translate 'where is the library?' to greek and turkish.",
    "Do you dream of electric sheep?",
    "Who keeps their coffee cold for a day?"
]

def run_vllm_benchmark(model_name, prompt_list):
    results_summary = []

    output_suffix = "_multigpu" if args.multigpu else "_single_gpu"

    model_name = MODEL_FULL_NAMES[model_name]
    model_name_short = model_name.split("/")[-1]

    os.makedirs(f"output/benchmarks/{model_name_short}", exist_ok=True)

    # warmup
    warmup_prompt = "Write a haiku about a cat that walks on his head."

    print(f"Warming up model... ({warmup_prompt[:20]}...)")
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": warmup_prompt}]
    )
    response_text = response.choices[0].message.content
    print(f"Warmed up model - response:\n{response_text}\n\n")

    out_run_str = ""

    for i, prompt in enumerate(prompt_list):
        # 1. Take 'Before' snapshot
        before = get_vllm_internal_metrics()

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.choices[0].message.content
        
        # 2. Take 'After' snapshot
        after = get_vllm_internal_metrics()

        # --- CALCULATE TOKENS ---
        # Counter metric: vllm:generation_tokens_total
        out_tokens = after.get("vllm:generation_tokens_total", 0) - before.get("vllm:generation_tokens_total", 0)

        # --- CALCULATE TTFT ---
        # Histogram: vllm:time_to_first_token_seconds_sum / _count
        ttft_sum_delta = after.get("vllm:time_to_first_token_seconds_sum", 0) - before.get("vllm:time_to_first_token_seconds_sum", 0)
        ttft_count_delta = after.get("vllm:time_to_first_token_seconds_count", 0) - before.get("vllm:time_to_first_token_seconds_count", 0)
        ttft = ttft_sum_delta / ttft_count_delta if ttft_count_delta > 0 else 0

        # --- CALCULATE LATENCY & TPS ---
        # Histogram: vllm:e2e_request_latency_seconds_sum
        latency_sum_delta = after.get("vllm:e2e_request_latency_seconds_sum", 0) - before.get("vllm:e2e_request_latency_seconds_sum", 0)
        tps = out_tokens / latency_sum_delta if latency_sum_delta > 0 else 0

        results_summary.append({
            "nof_tokens": out_tokens,
            "tps": tps,
            "ttft": ttft,
            "latency": latency_sum_delta
        })

        prompt_stat_str = f"Prompt {i+1} Done: {int(out_tokens)} tokens\t| TPS: {tps:.2f}\t| TTFT: {ttft:.3f}s"
        print(prompt_stat_str)
        out_run_str += prompt_stat_str + "\n"
        clean_name = "".join(c for c in prompt[:20] if c.isalnum() or c==' ').replace(" ", "_")
        with open(f"output/benchmarks/{model_name_short}/vllm_output_{i}_{clean_name}_{output_suffix}.txt", "w") as f:
            f.write(f"Prompt: {prompt}\n\nOutput:\n{response_text}")

    time_to_first_token_avg = np.mean([result["ttft"] for result in results_summary])
    time_to_first_token_std = np.std([result["ttft"] for result in results_summary])
    tokens_per_second_avg = np.mean([result["tps"] for result in results_summary])
    tokens_per_second_std = np.std([result["tps"] for result in results_summary])

    output_summary = f"\n\n{"="*50}\nModel: {model_name}_{output_suffix}\n{"-"*50}\nTime To First Token: {time_to_first_token_avg:.5f}s ± {time_to_first_token_std:.6f}s\nTokens Per Second: {tokens_per_second_avg:.2f} ± {tokens_per_second_std:.2f} t/s\n{"="*50}\n"
    print(output_summary)

    with open(f"output/benchmarks/{model_name_short}/vllm_summary_{output_suffix}.txt", "w") as f:
        f.write(out_run_str + output_summary)

def run_mistral_benchmark(model_name, prompt_list, prefix="native"):
    raw_latencies, output_tokens_counts, tps_list = [], [], []
    stage_timings: list[StageTiming] = []

    model_path =os.path.expanduser(f"~/models/{model_name}")
    model_name = MODEL_FULL_NAMES[model_name]
    model_name_short = model_name.split("/")[-1]

    print(f"$$ Model: {model_name}, {model_name_short}")

    output_suffix = "_multigpu" if args.multigpu else "_single_gpu"

    chat = Chat(model_path)

    os.makedirs(f"output/benchmarks/{model_name_short}", exist_ok=True)

    out_run_str = ""

    print(f"Model: {model_name_short}\nStarting benchmark on {len(prompt_list)} prompts...\n")

    # warmup
    warmup_prompt = "Write a haiku about a cat that walks on his head."

    print(f"Warming up model... ({warmup_prompt[:20]}...)")
    response, nof_tokens = chat(warmup_prompt)
    print(f"Warmed up model - response:\n{response}\n\n")

    for i, prompt in enumerate(prompt_list):
        start_time = time.perf_counter()

        if args.stage_timing:
            response, nof_tokens, timing = chat(prompt, return_timing=True)
            stage_timings.append(timing)
        else:
            response, nof_tokens = chat(prompt)
        
        end_time = time.perf_counter()
        latency = end_time - start_time

        raw_latencies.append(latency)
        output_tokens_counts.append(nof_tokens)
        tps_list.append(nof_tokens / latency)

        prompt_stat_str = f"Prompt {i+1} Done: {nof_tokens} tokens in {latency:.3f}s"
        print(prompt_stat_str)
        out_run_str += prompt_stat_str + "\n"
        if args.stage_timing:
            stage_str = (
                f"  tokenization={timing.tokenization_ms:.2f}ms | "
                f"prefill={timing.prefill_ms:.2f}ms ({timing.prefill_tps:.2f} tok/s) | "
                f"ttft={timing.ttft_ms:.2f}ms | "
                f"decode={timing.decode_ms:.2f}ms ({timing.decode_tps:.2f} tok/s) | "
                f"attn_prefill={timing.attn_prefill_ms:.2f}ms | "
                f"attn_decode={timing.attn_decode_ms:.2f}ms | "
                f"moe_dispatch+combine_decode={timing.moe_dispatch_combine_decode_ms:.2f}ms | "
                f"moe_expert_gemm_decode={timing.moe_expert_gemm_decode_ms:.2f}ms"
            )
            print(stage_str)
            out_run_str += stage_str + "\n"
            moe_breakdown_str = (
                f"    moe_gate={timing.moe_gate_decode_ms:.2f}ms | "
                f"moe_topk={timing.moe_topk_decode_ms:.2f}ms | "
                f"moe_softmax={timing.moe_softmax_decode_ms:.2f}ms | "
                f"moe_route_extract={timing.moe_route_extract_decode_ms:.2f}ms | "
                f"moe_gather={timing.moe_gather_decode_ms:.2f}ms | "
                f"moe_combine={timing.moe_combine_decode_ms:.2f}ms"
            )
            print(moe_breakdown_str)
            out_run_str += moe_breakdown_str + "\n"

        # Create safe filename
        clean_name = "".join(c for c in prompt[:20] if c.isalnum() or c==' ').replace(" ", "_")
        file_path = f"output/benchmarks/{model_name_short}/{prefix}_output_{i}_{clean_name}_{output_suffix}.txt"
        
        with open(file_path, "w") as f:
            f.write(f"Prompt: {prompt}\n\nModel Output:\n{response}") # <--- USE ACCUMULATED TEXT
    
    # ... (Rest of your metrics display code)
    print("\nAll outputs saved to output/benchmarks/")

    output_summary = f"\n\n{"="*50}\nModel: {model_name}_{output_suffix}\n{"-"*50}\nTokens Per Second: {np.mean(tps_list):.2f} ± {np.std(tps_list):.2f}\n{"="*50}\n"
    if args.stage_timing and stage_timings:
        output_summary += (
            f"Stage timing averages across prompts:\n"
            f"  Tokenization: {np.mean([t.tokenization_ms for t in stage_timings]):.2f} ms\n"
            f"  Prefill total: {np.mean([t.prefill_ms for t in stage_timings]):.2f} ms\n"
            f"  Prefill TPS: {np.mean([t.prefill_tps for t in stage_timings]):.2f} tok/s\n"
            f"  TTFT: {np.mean([t.ttft_ms for t in stage_timings]):.2f} ms\n"
            f"  Decode total: {np.mean([t.decode_ms for t in stage_timings]):.2f} ms\n"
            f"  Decode TPS: {np.mean([t.decode_tps for t in stage_timings]):.2f} tok/s\n"
            f"  Attention prefill: {np.mean([t.attn_prefill_ms for t in stage_timings]):.2f} ms\n"
            f"  Attention decode: {np.mean([t.attn_decode_ms for t in stage_timings]):.2f} ms\n"
            f"  MoE dispatch+combine (decode): {np.mean([t.moe_dispatch_combine_decode_ms for t in stage_timings]):.2f} ms\n"
            f"  MoE expert GEMM (decode): {np.mean([t.moe_expert_gemm_decode_ms for t in stage_timings]):.2f} ms\n"
            f"  MoE gate (decode): {np.mean([t.moe_gate_decode_ms for t in stage_timings]):.2f} ms\n"
            f"  MoE topk (decode): {np.mean([t.moe_topk_decode_ms for t in stage_timings]):.2f} ms\n"
            f"  MoE softmax (decode): {np.mean([t.moe_softmax_decode_ms for t in stage_timings]):.2f} ms\n"
            f"  MoE route_extract (decode): {np.mean([t.moe_route_extract_decode_ms for t in stage_timings]):.2f} ms\n"
            f"  MoE gather (decode): {np.mean([t.moe_gather_decode_ms for t in stage_timings]):.2f} ms\n"
            f"  MoE combine (decode): {np.mean([t.moe_combine_decode_ms for t in stage_timings]):.2f} ms\n"
        )
    print(output_summary)

    with open(f"output/benchmarks/{model_name_short}/{prefix}_summary_{output_suffix}.txt", "w") as f:
        f.write(out_run_str + output_summary)

if __name__ == "__main__":
    num_gpus_str = "multigpu" if args.multigpu else "single_gpu"
    if args.vllm:
        print(f"Benchmarking {num_gpus_str} with vLLM...")
        run_vllm_benchmark(args.model, prompts)
    else:
        if args.profile == "python":
            print(f"Benchmarking {num_gpus_str} with Mistral functions (Python profiler)...")
            prof = cProfile.Profile()
            prof.enable()
            run_mistral_benchmark(args.model, prompts)
            prof.disable()
            model_name_short = MODEL_FULL_NAMES[args.model].split("/")[-1]
            profile_suffix = "multigpu" if args.multigpu else "single_gpu"
            prof_dir = f"output/benchmarks/{model_name_short}"
            os.makedirs(prof_dir, exist_ok=True)
            prof_path = os.path.join(prof_dir, f"profile_python_{profile_suffix}.prof")
            txt_path = os.path.join(prof_dir, f"profile_python_{profile_suffix}.txt")
            prof.dump_stats(prof_path)
            with open(txt_path, "w") as f:
                ps = pstats.Stats(prof, stream=f)
                ps.strip_dirs()
                f.write("=== By cumulative time (call trees) ===\n\n")
                ps.sort_stats("cumulative").print_stats()
                f.write("\n\n=== By self time (leaf / heavy functions only) ===\n\n")
                ps.sort_stats("time").print_stats()
            print(f"Python profile saved: {prof_path}")
            print(f"Python profile summary: {txt_path}")
        elif args.profile == "torch":
            print(f"Benchmarking {num_gpus_str} with Mistral functions (PyTorch profiler)...")
            model_name_short = MODEL_FULL_NAMES[args.model].split("/")[-1]
            profile_suffix = "multigpu" if args.multigpu else "single_gpu"
            prof_dir = f"output/benchmarks/{model_name_short}"
            os.makedirs(prof_dir, exist_ok=True)
            json_path = os.path.join(prof_dir, f"profile_torch_{profile_suffix}.json")
            summary_path = os.path.join(prof_dir, f"profile_torch_{profile_suffix}_summary.txt")
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profile_prompts = prompts[:2]
            print(f"PyTorch profiler: using {len(profile_prompts)} prompt(s) for smaller trace.\n")
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=False,
                with_stack=False,
            ) as prof:
                run_mistral_benchmark(args.model, profile_prompts)
            prof.export_chrome_trace(json_path)
            with open(summary_path, "w") as f:
                f.write("=== By self CPU time ===\n\n")
                f.write(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=100))
                if torch.cuda.is_available():
                    f.write("\n\n=== By self GPU time ===\n\n")
                    f.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=100))
                f.write("\n\n=== By input shape (matrix sizes / batching) ===\n\n")
                f.write(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=150))
            print(f"PyTorch trace saved: {json_path}")
            print(f"PyTorch summary saved: {summary_path}")
        else:
            print(f"Benchmarking {num_gpus_str} with Mistral functions...")
            run_mistral_benchmark(args.model, prompts)
    