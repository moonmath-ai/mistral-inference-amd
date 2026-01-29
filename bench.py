import time
import os
import numpy as np
from openai import OpenAI
import requests

# 1. Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

METRICS_URL = "http://localhost:8000/metrics"
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")


# MODEL_SHORT = f"{MODEL_NAME.split('/')[-1]}_single_gpu"
MODEL_SHORT = f"{MODEL_NAME.split('/')[-1]}_dual_gpu"

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

def run_vllm_benchmark(prompt_list):
    results_summary = []

    # warmup
    warmup_prompt = "Write a haiku about a cat that walks on his head."

    print(f"Warming up model... ({warmup_prompt[:20]}...)")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": warmup_prompt}]
    )
    response_text = response.choices[0].message.content
    print(f"Warmed up model - response:\n{response_text}\n\n")

    for i, prompt in enumerate(prompt_list):
        # 1. Take 'Before' snapshot
        before = get_vllm_internal_metrics()

        response = client.chat.completions.create(
            model=MODEL_NAME,
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

        print(f"Prompt {i+1} Done: {int(out_tokens)} tokens | TPS: {tps:.2f} | TTFT: {ttft:.3f}s")
        clean_name = "".join(c for c in prompt[:20] if c.isalnum() or c==' ').replace(" ", "_")
        with open(f"output/benchmarks/model_{MODEL_SHORT}_output_{i}_{clean_name}.txt", "w") as f:
            f.write(f"Prompt: {prompt}\n\nOutput:\n{response_text}")

    time_to_first_token_avg = np.mean([result["ttft"] for result in results_summary])
    time_to_first_token_std = np.std([result["ttft"] for result in results_summary])
    tokens_per_second_avg = np.mean([result["tps"] for result in results_summary])
    tokens_per_second_std = np.std([result["tps"] for result in results_summary])

    output_summary = f"\n{"="*50}\nModel: {MODEL_NAME}\n{"-"*50}\nTime To First Token: {time_to_first_token_avg:.5f}s ± {time_to_first_token_std:.6f}s\nTokens Per Second: {tokens_per_second_avg:.2f} ± {tokens_per_second_std:.2f} t/s\n{"="*50}\n"
    print(output_summary)

    with open(f"output/benchmarks/model_{MODEL_SHORT}_summary.txt", "w") as f:
        f.write(output_summary)

def run_our_benchmark(prompt_list):
    raw_latencies, output_tokens_counts, ttfts, tps_list = [], [], [], []

    print(f"Model: {MODEL_NAME}\nStarting benchmark on {len(prompt_list)} prompts...\n")

    for i, prompt in enumerate(prompt_list):
        start_time = time.perf_counter()
        first_token_time = None
        tokens_generated = 0
        full_response_text = "" # <--- INITIALIZE ACCUMULATOR
        
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start_time
                
                full_response_text += content # <--- ACCUMULATE TEXT
                tokens_generated += 1
        
        end_time = time.perf_counter()
        total_latency = end_time - start_time
        
        # Store Metrics
        raw_latencies.append(round(total_latency, 3))
        output_tokens_counts.append(tokens_generated)
        ttfts.append(first_token_time)
        tps_list.append(tokens_generated / total_latency)
        
        print(f"Prompt {i+1} Done: {tokens_generated} tokens in {total_latency:.2f}s")

        # Create safe filename
        clean_name = "".join(c for c in prompt[:20] if c.isalnum() or c==' ').replace(" ", "_")
        file_path = f"output/benchmarks/model_output_{i}_{clean_name}.txt"
        
        with open(file_path, "w") as f:
            f.write(f"Prompt: {prompt}\n\nModel Output:\n{full_response_text}") # <--- USE ACCUMULATED TEXT
    
    # ... (Rest of your metrics display code)
    print("\nAll outputs saved to output/benchmarks/")
    
    # 3. Displaying the Data
    print("\n" + "="*50)
    print("RAW DATA LISTS")
    print("-" * 50)
    print(f"Total Latencies (s):  {raw_latencies}")
    print(f"Output Token Counts: {output_tokens_counts}")
    
    print("\n" + "="*50)
    print("AGGREGATE HARDWARE METRICS")
    print("-" * 50)
    # We use these because they are length-independent 'quality of service' metrics
    print(f"{'Metric':<25} | {'Avg ± Std':<15}")
    print("-" * 50)
    
    # TTFT: How fast the hardware responds (Prefill speed)
    print(f"{'Time to 1st Token (s)':<25} | {np.mean(ttfts):.3f} ± {np.std(ttfts):.3f}")
    
    # TPS: How fast the hardware generates (Decoding speed)
    print(f"{'Tokens Per Second':<25} | {np.mean(tps_list):.2f} ± {np.std(tps_list):.2f}")
    print("="*50)

if __name__ == "__main__":
    run_vllm_benchmark(prompts)