import os
import json
import argparse
import random
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# Allow importing chat from repo root (same as bench.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chat import Chat

SYSTEM_PROMPT = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"


def format_prompt(question):
    """Build a single prompt string for the Chat interface (text only)."""
    return SYSTEM_PROMPT + "\n\n" + question["question"]


def run_question(chat, question, args):
    """Run one question through the Chat interface. Returns (id, content, usage) or None on failure."""
    prompt = format_prompt(question)
    try:
        max_tokens = args.max_completion_tokens if args.max_completion_tokens is not None else 512
        response, nof_tokens = chat(
            prompt,
            max_tokens=max_tokens,
            temperature=args.temperature,
        )
    except Exception as e:
        print("Error:", e)
        return None
    if response is None:
        return None
    usage = {"completion_tokens": nof_tokens}
    return question["id"], response, usage


def main(args):
    dataset = load_dataset(args.dataset, split="test").to_dict()

    # Convert to list of dicts for iteration
    questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]

    # Restrict to text-only first so max_samples is exact (no crop from dropping image questions)
    questions = [q for q in questions if not q.get("image")]

    if args.max_samples is not None:
        n_take = min(args.max_samples, len(questions))
        if args.seed is not None:
            rng = random.Random(args.seed)
            questions = rng.sample(questions, n_take)
        else:
            questions = questions[:n_take]

    model_path = args.model
    if not os.path.isabs(model_path) and "/" not in model_path and not model_path.startswith("~"):
        model_path = os.path.expanduser(f"~/models/{model_path}")
    else:
        model_path = os.path.expanduser(model_path)

    model_name = Path(model_path).name
    output_filepath = f"hle_{model_name}.json"

    if os.path.exists(output_filepath):
        with open(output_filepath, "r") as f:
            predictions = json.load(f)
        questions = [q for q in questions if q["id"] not in predictions]
    else:
        predictions = {}

    if not questions:
        print("No new questions to run. All predictions already cached or all remaining have images.")
        return

    print(f"Model: {model_name}\nRunning evaluation on {len(questions)} questions via Chat interface...\n")
    chat = Chat(model_path)

    for question in tqdm(questions, desc="Predictions"):
        result = run_question(chat, question, args)
        if result is None:
            continue
        unique_id, response, usage = result
        predictions[unique_id] = {
            "model": model_name,
            "response": response,
            "usage": usage,
        }

    with open(output_filepath, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions saved to {output_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="HLE HuggingFace dataset name")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or short name (e.g. 7b_instruct_v.3 for ~/models/7b_instruct_v.3)",
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=None,
        help="Max completion tokens (default 512). Recommended to avoid model collapse.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling.")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Use exactly this many questions (text-only; images excluded before applying).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sampling questions when --max_samples is set (optional).",
    )
    args = parser.parse_args()
    main(args)

# python3 run_model_predictions.py --dataset cais/hle --model 8x7b_instruct_v.1 --max_samples 200 --seed 0 --temperature 0 --max_completion_tokens 2048