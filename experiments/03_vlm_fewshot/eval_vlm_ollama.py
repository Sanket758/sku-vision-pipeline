"""Evaluate VLM (via Ollama) on zero-shot and few-shot shelf detection.

Usage:
    python eval_vlm_ollama.py --mode zero_shot --image ../../Dataset/raw/kaufland/IMG20260601171129.jpg
    python eval_vlm_ollama.py --mode few_shot --image test.jpg --shots ../../Dataset/processed_yolo/images/train/*.jpg
"""

import argparse
import sys
import time
import requests
import base64
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import hardware_utils as hw
from shared.metrics_logger import MetricsLogger

def parse_args():
    parser = argparse.ArgumentParser(description="VLM Few-Shot Evaluation with Ollama")
    parser.add_argument("--mode", type=str, choices=["zero_shot", "few_shot"], default="zero_shot")
    parser.add_argument("--image", type=str, required=True, help="Query image path")
    parser.add_argument("--shots", type=str, nargs="+", default=[], help="Few-shot example image paths")
    parser.add_argument("--num_shots", type=int, default=3, help="Number of shot examples to use")
    parser.add_argument("--model", type=str, default="moondream:latest", help="Ollama model to use")
    parser.add_argument("--prompt_file", type=str, default=None, help="Override prompt template path")
    parser.add_argument("--output", type=str, default=None, help="Output results path")
    parser.add_argument("--batch", type=str, default=None, help="Batch evaluate a directory of images")
    return parser.parse_args()

def load_prompt(mode: str, num_shots: int = 3) -> str:
    prompt_dir = Path(__file__).parent / "prompts"
    if mode == "zero_shot":
        pf = prompt_dir / "zero_shot_detection.txt"
    else:
        pf = prompt_dir / "few_shot_detection.txt"

    if not pf.exists():
        return "Describe all products visible on this supermarket shelf."

    prompt = pf.read_text().strip()
    if mode == "few_shot":
        prompt = prompt.replace("{num_shots}", str(num_shots))
    return prompt

def encode_image(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    args = parse_args()

    query_path = Path(args.image)
    if not args.batch and not query_path.exists():
        print(f"ERROR: Image not found: {query_path}")
        sys.exit(1)

    print("=" * 60)
    print(hw.hw_summary())
    print(f"Mode:  {args.mode}")
    if not args.batch:
        print(f"Image: {query_path}")
    print(f"Model: {args.model} (Ollama)")
    print("=" * 60)

    logger = MetricsLogger(f"vlm_ollama_{args.mode}", config=vars(args))
    prompt = args.prompt_file or load_prompt(args.mode, args.num_shots)
    logger.log_hyperparams({"model": args.model, "mode": args.mode, "prompt_length": len(prompt)})

    # Pull or check model
    print(f"\nChecking Ollama model {args.model}...")
    t0 = time.time()
    try:
        models_resp = requests.get("http://localhost:11434/api/tags")
        models_resp.raise_for_status()
        available_models = [m["name"] for m in models_resp.json()["models"]]
        if args.model not in available_models:
            print(f"Model {args.model} not found locally. Please run 'ollama pull {args.model}'")
            sys.exit(1)
    except Exception as e:
        print(f"Could not connect to Ollama API: {e}")
        sys.exit(1)

    load_time = time.time() - t0
    logger.log_metric("model_check_time_s", load_time)

    if args.mode == "few_shot" and args.shots:
        shot_paths = [Path(s) for s in args.shots[:args.num_shots] if Path(s).exists()]
        print(f"Using {len(shot_paths)} shot examples")
    else:
        shot_paths = []

    def run_inference(img_path: Path) -> dict:
        t0 = time.time()
        
        images_b64 = []
        if args.mode == "few_shot":
            for sp in shot_paths:
                images_b64.append(encode_image(sp))
        images_b64.append(encode_image(img_path))
        
        # Build prompt: for few-shot we might need a specific structure, but simple appending of images might work with Ollama depending on the model.
        # Most VLMs in Ollama accept multiple images. We send the prompt + images.
        payload = {
            "model": args.model,
            "prompt": prompt,
            "images": images_b64,
            "stream": False
        }
        
        try:
            resp = requests.post("http://localhost:11434/api/generate", json=payload)
            resp.raise_for_status()
            result_text = resp.json()["response"]
            eval_count = resp.json().get("eval_count", 0)
        except Exception as e:
            result_text = f"API Error: {e}"
            eval_count = 0

        elapsed = time.time() - t0
        tokens = len(result_text.split()) if isinstance(result_text, str) else 0

        return {"text": result_text, "tokens_s": tokens / elapsed if elapsed > 0 else 0, "time_s": elapsed}

    if args.batch:
        batch_dir = Path(args.batch)
        images = sorted(batch_dir.glob("*.jpg")) + sorted(batch_dir.glob("*.png"))
        if not images:
            print(f"No images found in {batch_dir}")
            sys.exit(1)
            
        results = []

        for img_path in images:
            print(f"\n--- {img_path.name} ---")
            res = run_inference(img_path)
            print(res["text"])
            results.append({"image": img_path.name, **res})

        avg_time = sum(r["time_s"] for r in results) / len(results)
        avg_tokens_s = sum(r["tokens_s"] for r in results) / len(results)

        logger.log_metrics({
            "batch_size": len(results),
            "avg_inference_time_s": round(avg_time, 3),
            "avg_tokens_per_sec": round(avg_tokens_s, 1),
        })
        logger._write_file("batch_results.json", results)

        print(f"\nBatch Summary ({len(results)} images):")
        print(f"  Avg inference time: {avg_time:.2f}s")
        print(f"  Avg tokens/sec:     {avg_tokens_s:.1f}")

    else:
        print(f"\n--- {query_path.name} ---")
        result = run_inference(query_path)
        print(result["text"])

        logger.log_metrics({
            "inference_time_s": round(result["time_s"], 3),
            "tokens_per_sec": round(result["tokens_s"], 1),
            "response_length": len(result["text"]),
        })
        logger.save_artifact(str(query_path), f"query_{query_path.name}")

        print(f"\n  Time: {result['time_s']:.2f}s")
        print(f"  Speed: {result['tokens_s']:.1f} tokens/sec")

    logger.flush()
    print(f"\nResults saved to: {logger.get_run_path()}")

if __name__ == "__main__":
    main()
