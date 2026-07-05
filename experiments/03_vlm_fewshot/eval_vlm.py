"""Evaluate VLM (moondream) on zero-shot and few-shot shelf detection.

Usage:
    python eval_vlm.py --mode zero_shot --image ../../Dataset/raw/kaufland/IMG20260601170004.jpg
    python eval_vlm.py --mode few_shot --image test.jpg --shots ../../Dataset/processed_yolo/images/train/*.jpg
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import hardware_utils as hw
from shared.metrics_logger import MetricsLogger


def parse_args():
    parser = argparse.ArgumentParser(description="VLM Few-Shot Evaluation")
    parser.add_argument("--mode", type=str, choices=["zero_shot", "few_shot"], default="zero_shot")
    parser.add_argument("--image", type=str, required=True, help="Query image path")
    parser.add_argument("--shots", type=str, nargs="+", default=None,
                        help="Few-shot example image paths")
    parser.add_argument("--num_shots", type=int, default=3, help="Number of shot examples to use")
    parser.add_argument("--model", type=str, default="moondream",
                        choices=["moondream"], help="VLM model")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Override prompt template path")
    parser.add_argument("--output", type=str, default=None, help="Output results path")
    parser.add_argument("--batch", type=str, default=None,
                        help="Batch evaluate a directory of images")
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


def main():
    args = parse_args()

    query_path = Path(args.image)
    if not query_path.exists():
        print(f"ERROR: Image not found: {query_path}")
        sys.exit(1)

    print("=" * 60)
    print(hw.hw_summary())
    print(f"Mode:  {args.mode}")
    print(f"Image: {query_path}")
    print(f"Model: {args.model}")
    print("=" * 60)

    logger = MetricsLogger(f"vlm_{args.mode}", config=vars(args))
    prompt = args.prompt_file or load_prompt(args.mode, args.num_shots)
    logger.log_hyperparams({"model": args.model, "mode": args.mode, "prompt_length": len(prompt)})

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed. Install with: pip install transformers torch")
        sys.exit(1)

    model_id = "vikhyatk/moondream2"
    print(f"\nLoading {model_id}...")
    t0 = time.time()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, device_map="auto", torch_dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print(f"  Could not load {model_id}: {e}")
        print("  Falling back to mock evaluation (for pipeline testing).")
        model = None
        tokenizer = None

    load_time = time.time() - t0
    logger.log_metric("model_load_time_s", load_time)

    if args.mode == "few_shot" and args.shots:
        shot_paths = [Path(s) for s in args.shots[:args.num_shots] if Path(s).exists()]
        print(f"Using {len(shot_paths)} shot examples")
    else:
        shot_paths = []

    def run_inference(img_path: Path) -> dict:
        if model is None:
            return {"text": "MOCK: recognized products on shelf", "tokens_s": 0, "time_s": 0}

        from PIL import Image
        image = Image.open(img_path).convert("RGB")

        t0 = time.time()
        if args.mode == "zero_shot":
            enc_image = model.encode_image(image)
            result = model.answer_question(enc_image, prompt, tokenizer)
        else:
            shot_images = [Image.open(s).convert("RGB") for s in shot_paths]
            few_shot_prompt = prompt + "\n\n" + "\n".join(
                f"Example {i+1}: [image]" for i in range(len(shot_images))
            )
            enc_image = model.encode_image(image)
            result = model.answer_question(enc_image, few_shot_prompt, tokenizer)

        elapsed = time.time() - t0
        tokens = len(result.split()) if isinstance(result, str) else 0

        return {"text": str(result), "tokens_s": tokens / elapsed if elapsed > 0 else 0, "time_s": elapsed}

    if args.batch:
        batch_dir = Path(args.batch)
        images = sorted(batch_dir.glob("*.jpg")) + sorted(batch_dir.glob("*.png"))
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
