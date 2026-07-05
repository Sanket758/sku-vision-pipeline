"""Benchmark vision model latency on 20 test crops.
Tests: moondream:1.8b, optionally llava:7b if VRAM allows.

Usage:
    python benchmark_vlm.py                    # default: moondream
    python benchmark_vlm.py --model llava:7b   # test llava
    python benchmark_vlm.py --all              # test all available
"""

import argparse
import base64
import json
import time
from pathlib import Path

import requests

CROP_DIR = Path("/home/sanket758/Education/BSBI/Masters-Thesis/Dataset/processed_retrieval")
OLLAMA_URL = "http://localhost:11434/api/chat"

TEST_CROPS = [
    "IMG20260611210113_det051.jpg", "IMG20260611205035_det073.jpg",
    "IMG20260611210113_det062.jpg", "IMG20260611210104_det046.jpg",
    "IMG20260611210043_det018.jpg", "IMG20260611210131_det048.jpg",
    "IMG20260611205055_det051.jpg", "IMG20260611210113_det041.jpg",
    "IMG20260611205055_det045.jpg", "IMG20260611205114_det060.jpg",
    "IMG20260611205118_det028.jpg", "IMG20260611210052_det140.jpg",
    "IMG20260611205125_det080.jpg", "IMG20260611210131_det043.jpg",
    "IMG20260611205041_det011.jpg", "IMG20260611205111_det153.jpg",
    "IMG20260611205035_det060.jpg", "IMG20260611210104_det099.jpg",
    "IMG20260611205048_det029.jpg", "IMG20260611205114_det051.jpg",
]

PRODUCT_PROMPT = (
    "Look at this retail product image. Identify the BRAND and PRODUCT TYPE. "
    "Respond with ONLY the brand name and product name, like this:\n"
    "BRAND: <brand>\n"
    "PRODUCT: <product name+type>\n"
    "If you can't identify it, respond: BRAND: UNKNOWN"
)

OCR_PROMPT = (
    "Read all text on this product label carefully. Return all product text exactly as written, "
    "including brand name, product name, flavour, size, and任何 other visible text. "
    "If you cannot read anything, return 'NO_TEXT_FOUND'."
)


def test_model(model_name, prompt, crops, max_crops=20):
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"Prompt type: {'OCR' if 'Read all text' in prompt else 'Brand/Product'}")
    print(f"Crops: {min(max_crops, len(crops))}")
    print(f"{'='*60}")

    times = []
    results = []
    errors = 0

    for i, fname in enumerate(crops[:max_crops]):
        img_path = CROP_DIR / fname
        if not img_path.exists():
            print(f"  [{i+1}/{max_crops}] SKIP {fname} — file not found")
            continue

        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        img_size_mb = len(b64) * 3 / 4 / 1024 / 1024

        t0 = time.time()
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt, "images": [b64]}],
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
                timeout=300,
            )
            resp.raise_for_status()
            text = resp.json()["message"]["content"].strip()
            elapsed = time.time() - t0
            times.append(elapsed)

            results.append({
                "crop": fname,
                "model": model_name,
                "response": text,
                "time_s": round(elapsed, 2),
                "img_size_mb": round(img_size_mb, 1),
            })

            print(
                f"  [{i+1}/{max_crops}] {elapsed:5.1f}s | "
                f"{text[:60].replace(chr(10), ' '):<60}"
            )
        except Exception as e:
            errors += 1
            elapsed = time.time() - t0
            print(f"  [{i+1}/{max_crops}] ERROR after {elapsed:.0f}s: {e}")

    # Stats
    if times:
        import numpy as np
        print(f"\n{'='*60}")
        print(f"Results for {model_name}:")
        print(f"  Completed: {len(times)}/{max_crops - errors + errors} (errors: {errors})")
        print(f"  Mean:     {np.mean(times):.1f}s")
        print(f"  Median:   {np.median(times):.1f}s")
        print(f"  Min:      {min(times):.1f}s")
        print(f"  Max:      {max(times):.1f}s")
        print(f"  Total:    {sum(times):.1f}s ({sum(times)/60:.1f} min)")
        print(f"  Per crop: {np.mean(times):.1f}s")
        estimate_3000 = np.mean(times) * 3005 / 60
        estimate_full = np.mean(times) * 36923 / 3600
        print(f"  Est. 3,005 new crops: {estimate_3000:.0f} min ({estimate_3000/60:.1f} hr)")
        print(f"  Est. 36,923 ALL crops: {estimate_full:.1f} hr")
    else:
        print("\n  No successful results.")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="moondream:latest",
                        choices=["moondream:latest", "llava:7b", "granite4.1:3b"])
    parser.add_argument("--ocr", action="store_true", help="Use OCR prompt instead of brand/product")
    parser.add_argument("--max", type=int, default=20, help="Max crops to test")
    parser.add_argument("--all", action="store_true", help="Test all available models")
    args = parser.parse_args()

    prompt = OCR_PROMPT if args.ocr else PRODUCT_PROMPT

    if args.all:
        for model in ["moondream:latest", "granite4.1:3b", "llava:7b"]:
            test_model(model, prompt, TEST_CROPS, args.max)
    else:
        results = test_model(args.model, prompt, TEST_CROPS, args.max)

        # Save results
        outpath = Path(f"experiments/annotation_tool/data/vlm_benchmark_{args.model.replace(':', '_')}.json")
        with open(outpath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
