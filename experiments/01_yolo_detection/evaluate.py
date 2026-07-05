"""Local evaluation script for YOLO models.
Generates thesis-level metrics: mAP charts, confusion matrices, entropy distributions.

Usage:
    python evaluate.py --model runs/train/exp/weights/best.pt --data ../../Dataset/processed_yolo/data.yaml
    python evaluate.py --model runs/train/exp/weights/best.pt --data ../../Dataset/processed_yolo/data.yaml --name yolov8_eval
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import hardware_utils as hw
from shared.metrics_logger import DetMetricsLogger
from shared.dataset_utils import YOLO_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Evaluation for Thesis Metrics")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for mAP (use 0.001 for proper mAP, 0.25 for deployment)")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS (0.6 for mAP, 0.45 for deployment)")
    parser.add_argument("--name", type=str, default=None, help="Evaluation run name")
    parser.add_argument("--save_json", action="store_true", default=True, help="Save JSON results")
    parser.add_argument("--plot", action="store_true", default=True, help="Generate plots")
    return parser.parse_args()


def compute_entropy(confidences: np.ndarray) -> float:
    confidences = np.clip(confidences, 1e-10, 1.0)
    entropy = -np.sum(confidences * np.log(confidences)) / len(confidences)
    return float(entropy)


def main():
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    data_yaml = args.data or str(YOLO_DIR / "data.yaml")
    if not Path(data_yaml).exists():
        print(f"ERROR: data.yaml not found at {data_yaml}")
        sys.exit(1)

    print("=" * 60)
    print(hw.hw_summary())
    print(f"Model: {model_path}")
    print(f"Data:  {data_yaml}")
    print("=" * 60)

    logger = DetMetricsLogger(
        experiment_name=f"eval_{model_path.stem}",
        config={
            "model": str(model_path),
            "data": data_yaml,
            "conf_thres": args.conf,
            "iou_thres": args.iou,
            "imgsz": args.imgsz,
        },
    )

    try:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
    except ImportError:
        print("ERROR: ultralytics not installed. Install with: pip install ultralytics")
        sys.exit(1)

    print("\nRunning validation...")
    val_results = model.val(
        data=data_yaml,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save_json=args.save_json,
        plots=args.plot,
    )

    metrics = val_results.results_dict if hasattr(val_results, "results_dict") else {}
    logger.log_metrics(metrics, step=0)

    print("\n--- Thesis-Level Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if hasattr(val_results, "ap_class50") and hasattr(val_results, "class_names"):
        logger.log_class_metrics({
            "ap_class50": val_results.ap_class50.tolist(),
            "class_names": val_results.class_names,
        })
    elif hasattr(val_results, "class_metrics"):
        logger.log_class_metrics(val_results.class_metrics)

    confidences = []
    entropy_values = []

    print("\nRunning test inference for confidence/entropy analysis...")
    try:
        from ultralytics.utils import ASSETS
        test_images = list(Path(YOLO_DIR / "images" / "test").glob("*"))
        if not test_images:
            test_images = list(Path(YOLO_DIR / "images" / "val").glob("*"))

        for img_path in test_images:
            results = model(str(img_path), imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    confs = r.boxes.conf.cpu().numpy()
                    confidences.extend(confs.tolist())
                    ent = compute_entropy(confs)
                    entropy_values.append(ent)

        if confidences:
            logger.log_entropy_distribution(entropy_values, step=0)

            print(f"\nConfidence Distribution:")
            print(f"  Mean:     {np.mean(confidences):.4f}")
            print(f"  Std:      {np.std(confidences):.4f}")
            print(f"  Min:      {np.min(confidences):.4f}")
            print(f"  Max:      {np.max(confidences):.4f}")
            print(f"  Samples:  {len(confidences)}")
            print(f"\nEntropy Distribution:")
            print(f"  Mean:     {np.mean(entropy_values):.4f}" if entropy_values else "  N/A")

    except Exception as e:
        print(f"  Skipping entropy analysis: {e}")

    logger.flush()
    print(f"\nEvaluation complete. Results saved to: {logger.get_run_path()}")


if __name__ == "__main__":
    main()
