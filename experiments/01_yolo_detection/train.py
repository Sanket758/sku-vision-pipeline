"""Cloud-ready YOLOv8/YOLOv10 training script.
Usage:
    python train.py --model yolov8m.pt --config configs/yolov8_german_supermarket.yaml
    python train.py --model yolov10m.pt --config configs/yolov10_german_supermarket.yaml --env kaggle
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import hardware_utils as hw
from shared.metrics_logger import DetMetricsLogger


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Training for German Supermarket SKU Detection")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="Model name or path")
    parser.add_argument("--config", type=str, default="configs/yolov8_german_supermarket.yaml",
                        help="Path to hyperparameter config")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data.yaml (default: Dataset/processed_yolo/data.yaml)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--env", type=str, default=None,
                        choices=["kaggle", "colab", "local"], help="Override environment detection")
    parser.add_argument("--project", type=str, default="runs", help="Output project directory")
    parser.add_argument("--name", type=str, default=None, help="Run name")
    parser.add_argument("--device", type=str, default="", help="Device to use")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    env = args.env or hw.detect_environment()
    hw_config = hw.auto_config()

    print("=" * 60)
    print(hw.hw_summary())
    print("=" * 60)

    project_root = Path(__file__).resolve().parents[2]
    data_yaml = args.data or str(project_root / "Dataset" / "processed_yolo" / "data.yaml")

    if not Path(data_yaml).exists():
        print(f"ERROR: data.yaml not found at {data_yaml}")
        print("Run shared/dataset_utils.py to prepare the dataset first.")
        sys.exit(1)

    logger = DetMetricsLogger(
        experiment_name=f"yolo_{Path(args.model).stem}",
        config={"env": env, "model": args.model, "imgsz": args.imgsz, **hw_config},
    )

    import yaml
    with open(args.config) as f:
        hyp = yaml.safe_load(f)

    if args.epochs:
        hyp["epochs"] = args.epochs

    logger.log_hyperparams({"model": args.model, "data": data_yaml, "hyp": hyp, **hw_config})

    kwargs = dict(
        data=data_yaml,
        epochs=hyp.get("epochs", 100),
        imgsz=args.imgsz,
        device=args.device or hw_config["device"],
        project=args.project,
        name=args.name or logger.run_id,
        exist_ok=True,
        resume=args.resume,
        amp=hw_config["amp"],
        patience=hyp.get("patience", 15),
        lr0=hyp.get("lr0", 0.001),
        optimizer=hyp.get("optimizer", "AdamW"),
        weight_decay=hyp.get("weight_decay", 0.0005),
        warmup_epochs=hyp.get("warmup_epochs", 3),
        box=hyp.get("box", 7.5),
        cls=hyp.get("cls", 0.5),
        dfl=hyp.get("dfl", 1.5),
        label_smoothing=hyp.get("label_smoothing", 0.05),
        hsv_h=hyp.get("hsv_h", 0.015),
        hsv_s=hyp.get("hsv_s", 0.7),
        hsv_v=hyp.get("hsv_v", 0.4),
        degrees=hyp.get("degrees", 10.0),
        translate=hyp.get("translate", 0.1),
        scale=hyp.get("scale", 0.5),
        shear=hyp.get("shear", 2.0),
        flipud=hyp.get("flipud", 0.0),
        fliplr=hyp.get("fliplr", 0.5),
        mosaic=hyp.get("mosaic", 1.0),
        mixup=hyp.get("mixup", 0.1),
        copy_paste=hyp.get("copy_paste", 0.1),
        erasing=hyp.get("erasing", 0.4),
        dropout=hyp.get("dropout", 0.1),
        verbose=True,
    )

    if args.batch:
        kwargs["batch"] = args.batch
    elif hw_config["env"] == "local" and hw_config.get("vram_gb", 0) < 7:
        kwargs["batch"] = -1

    print(f"\nStarting training: {args.model} on {env}")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {hyp.get('epochs', 100)}")
    print(f"  Device: {kwargs['device']}")
    print(f"  AMP: {kwargs['amp']}")
    print(f"  Batch: {kwargs.get('batch', 'auto')}")
    print(f"  Output: {args.project}/{args.name or logger.run_id}")
    print()

    try:
        from ultralytics import YOLO

        model = YOLO(args.model)
        results = model.train(**{k: v for k, v in kwargs.items() if k != "amp"})

        final_metrics = results.results_dict if hasattr(results, "results_dict") else {}
        if final_metrics:
            logger.log_metrics(final_metrics, step=hyp.get("epochs", 100))

        logger.flush()
        print(f"\nTraining complete. Run logs: {logger.get_run_path()}")

        if env in ("kaggle", "colab"):
            output_dir = Path(args.project) / (args.name or logger.run_id)
            best_pt = output_dir / "weights" / "best.pt"
            if best_pt.exists():
                import shutil
                shutil.copy2(best_pt, logger.get_run_path() / "best_model.pt")
                logger.save_artifact(str(best_pt), "best_model.pt")
                print(f"Best model saved to {logger.get_run_path() / 'best_model.pt'}")

    except ImportError:
        print("ERROR: ultralytics not installed. Install with: pip install ultralytics")
        sys.exit(1)


if __name__ == "__main__":
    main()
