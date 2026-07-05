import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import hardware_utils as hw


RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


class MetricsLogger:
    """Minimal thesis-level metrics logger.
    
    Logs to local JSON files under runs/ so results survive Kaggle/Colab sessions.
    MLFlow auto-logging can sit on top when available.
    """

    def __init__(self, experiment_name: str, config: Optional[dict] = None):
        self.experiment_name = experiment_name
        self.config = config or {}
        self.env = hw.detect_environment()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = RUNS_DIR / f"{experiment_name}_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics: list[dict] = []
        self._hyperparams: dict = {}
        self._artifacts: list[Path] = []

        hardware_info = {
            "env": self.env,
            "device": str(hw.get_device()),
            "gpu_name": hw.get_gpu_name(),
            "vram_gb": round(hw.get_vram_gb(), 2),
        }

        self._write_file("hardware.json", hardware_info)
        self._write_file("config.json", config or {})

    def _write_file(self, name: str, data):
        (self.run_dir / name).write_text(json.dumps(data, indent=2, default=str))

    def log_hyperparams(self, params: dict):
        self._hyperparams.update(params)
        self._write_file("hyperparams.json", self._hyperparams)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        entry = {"key": key, "value": value, "step": step, "timestamp": datetime.now().isoformat()}
        self._metrics.append(entry)

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        for k, v in metrics.items():
            self.log_metric(k, v, step)

    def log_confusion_matrix(self, cm: list, class_names: list[str], step: int = 0):
        artifact = {"confusion_matrix": cm, "class_names": class_names, "step": step}
        self._write_file(f"confusion_matrix_step_{step}.json", artifact)

    def log_entropy_distribution(self, entropies: list[float], step: int = 0):
        artifact = {"entropies": entropies, "step": step, "mean": float(sum(entropies) / len(entropies)) if entropies else 0.0}
        self._write_file(f"entropy_dist_step_{step}.json", artifact)

    def log_class_metrics(self, class_metrics: dict, step: int = 0):
        self._write_file(f"class_metrics_step_{step}.json", class_metrics)

    def save_artifact(self, local_path: str, artifact_name: Optional[str] = None):
        src = Path(local_path)
        if not src.exists():
            return
        dst = self.run_dir / (artifact_name or src.name)
        import shutil
        shutil.copy2(src, dst)
        self._artifacts.append(dst)

    def flush(self):
        self._write_file("metrics.json", self._metrics)
        summary = {
            "run_id": self.run_id,
            "experiment": self.experiment_name,
            "env": self.env,
            "config": self.config,
            "hyperparams": self._hyperparams,
            "metric_count": len(self._metrics),
            "artifact_count": len(self._artifacts),
        }
        self._write_file("run_summary.json", summary)

    def get_run_path(self) -> Path:
        return self.run_dir

    def get_metrics_df(self):
        try:
            import pandas as pd
            return pd.DataFrame(self._metrics)
        except ImportError:
            return None


class DetMetricsLogger(MetricsLogger):
    """Extends MetricsLogger with detection-specific logging."""

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  map50: float, map5095: float, precision: float, recall: float, f1: float):
        self.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mAP@50": map50,
            "mAP@50-95": map5095,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }, step=epoch)

    def log_overfitting_check(self, train_losses: list[float], val_losses: list[float]):
        divergence = [(v - t) / t * 100 for t, v in zip(train_losses, val_losses) if t > 0]
        self._write_file("overfitting_analysis.json", {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "divergence_pct": divergence,
            "max_divergence_pct": max(divergence) if divergence else 0.0,
            "verdict": "OVERFITTING" if any(d > 15 for d in divergence) else "No significant overfitting detected",
        })
