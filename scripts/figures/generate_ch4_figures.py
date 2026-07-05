#!/usr/bin/env python3
"""
Generate 3 publication-quality thesis figures (Chapter 4) from experiment data.

Figures:
  fig_4_1_training_curves.png — YOLOv8n vs YOLOv10n training curves
  fig_4_2_per_class_f1.png   — Per-SKU top-1 accuracy bar chart
  fig_4_3_embedding_comparison.png — DINOv3 vs MNet vs Hybrid embedding ablation
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output/figures")
EXP_DIR    = Path("experiments")
DPI        = 1200
FALLBACK_DPI = 600
MAX_SIZE_MB = 1.0

COLORS = {
    "blue":   "#4C72B0",
    "orange": "#DD8452",
    "green":  "#55A868",
    "red":    "#C44E52",
    "purple": "#8172B3",
    "brown":  "#937860",
    "yellow": "#E5B642",
}

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi":     150,   # screen preview; save uses DPI arg
    "savefig.dpi":    DPI,
    "savefig.bbox":   "tight",
    "axes.grid":      True,
    "axes.axisbelow": True,
})

# ── Helpers ─────────────────────────────────────────────────────────────────

def smooth_series(y, window=5):
    """Simple moving average with reflect padding."""
    if window < 2:
        return y
    pad = window // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(y_pad, kernel, mode="valid")[:len(y)]

def save_fig(fig, filename):
    """Save figure with size constraint; fallback DPI if oversize."""
    path = OUTPUT_DIR / filename
    save_kw = dict(dpi=DPI, bbox_inches="tight", facecolor="white",
                   pil_kwargs={"compress_level": 9})
    fig.savefig(path, **save_kw)
    size_kb = os.path.getsize(path) / 1024
    if size_kb > MAX_SIZE_MB * 1024:
        save_kw["dpi"] = FALLBACK_DPI
        fig.savefig(path, **save_kw)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {filename}: {size_kb:.0f} KB (fallback {FALLBACK_DPI} DPI)")
    else:
        print(f"  {filename}: {size_kb:.0f} KB ({DPI} DPI)")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4-1: Training Curves
# ═══════════════════════════════════════════════════════════════════════════

def fig_4_1_training_curves():
    print("\n[fig_4_1] Training curves …")

    csv_v8n  = EXP_DIR / "01_yolo_detection/runs/detect/runs/yolo_v8n_baseline/results.csv"
    csv_v10n = EXP_DIR / "01_yolo_detection/runs/detect/runs/yolo_v10n_baseline/results.csv"

    df8  = pd.read_csv(csv_v8n)
    df10 = pd.read_csv(csv_v10n)

    window = 5

    # Total loss = box_loss + cls_loss
    train_loss8  = smooth_series(df8["train/box_loss"].values  + df8["train/cls_loss"].values,  window)
    val_loss8    = smooth_series(df8["val/box_loss"].values    + df8["val/cls_loss"].values,    window)
    train_loss10 = smooth_series(df10["train/box_loss"].values + df10["train/cls_loss"].values, window)
    val_loss10   = smooth_series(df10["val/box_loss"].values   + df10["val/cls_loss"].values,   window)

    epochs = df8["epoch"].values  # 1-based

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # ── Left: Loss ──────────────────────────────────────────────────────
    ax1.plot(epochs, train_loss8,  color=COLORS["blue"],   lw=1.2, label="YOLOv8n Train")
    ax1.plot(epochs, val_loss8,   color=COLORS["blue"],   lw=1.2, ls="--", label="YOLOv8n Val")
    ax1.plot(epochs, train_loss10, color=COLORS["orange"], lw=1.2, label="YOLOv10n Train")
    ax1.plot(epochs, val_loss10,  color=COLORS["orange"], lw=1.2, ls="--", label="YOLOv10n Val")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Loss (box + cls)")
    ax1.set_title("Training & Validation Loss")
    ax1.legend(frameon=True, fancybox=False, fontsize=7)
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.set_xlim(1, 100)

    # ── Right: mAP ──────────────────────────────────────────────────────
    ax2.plot(epochs, df8["metrics/mAP50(B)"],     color=COLORS["blue"],   lw=1.2, label="YOLOv8n mAP₅₀")
    ax2.plot(epochs, df8["metrics/mAP50-95(B)"],  color=COLORS["blue"],   lw=1.2, ls="--", label="YOLOv8n mAP₅₀₋₉₅")
    ax2.plot(epochs, df10["metrics/mAP50(B)"],    color=COLORS["orange"], lw=1.2, label="YOLOv10n mAP₅₀")
    ax2.plot(epochs, df10["metrics/mAP50-95(B)"], color=COLORS["orange"], lw=1.2, ls="--", label="YOLOv10n mAP₅₀₋₉₅")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mAP")
    ax2.set_title("Detection Accuracy")
    ax2.legend(frameon=True, fancybox=False, fontsize=7)
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_xlim(1, 100)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.tight_layout(pad=2.0)
    save_fig(fig, "fig_4_1_training_curves.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4-2: Per-SKU F1 (Top-1 Accuracy)
# ═══════════════════════════════════════════════════════════════════════════

def fig_4_2_per_class_f1():
    print("\n[fig_4_2] Per-SKU accuracy …")

    eval_path = EXP_DIR / "curated_pipeline/data/eval_results/eval_results.json"
    with open(eval_path) as f:
        data = json.load(f)

    per_class = data["experiments"]["persku_retrieval"]["per_class"]
    overall   = data["experiments"]["persku_retrieval"]["overall"]
    mean_acc  = overall["top1_mean"]  # 0.9369

    # Build sorted list of (sku_code, top1_acc)
    items = [(k, v["top1_acc"]) for k, v in per_class.items()]
    items_sorted = sorted(items, key=lambda x: x[1])  # ascending

    # Get bottom 30 and top 30
    n_show = 30
    bottom = items_sorted[:n_show]
    top    = items_sorted[-n_show:]

    # Combine: bottom then a gap then top
    # We'll use a visual separator between the two groups
    labels_bottom = [b[0] for b in bottom]
    vals_bottom   = [b[1] * 100 for b in bottom]
    labels_top    = [t[0] for t in top]
    vals_top      = [t[1] * 100 for t in top]

    # Combined list with a blank separator row
    gap_label = ""
    all_labels = labels_bottom + [gap_label] + labels_top
    all_vals   = vals_bottom   + [0]         + vals_top

    y_pos = np.arange(len(all_labels))

    fig, ax = plt.subplots(figsize=(7, 8))

    # Color each bar
    bar_colors = []
    for v in all_vals:
        if v == 0:
            bar_colors.append("white")
        elif v >= 90:
            bar_colors.append(COLORS["green"])
        elif v >= 70:
            bar_colors.append(COLORS["yellow"])
        else:
            bar_colors.append(COLORS["red"])

    bars = ax.barh(y_pos, all_vals, color=bar_colors, edgecolor="gray",
                   linewidth=0.3, height=0.7)

    # Mean accuracy line
    ax.axvline(mean_acc * 100, color="black", lw=1.2, ls="--", label=f"Mean: {mean_acc*100:.1f}%")

    # Label 5 worst and 5 best
    worst5_indices = list(range(5))                         # first 5 of bottom
    best5_indices  = [len(bottom) + 1 + i for i in range(n_show - 5, n_show)]  # last 5 of top

    annotation_ys = []
    for idx in worst5_indices:
        ax.annotate(f"{all_labels[idx]}={all_vals[idx]:.1f}%",
                    xy=(all_vals[idx], y_pos[idx]),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=6, fontweight="bold", va="center",
                    color=COLORS["red"])
        annotation_ys.append(y_pos[idx])

    for idx in best5_indices:
        ax.annotate(f"{all_labels[idx]}={all_vals[idx]:.1f}%",
                    xy=(all_vals[idx], y_pos[idx]),
                    xytext=(-8, 0), textcoords="offset points",
                    fontsize=6, fontweight="bold", va="center", ha="right",
                    color=COLORS["green"])
        annotation_ys.append(y_pos[idx])

    # Style
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_labels, fontsize=6)
    ax.set_xlabel("Top-1 Accuracy (%)")
    ax.set_title("Per-SKU Retrieval Accuracy")
    ax.legend(frameon=True, fancybox=False, fontsize=7, loc="lower right")
    ax.set_xlim(-5, 105)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Horizontal separator line between groups
    sep_y = n_show  # the gap index
    ax.axhline(y=sep_y - 0.5, color="gray", lw=0.8, ls="-")

    fig.tight_layout()
    save_fig(fig, "fig_4_2_per_class_f1.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4-3: Embedding Comparison (DINOv3 vs MNet vs Hybrid)
# ═══════════════════════════════════════════════════════════════════════════

def fig_4_3_embedding_comparison():
    print("\n[fig_4_3] Embedding comparison …")

    eval_path = EXP_DIR / "curated_pipeline/data/eval_results/eval_results.json"
    with open(eval_path) as f:
        data = json.load(f)

    results = data["experiments"]["hybrid_ablation"]["results"]

    # Parse results into structured format
    k_values = [1, 3, 5]
    methods  = ["dino", "mnet", "hybrid"]
    labels   = ["DINOv2", "MobileNetV2", "Hybrid (0.55/0.45)"]

    accuracy_map = {k: {m: None for m in methods} for k in k_values}
    for key, v in results.items():
        k = v["k_shot"]
        m = v["method"]
        accuracy_map[k][m] = v["accuracy"] * 100

    fig, ax = plt.subplots(figsize=(5.5, 4))

    width = 0.22
    x = np.arange(len(k_values))

    colors_method = [COLORS["blue"], COLORS["orange"], COLORS["green"]]

    for i, (method, label) in enumerate(zip(methods, labels)):
        vals = [accuracy_map[k][method] for k in k_values]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=colors_method[i], edgecolor="gray", linewidth=0.4)

        # Value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=7,
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["K = 1", "K = 3", "K = 5"], fontsize=9)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Few-Shot Embedding Ablation")
    ax.legend(frameon=True, fancybox=False, fontsize=7, loc="lower right")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 100)

    fig.tight_layout()
    save_fig(fig, "fig_4_3_embedding_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig_4_1_training_curves()
    fig_4_2_per_class_f1()
    fig_4_3_embedding_comparison()

    print("\nAll figures generated in output/figures/")
