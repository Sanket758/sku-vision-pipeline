#!/usr/bin/env python3
"""
generate_figures.py — Generate 5 publication-quality thesis figures.

Usage:
    python scripts/generate_figures.py

Output: output/figures/fig_*.png (1200 DPI, < 8×6 inches, < 1 MB each)
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Force non-interactive backend before any pyplot import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_JSON = BASE_DIR / "experiments" / "curated_pipeline" / "data" / "eval_results" / "eval_results.json"
OUT_DIR = BASE_DIR / "output" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_DIR = BASE_DIR / "Dataset" / "raw"

# ── Color palette (ColorBrewer-inspired, academic) ─────────────────────────
C_BLUE = "#4C72B0"       # steel blue
C_ORANGE = "#DD8452"     # burnt orange
C_GREEN = "#55A868"      # yellowgreen-ish
C_RED = "#C44E52"        # brick red
C_PURPLE = "#8172B3"     # muted purple
C_CREAM = "#F5E6C8"      # warm background
C_GREY = "#AAAAAA"
C_HEAT_CMAP = "RdYlBu_r"  # blue → white → red (reversed RdYlBu)

# Publication settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 1200,
    "savefig.dpi": 1200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ── Helpers ────────────────────────────────────────────────────────────────

def load_eval():
    with open(EVAL_JSON) as f:
        return json.load(f)


def save_fig(fig, filename, max_inches=(8, 6), max_bytes=1_000_000):
    """Save figure at 1200 DPI within size and file size constraints."""
    path = OUT_DIR / filename
    # Clamp figure size
    w, h = fig.get_size_inches()
    scale = min(max_inches[0] / w, max_inches[1] / h, 1.0)
    if scale < 1.0:
        fig.set_size_inches(w * scale, h * scale)

    # Try saving at full 1200 DPI; fall back to 600 if oversize
    for dpi in (1200, 600, 300):
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05,
                    facecolor="white", edgecolor="none")
        size = os.path.getsize(path)
        if size <= max_bytes:
            print(f"  ✓ {filename}  ({size//1024} KB @ {dpi} DPI, "
                  f"{fig.get_size_inches()[0]:.1f}×{fig.get_size_inches()[1]:.1f} in)")
            return
    print(f"  ⚠ {filename}  ({size//1024} KB — oversize, consider manual resize)")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Per-SKU accuracy distribution histogram
# ══════════════════════════════════════════════════════════════════════════
def fig1_persku_histogram(data):
    print("─" * 50)
    print("Figure 1: Per-SKU Accuracy Distribution")

    per_class = data["experiments"]["persku_retrieval"]["per_class"]
    accs = [v["top1_acc"] for v in per_class.values()]
    mean_val = np.mean(accs)
    std_val = np.std(accs)

    fig, ax = plt.subplots(figsize=(6, 4))
    n, bins, patches = ax.hist(accs, bins=20, color=C_BLUE, edgecolor="white",
                                linewidth=0.5, alpha=0.85, zorder=3)

    # Mean line
    ax.axvline(mean_val, color=C_RED, linestyle="--", linewidth=1.2, zorder=4)
    ax.axvline(mean_val - std_val, color=C_GREY, linestyle=":", linewidth=0.8, zorder=4)
    ax.axvline(mean_val + std_val, color=C_GREY, linestyle=":", linewidth=0.8, zorder=4)

    # Annotation
    ylim = ax.get_ylim()
    ax.text(mean_val + 0.01, ylim[1] * 0.92,
            f"μ = {mean_val:.1%}\nσ = {std_val:.1%}",
            color=C_RED, fontsize=8, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_title("Distribution of Per-SKU Retrieval Accuracy")
    ax.set_xlabel("Top-1 Accuracy")
    ax.set_ylabel("Number of SKUs")
    ax.set_xlim(0.0, 1.05)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Count annotation for low-accuracy tail
    low_count = sum(1 for a in accs if a < 0.6)
    if low_count > 0:
        ax.annotate(f"{low_count} SKUs < 60%",
                    xy=(0.3, 2), fontsize=7, color=C_RED, fontstyle="italic",
                    bbox=dict(facecolor="wheat", alpha=0.6, boxstyle="round"))

    save_fig(fig, "fig_ch4_per_sku_distribution.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Few-shot performance heatmap
# ══════════════════════════════════════════════════════════════════════════
def fig2_fewshot_heatmap(data):
    print("─" * 50)
    print("Figure 2: Few-Shot Performance Heatmap")

    results = data["experiments"]["fewshot_hybrid"]["results"]
    n_ways = [5, 10, 20]
    k_shots = [1, 3, 5]
    grid = np.full((len(n_ways), len(k_shots)), np.nan)
    grid_std = np.full((len(n_ways), len(k_shots)), np.nan)

    for i, n in enumerate(n_ways):
        for j, k in enumerate(k_shots):
            key = f"N{n}_K{k}"
            if key in results:
                grid[i, j] = results[key]["top1_mean"] * 100  # percent
                grid_std[i, j] = results[key]["top1_std"] * 100

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.imshow(grid, cmap=C_HEAT_CMAP, vmin=80, vmax=100, aspect="auto")

    # Annotate cells
    for i in range(len(n_ways)):
        for j in range(len(k_shots)):
            val = grid[i, j]
            sval = grid_std[i, j]
            if not np.isnan(val):
                text = f"{val:.1f}%"
                if not np.isnan(sval):
                    text += f"\n±{sval:.1f}"
                color = "white" if val < 88 else "black"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=8, fontweight="bold", color=color)

    ax.set_xticks(range(len(k_shots)))
    ax.set_yticks(range(len(n_ways)))
    ax.set_xticklabels([f"K={k}" for k in k_shots])
    ax.set_yticklabels([f"N={n}" for n in n_ways])
    ax.set_title("Few-Shot Retrieval Performance")
    ax.set_xlabel("Shots")
    ax.set_ylabel("Ways")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Top-1 Accuracy (%)", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    save_fig(fig, "fig_ch4_fewshot_heatmap.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Embedding ablation comparison (grouped bar)
# ══════════════════════════════════════════════════════════════════════════
def fig3_ablation_comparison(data):
    print("─" * 50)
    print("Figure 3: Embedding Ablation Comparison")

    results = data["experiments"]["hybrid_ablation"]["results"]
    # Organize: K=1,3,5  each with dino, mnet, hybrid
    k_vals = [1, 3, 5]
    methods = ["dino", "mnet", "hybrid"]
    method_labels = ["DINOv3", "MNet", "Hybrid"]
    colors = [C_GREEN, C_ORANGE, C_BLUE]

    acc_matrix = np.zeros((len(k_vals), len(methods)))
    for i, k in enumerate(k_vals):
        for j, m in enumerate(methods):
            key = f"K{k}_{m}"
            acc_matrix[i, j] = results[key]["accuracy"] * 100

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(k_vals))
    bar_width = 0.25

    for j in range(len(methods)):
        bars = ax.bar(x + j * bar_width - bar_width, acc_matrix[:, j],
                      bar_width, label=method_labels[j], color=colors[j],
                      edgecolor="white", linewidth=0.5, alpha=0.9, zorder=3)
        # Annotate values
        for bar, val in zip(bars, acc_matrix[:, j]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"K = {k}" for k in k_vals])
    ax.set_ylim(0, 105)
    ax.set_title("Embedding Component Ablation at Varying Gallery Sizes")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.legend(loc="lower right", framealpha=0.85)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    save_fig(fig, "fig_ch4_ablation_comparison.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Detection comparison (grouped bar — hardcoded from Table 4.1)
# ══════════════════════════════════════════════════════════════════════════
def fig4_detection_comparison():
    print("─" * 50)
    print("Figure 4: Detection Baseline Comparison")

    metrics = ["mAP@50", "mAP@50:95", "Precision", "Recall"]
    yolo8n = [0.805, 0.663, 0.801, 0.701]
    yolo10n = [0.790, 0.710, 0.771, 0.732]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(metrics))
    bar_width = 0.30

    bars1 = ax.bar(x - bar_width / 2, yolo8n, bar_width, label="YOLOv8n",
                   color=C_BLUE, edgecolor="white", linewidth=0.5, alpha=0.9, zorder=3)
    bars2 = ax.bar(x + bar_width / 2, yolo10n, bar_width, label="YOLOv10n",
                   color=C_ORANGE, edgecolor="white", linewidth=0.5, alpha=0.9, zorder=3)

    for bars, vals in [(bars1, yolo8n), (bars2, yolo10n)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.set_title("Detection Baseline Comparison")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", framealpha=0.85)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    save_fig(fig, "fig_ch4_detection_comparison.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Dataset sample montage (3×2 grid)
# ══════════════════════════════════════════════════════════════════════════
def fig5_dataset_montage():
    print("─" * 50)
    print("Figure 5: Dataset Sample Montage")

    # Curated selection based on brightness/contrast analysis
    # Each tuple: (filepath, store_label, condition_label)
    candidates = [
        ("kaufland/IMG20260601170154.jpg", "Kaufland", "Well-Lit Shelf"),
        ("kaufland/IMG20260601170913.jpg", "Kaufland", "Glare / Reflective Packaging"),
        ("netto/IMG20260601100354.jpg",  "Netto",   "Dense Product Overlap"),
        ("kaufland/IMG20260601170302.jpg", "Kaufland", "Shadow / Uneven Lighting"),
        ("netto/IMG20260529095752.jpg",  "Netto",   "Cardboard Pallet Display"),
        ("kaufland/IMG20260601170749.jpg", "Kaufland", "Close-Up / Partial Occlusion"),
    ]

    images = []
    labels = []
    for rel_path, store, condition in candidates:
        full_path = RAW_DIR / rel_path
        if not full_path.exists():
            print(f"  ⚠ Image not found: {full_path}")
            continue
        img = Image.open(full_path).convert("RGB")
        # Resize keeping aspect ratio, target ~480px wide
        target_w = 480
        ratio = target_w / img.width
        target_h = int(img.height * ratio)
        img = img.resize((target_w, target_h), Image.LANCZOS)
        images.append(img)
        labels.append(f"{store} — {condition}")

    if len(images) < 6:
        print("  ✗ Not enough images found for montage, skipping")
        return

    # 3 rows × 2 cols
    n_rows, n_cols = 3, 2
    # Find the target height that best fits all images (use median aspect ratio)
    aspect_ratios = [im.height / im.width for im in images]
    med_aspect = np.median(aspect_ratios)
    cell_w = 480
    cell_h = int(cell_w * med_aspect)
    cell_h = max(300, min(420, cell_h))  # reasonable range

    # Border and padding
    border = 2  # white border between cells
    title_h = 45  # space for top title
    label_h = 24  # space for bottom label per cell

    full_w = n_cols * cell_w + (n_cols + 1) * border
    full_h = title_h + n_rows * (cell_h + label_h) + (n_rows + 1) * border

    canvas = Image.new("RGB", (full_w, full_h), "white")
    draw = ImageDraw.Draw(canvas)

    # Title
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except (OSError, IOError):
        title_font = ImageFont.load_default()
    title_text = "Sample Shelf Images from Collected Dataset"
    # Center the title
    try:
        bbox = draw.textbbox((0, 0), title_text, font=title_font)
        tw = bbox[2] - bbox[0]
    except (AttributeError, TypeError):
        tw = len(title_text) * 14  # rough estimate
    draw.text(((full_w - tw) // 2, 10), title_text, fill="black", font=title_font)

    # Label font
    try:
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except (OSError, IOError):
        label_font = ImageFont.load_default()

    for idx, (img, label) in enumerate(zip(images, labels)):
        col = idx % n_cols
        row = idx // n_cols

        # Resize image to cell dimensions
        img_resized = img.resize((cell_w, cell_h), Image.LANCZOS)

        x = border + col * (cell_w + border)
        y = title_h + border + row * (cell_h + label_h + border)

        canvas.paste(img_resized, (x, y))

        # Label below image
        try:
            bbox = draw.textbbox((0, 0), label, font=label_font)
            lw = bbox[2] - bbox[0]
        except (AttributeError, TypeError):
            lw = len(label) * 10
        draw.text(
            (x + (cell_w - lw) // 2, y + cell_h + 4),
            label, fill="black", font=label_font
        )

    save_path = OUT_DIR / "fig_ch3_dataset_samples.png"
    # Save as PNG with optimize; reduce cell size further if oversize
    save_path_png = save_path.with_suffix(".png")
    canvas.save(save_path_png, "PNG", optimize=True)
    size_kb = os.path.getsize(save_path_png) // 1024
    if size_kb > 1000:
        # Too large — re-render at half resolution
        smaller = canvas.resize((full_w // 2, full_h // 2), Image.LANCZOS)
        smaller.save(save_path_png, "PNG", optimize=True)
        size_kb = os.path.getsize(save_path_png) // 1024
        print(f"  ✓ fig_ch3_dataset_samples.png  ({size_kb} KB, "
              f"{full_w//2}×{full_h//2} px — downsized to meet size limit)")
    else:
        print(f"  ✓ fig_ch3_dataset_samples.png  ({size_kb} KB, "
              f"{full_w}×{full_h} px)")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 50)
    print("Thesis Figure Generator")
    print(f"Eval data: {EVAL_JSON}")
    print(f"Output:    {OUT_DIR}")
    print("=" * 50)

    data = load_eval()

    fig1_persku_histogram(data)
    fig2_fewshot_heatmap(data)
    fig3_ablation_comparison(data)
    fig4_detection_comparison()
    fig5_dataset_montage()

    print("─" * 50)
    print("Done. All figures in output/figures/")
    print("=" * 50)


if __name__ == "__main__":
    main()
