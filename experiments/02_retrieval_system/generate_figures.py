"""Generate figures for Chapter 4 from experiment results.

Creates:
- fig_4_6_retrieval_accuracy.png: CLIP vs DINOv2 retrieval Top-1/5/10
- fig_4_7_fewshot_results.png: Few-shot accuracy vs K for different N
- fig_4_8_text_query.png: Text-query retrieval accuracy across templates

Usage:
    python generate_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox_inches": "tight",
    "font.family": "sans-serif",
})

# ── Paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE / "experiments" / "02_retrieval_system" / "results"
FIGURES_DIR = BASE / "output" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def fig_4_6_retrieval_accuracy() -> None:
    """Grouped bar chart: CLIP vs DINOv2 retrieval Top-1/5/10."""
    # Load results
    with open(RESULTS_DIR / "exp1_clip_retrieval.json") as f:
        clip_full = json.load(f)

    with open(RESULTS_DIR / "exp2_dinov2_retrieval.json") as f:
        dinov2 = json.load(f)

    # Extract accuracies
    ks = [1, 5, 10]
    clip_full_acc = [clip_full["overall_accuracy"][f"top{k}"] for k in ks]
    clip_labeled_acc = [dinov2["clip_labeled_only_accuracy"][f"top{k}"] for k in ks]
    dinov2_acc = [dinov2["dinov2_overall_accuracy"][f"top{k}"] for k in ks]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ks))
    width = 0.25

    bars1 = ax.bar(x - width, clip_full_acc, width,
                   label=f"CLIP (full index, n={clip_full['index_vectors']:,})",
                   color="#2196F3", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x, clip_labeled_acc, width,
                   label=f"CLIP (labeled only, n={dinov2['database_size']})",
                   color="#64B5F6", edgecolor="white", linewidth=0.5)
    bars3 = ax.bar(x + width, dinov2_acc, width,
                   label=f"DINOv2 (labeled only, n={dinov2['database_size']})",
                   color="#FF9800", edgecolor="white", linewidth=0.5)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1%}",
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Top-K")
    ax.set_ylabel("Retrieval Accuracy")
    ax.set_title("Image Retrieval Accuracy: CLIP vs DINOv2")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in ks])
    ax.set_ylim(0, 1.12)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = FIGURES_DIR / "fig_4_6_retrieval_accuracy.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def fig_4_7_fewshot_results() -> None:
    """Line plot: few-shot accuracy vs K for different N values."""
    with open(RESULTS_DIR / "exp3_fewshot.json") as f:
        data = json.load(f)

    results = data["results"]
    n_values = data["n_values"]
    k_values = data["k_values"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {5: "#2196F3", 10: "#FF9800", 20: "#4CAF50"}
    markers = {5: "o", 10: "s", 20: "^"}

    # Top-1 accuracy
    for n in n_values:
        top1_means = []
        top1_stds = []
        for k in k_values:
            key = f"N{n}_K{k}"
            top1_means.append(results[key]["top1_mean"])
            top1_stds.append(results[key]["top1_std"])

        ax1.errorbar(k_values, top1_means, yerr=top1_stds,
                    marker=markers[n], color=colors[n],
                    label=f"N={n}", capsize=4, linewidth=2,
                    markersize=8)

    ax1.set_xlabel("K (support images per class)")
    ax1.set_ylabel("Top-1 Accuracy")
    ax1.set_title("Few-Shot Top-1 Accuracy")
    ax1.set_xticks(k_values)
    ax1.set_ylim(0.4, 1.02)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Top-5 accuracy
    for n in n_values:
        top5_means = []
        top5_stds = []
        for k in k_values:
            key = f"N{n}_K{k}"
            top5_means.append(results[key]["top5_mean"])
            top5_stds.append(results[key]["top5_std"])

        ax2.errorbar(k_values, top5_means, yerr=top5_stds,
                    marker=markers[n], color=colors[n],
                    label=f"N={n}", capsize=4, linewidth=2,
                    markersize=8)

    ax2.set_xlabel("K (support images per class)")
    ax2.set_ylabel("Top-5 Accuracy")
    ax2.set_title("Few-Shot Top-5 Accuracy")
    ax2.set_xticks(k_values)
    ax2.set_ylim(0.85, 1.02)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig_4_7_fewshot_results.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def fig_4_8_text_query() -> None:
    """Bar chart: text-query accuracy across prompt templates."""
    with open(RESULTS_DIR / "exp4_text_query.json") as f:
        data = json.load(f)

    templates = list(data["results_by_template"].keys())
    ks = [1, 5, 10]
    best = data["best_template"]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(templates))
    width = 0.25
    colors_k = {1: "#2196F3", 5: "#FF9800", 10: "#4CAF50"}

    for i, k in enumerate(ks):
        accs = [
            data["results_by_template"][t]["overall_accuracy"][f"top{k}"]
            for t in templates
        ]
        bars = ax.bar(x + i * width, accs, width,
                     label=f"Top-{k}", color=colors_k[k],
                     edgecolor="white", linewidth=0.5)

        # Add value labels on Top-1 bars
        if k == 1:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.1%}",
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha="center", va="bottom", fontsize=8)

    # Mark the best template
    best_idx = templates.index(best)
    ax.annotate("★ Best", xy=(best_idx + 0.25, 0.02),
               ha="center", fontsize=10, color="#E91E63", fontweight="bold")

    ax.set_xlabel("Prompt Template")
    ax.set_ylabel("Accuracy")
    ax.set_title("Zero-Shot Text-Query SKU Recognition")
    ax.set_xticks(x + width)

    # Prettier template labels
    template_labels = []
    for t in templates:
        prompt = data["prompt_templates"][t]
        label = prompt.replace("{class_name}", "<SKU>")
        template_labels.append(f'"{label}"')

    ax.set_xticklabels(template_labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig_4_8_text_query.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    """Generate all figures."""
    print("Generating Chapter 4 figures...")
    print()

    fig_4_6_retrieval_accuracy()
    fig_4_7_fewshot_results()
    fig_4_8_text_query()

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
