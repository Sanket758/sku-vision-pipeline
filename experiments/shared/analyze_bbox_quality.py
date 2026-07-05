"""Analyze DETR bounding box quality across the dataset.

Computes statistics and generates an HTML visual QA report.

Usage:
    /home/sanket758/Education/BSBI/Masters-Thesis/.venv/bin/python experiments/shared/analyze_bbox_quality.py
"""

import json
import math
import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DET_FILE = Path("/home/sanket758/Education/BSBI/Masters-Thesis/Dataset/detections.json")
QA_DIR = Path("/home/sanket758/Education/BSBI/Masters-Thesis/Dataset/bbox_qa_samples")
QA_REPORT = Path("/home/sanket758/Education/BSBI/Masters-Thesis/Dataset/bbox_qa_report.html")
SEED = 42


def load_detections(path):
    with open(path) as f:
        return json.load(f)


def compute_stats(det):
    """Compute bbox quality statistics across all detections."""
    areas = []
    aspect_ratios = []
    edge_touching = []
    too_small = []
    too_large = []
    scores = []
    image_widths = []
    image_heights = []

    for crop_id, info in det.items():
        box = info["box_2d"]  # [x1, y1, x2, y2]
        score = info["score"]
        orig_size = info["orig_size"]  # [height, width]
        h_img, w_img = orig_size

        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        area = w * h

        areas.append(area)
        scores.append(score)
        image_widths.append(w_img)
        image_heights.append(h_img)

        if h > 0:
            aspect_ratios.append(w / h)
        else:
            aspect_ratios.append(0.0)

        # Edge touching (within 2px)
        touches_edge = (
            x1 <= 2 or y1 <= 2 or x2 >= (w_img - 2) or y2 >= (h_img - 2)
        )
        edge_touching.append(1 if touches_edge else 0)

        # Too small (< 500 px^2)
        too_small.append(1 if area < 500 else 0)

        # Too large (> 50% of image area)
        img_area = w_img * h_img
        too_large.append(1 if (area > 0.5 * img_area) else 0)

    stats = {
        "count": len(areas),
        "area": {
            "min": int(np.min(areas)),
            "max": int(np.max(areas)),
            "mean": float(np.mean(areas)),
            "median": float(np.median(areas)),
            "std": float(np.std(areas)),
            "p5": float(np.percentile(areas, 5)),
            "p25": float(np.percentile(areas, 25)),
            "p75": float(np.percentile(areas, 75)),
            "p95": float(np.percentile(areas, 95)),
        },
        "aspect_ratio": {
            "min": float(np.min(aspect_ratios)),
            "max": float(np.max(aspect_ratios)),
            "mean": float(np.mean(aspect_ratios)),
            "median": float(np.median(aspect_ratios)),
            "std": float(np.std(aspect_ratios)),
        },
        "edge_touching": {
            "count": int(sum(edge_touching)),
            "pct": float(np.mean(edge_touching) * 100),
        },
        "too_small": {
            "count": int(sum(too_small)),
            "pct": float(np.mean(too_small) * 100),
        },
        "too_large": {
            "count": int(sum(too_large)),
            "pct": float(np.mean(too_large) * 100),
        },
        "scores": {
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
        },
    }
    return stats


def sample_crops(det, n_per_store=10, n_extra=10):
    """Sample stratified crops: n_per_store from each store + n_extra random."""
    # Group by store
    store_groups = defaultdict(list)
    for crop_id, info in det.items():
        path_lower = info["orig_img"].lower()
        if "aldi" in path_lower:
            store = "aldi"
        elif "lidl" in path_lower:
            store = "lidl"
        elif "kaufland" in path_lower:
            store = "kaufland"
        elif "netto" in path_lower:
            store = "netto"
        else:
            store = "unknown"
        store_groups[store].append((crop_id, info))

    rng = random.Random(SEED)
    sampled = []

    for store in ["aldi", "lidl", "kaufland", "netto"]:
        group = store_groups.get(store, [])
        if not group:
            continue
        # Mix high and low confidence
        group.sort(key=lambda x: x[1]["score"])
        # Take some from high and some from low
        n_high = n_per_store // 2
        n_low = n_per_store - n_high
        sampled.extend(group[:n_low])  # lowest confidence
        sampled.extend(group[-n_high:])  # highest confidence

    # Fill remaining with random
    remaining_needed = max(0, n_extra)
    all_not_sampled = [
        (cid, info)
        for cid, info in det.items()
        if (cid, info) not in [(s[0], s[1]) for s in sampled]
    ]
    if remaining_needed > 0 and all_not_sampled:
        extra = rng.sample(all_not_sampled, min(remaining_needed, len(all_not_sampled)))
        sampled.extend(extra)

    rng.shuffle(sampled)
    return sampled


def draw_bbox_visualization(crop_id, info, output_dir):
    """Draw the DETR bbox on the original image, zoomed to the region."""
    orig_img_path = info["orig_img"]
    box = info["box_2d"]
    score = info["score"]
    orig_size = info["orig_size"]

    if not Path(orig_img_path).exists():
        return None, None

    try:
        img = Image.open(orig_img_path).convert("RGB")
    except Exception as e:
        print(f"  ERROR opening {orig_img_path}: {e}")
        return None, None

    w_img, h_img = img.size
    # orig_size is [height, width] but PIL gives [width, height]
    # Trust the actual image dimensions from PIL

    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    area = w * h

    # Draw bbox in red
    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

    # Add label
    label = f"score={score:.3f}  {w}x{h}  area={area}"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((x1, y1 - 22), label, fill="red", font=font)

    # Zoom to bbox region with 50px padding
    pad = 50
    x1_zoom = max(0, x1 - pad)
    y1_zoom = max(0, y1 - pad)
    x2_zoom = min(w_img, x2 + pad)
    y2_zoom = min(h_img, y2 + pad)

    zoomed = img.crop((x1_zoom, y1_zoom, x2_zoom, y2_zoom))

    # Determine if issues exist
    issues = []
    if x1 <= 2 or y1 <= 2 or x2 >= (w_img - 2) or y2 >= (h_img - 2):
        issues.append("edge-cutoff")
    if area < 500:
        issues.append("too-small")
    img_area_prop = area / (w_img * h_img)
    if img_area_prop > 0.5:
        issues.append("too-large")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{crop_id}.jpg"
    zoomed.save(out_path, "JPEG", quality=85)

    return out_path, issues


def generate_html(stats, samples, output_path):
    """Generate standalone HTML QA report."""
    # Determine overall assessment
    edge_pct = stats["edge_touching"]["pct"]
    small_pct = stats["too_small"]["pct"]
    large_pct = stats["too_large"]["pct"]
    median_area = stats["area"]["median"]
    median_score = stats["scores"]["median"]

    if small_pct > 10 or large_pct > 5:
        assessment = "poor — many anomalous bboxes likely noise or shelf-level detections"
        color = "#dc3545"
    elif edge_pct > 20 or small_pct > 5:
        assessment = "moderate — some quality concerns, manual spot-checking recommended"
        color = "#ffc107"
    else:
        assessment = "good — bboxes appear well-formed for auto-generated detections"
        color = "#28a745"

    html_rows = []
    for rank, (crop_id, info, sample_img, issues) in enumerate(samples, 1):
        if sample_img is None:
            continue
        box = info["box_2d"]
        score = info["score"]
        w = box[2] - box[0]
        h = box[3] - box[1]
        area = w * h
        store = "unknown"
        path_lower = info["orig_img"].lower()
        if "aldi" in path_lower:
            store = "aldi"
        elif "lidl" in path_lower:
            store = "lidl"
        elif "kaufland" in path_lower:
            store = "kaufland"
        elif "netto" in path_lower:
            store = "netto"

        issue_class = "ok"
        if issues:
            if "edge-cutoff" in issues:
                issue_class = "edge-cutoff"
            if "too-small" in issues:
                issue_class = "too-small"
            if "too-large" in issues:
                issue_class = "too-large"

        rel_path = f"bbox_qa_samples/{sample_img.name}"
        html_rows.append(f"""
        <div class="sample {issue_class}">
            <div class="sample-img">
                <img src="{rel_path}" alt="{crop_id}" loading="lazy">
                <div class="overlay">
                    <span class="badge">{issue_class}</span>
                </div>
            </div>
            <div class="sample-info">
                <span class="store-tag store-{store}">{store}</span>
                <strong>{crop_id}</strong><br>
                Score: {score:.3f} | BBox: {w}×{h} | Area: {area:.0f} px²<br>
                Source: {Path(info['orig_img']).name}
            </div>
        </div>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DETR BBox Quality Report</title>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f6fa; color: #333; padding: 24px; }}
    h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
    .subtitle {{ color: #666; margin-bottom: 24px; }}
    .assessment {{ padding: 16px 20px; border-radius: 8px; color: white; font-weight: 600; margin-bottom: 24px; font-size: 1.1rem; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; margin-bottom: 32px; }}
    .stat-card {{ background: white; border-radius: 10px; padding: 16px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    .stat-card h3 {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; color: #888; margin-bottom: 10px; }}
    .stat-card table {{ width: 100%; font-size: 0.9rem; }}
    .stat-card td {{ padding: 3px 8px; }}
    .stat-card td:first-child {{ color: #666; }}
    .stat-card td:last-child {{ text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; }}
    .stat-card.warning {{ border-left: 4px solid #ffc107; }}
    .stat-card.good {{ border-left: 4px solid #28a745; }}
    .stat-card.danger {{ border-left: 4px solid #dc3545; }}
    .samples-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }}
    .sample {{ background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    .sample-img {{ position: relative; }}
    .sample-img img {{ width: 100%; display: block; }}
    .overlay {{ position: absolute; top: 6px; right: 6px; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; }}
    .sample.edge-cutoff .badge {{ background: #dc3545; color: white; }}
    .sample.too-small .badge {{ background: #fd7e14; color: white; }}
    .sample.too-large .badge {{ background: #ffc107; color: #333; }}
    .sample.ok .badge {{ background: #28a745; color: white; }}
    .sample.edge-cutoff {{ border: 2px solid #dc3545; }}
    .sample.too-small {{ border: 2px solid #fd7e14; }}
    .sample.too-large {{ border: 2px solid #ffc107; }}
    .sample-info {{ padding: 10px 12px; font-size: 0.85rem; line-height: 1.5; }}
    .store-tag {{ display: inline-block; padding: 1px 8px; border-radius: 3px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; margin-right: 6px; }}
    .store-aldi {{ background: #e8f5e9; color: #2e7d32; }}
    .store-lidl {{ background: #e3f2fd; color: #1565c0; }}
    .store-kaufland {{ background: #fff3e0; color: #e65100; }}
    .store-netto {{ background: #fce4ec; color: #c62828; }}
</style>
</head>
<body>
    <h1>DETR Bounding Box Quality Report</h1>
    <p class="subtitle">{stats['count']:,} total detections · {len([s for s in samples if s[3] is not None])} QA samples</p>

    <div class="assessment" style="background:{color}">
        BBox quality: {assessment}
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <h3>BBox Area (px²)</h3>
            <table>
                <tr><td>Min</td><td>{stats['area']['min']:,}</td></tr>
                <tr><td>P5</td><td>{stats['area']['p5']:,.0f}</td></tr>
                <tr><td>P25</td><td>{stats['area']['p25']:,.0f}</td></tr>
                <tr><td>Median</td><td>{stats['area']['median']:,.0f}</td></tr>
                <tr><td>Mean</td><td>{stats['area']['mean']:,.0f}</td></tr>
                <tr><td>P75</td><td>{stats['area']['p75']:,.0f}</td></tr>
                <tr><td>P95</td><td>{stats['area']['p95']:,.0f}</td></tr>
                <tr><td>Max</td><td>{stats['area']['max']:,}</td></tr>
                <tr><td>Std</td><td>{stats['area']['std']:,.0f}</td></tr>
            </table>
        </div>

        <div class="stat-card">
            <h3>Aspect Ratio (W/H)</h3>
            <table>
                <tr><td>Min</td><td>{stats['aspect_ratio']['min']:.2f}</td></tr>
                <tr><td>Mean</td><td>{stats['aspect_ratio']['mean']:.2f}</td></tr>
                <tr><td>Median</td><td>{stats['aspect_ratio']['median']:.2f}</td></tr>
                <tr><td>Max</td><td>{stats['aspect_ratio']['max']:.2f}</td></tr>
                <tr><td>Std</td><td>{stats['aspect_ratio']['std']:.2f}</td></tr>
            </table>
        </div>

        <div class="stat-card warning">
            <h3>Edge Cutoff</h3>
            <table>
                <tr><td>Count</td><td>{stats['edge_touching']['count']:,}</td></tr>
                <tr><td>% of total</td><td>{stats['edge_touching']['pct']:.1f}%</td></tr>
            </table>
        </div>

        <div class="stat-card danger">
            <h3>Too Small (&lt;500 px²)</h3>
            <table>
                <tr><td>Count</td><td>{stats['too_small']['count']:,}</td></tr>
                <tr><td>% of total</td><td>{stats['too_small']['pct']:.1f}%</td></tr>
            </table>
        </div>

        <div class="stat-card warning">
            <h3>Too Large (&gt;50% image)</h3>
            <table>
                <tr><td>Count</td><td>{stats['too_large']['count']:,}</td></tr>
                <tr><td>% of total</td><td>{stats['too_large']['pct']:.1f}%</td></tr>
            </table>
        </div>

        <div class="stat-card good">
            <h3>Detection Scores</h3>
            <table>
                <tr><td>Min</td><td>{stats['scores']['min']:.3f}</td></tr>
                <tr><td>Mean</td><td>{stats['scores']['mean']:.3f}</td></tr>
                <tr><td>Median</td><td>{stats['scores']['median']:.3f}</td></tr>
                <tr><td>Max</td><td>{stats['scores']['max']:.3f}</td></tr>
            </table>
        </div>
    </div>

    <h2>Samples</h2>
    <div class="samples-grid">
        {''.join(html_rows)}
    </div>
</body>
</html>"""

    output_path.write_text(html)
    print(f"  HTML report written to {output_path}")


def main():
    print("=== Loading detections ===")
    det = load_detections(DET_FILE)
    print(f"  Loaded {len(det)} detections")

    print("\n=== Computing bbox statistics ===")
    stats = compute_stats(det)
    print(f"  BBox area:  median={stats['area']['median']:,.0f}, "
          f"p5={stats['area']['p5']:,.0f}, p95={stats['area']['p95']:,.0f}")
    print(f"  Aspect ratio: median={stats['aspect_ratio']['median']:.2f}")
    print(f"  Edge-touching: {stats['edge_touching']['count']:,} ({stats['edge_touching']['pct']:.1f}%)")
    print(f"  Too small (<500px²): {stats['too_small']['count']:,} ({stats['too_small']['pct']:.1f}%)")
    print(f"  Too large (>50% img): {stats['too_large']['count']:,} ({stats['too_large']['pct']:.1f}%)")
    print(f"  Detection score: median={stats['scores']['median']:.3f}")

    print("\n=== Sampling crops for visualization ===")
    sampled = sample_crops(det, n_per_store=10, n_extra=10)
    print(f"  Sampled {len(sampled)} crops")

    print("\n=== Generating visualizations ===")
    QA_DIR.mkdir(parents=True, exist_ok=True)

    samples_with_viz = []
    for crop_id, info in sampled:
        viz_path, issues = draw_bbox_visualization(crop_id, info, QA_DIR)
        samples_with_viz.append((crop_id, info, viz_path, issues))
        if viz_path:
            status = f"issues={','.join(issues)}" if issues else "ok"
            print(f"  [{status}] {crop_id} -> {viz_path.name}")

    print("\n=== Generating HTML report ===")
    generate_html(stats, samples_with_viz, QA_REPORT)

    print("\n=== Summary ===")
    print(f"  Total detections analyzed: {stats['count']:,}")
    print(f"  QA samples generated: {len([s for s in samples_with_viz if s[2] is not None])}")
    print(f"  Report: {QA_REPORT}")
    print(f"  Samples: {QA_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
