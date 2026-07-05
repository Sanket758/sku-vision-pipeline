"""
Experiment 3: SKU-precision clustering for Cien.

DINOv2 + UMAP + HDBSCAN baseline (over-segmented, high precision).
For each cluster: contact sheet, purity stats, intra-cluster distance.
For each centroid: nearest cross-cluster neighbors (suggest manual merges).
For each crop: top-5 nearest neighbors across dataset (embedding sanity).

Usage:
    /home/sanket758/Education/BSBI/Masters-Thesis/.venv/bin/python \
        experiments/shared/explore_subclusters.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoImageProcessor, AutoModel

CROP_DIR = Path("/home/sanket758/Education/BSBI/Masters-Thesis/Dataset/processed_retrieval")
CLUSTERS_FILE = Path("/home/sanket758/Education/BSBI/Masters-Thesis/experiments/annotation_tool/data/clusters.json")
OCR_FILE = Path("/home/sanket758/Education/BSBI/Masters-Thesis/experiments/annotation_tool/data/ocr_results.jsonl")
OUT_DIR = Path("/home/sanket758/Education/BSBI/Masters-Thesis/experiments/shared/subcluster_experiment")

DINOV2_MODEL = "facebook/dinov2-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8


def log(m):
    print(m, flush=True)


def load_cien():
    data = json.loads(CLUSTERS_FILE.read_text())
    for c in data["clusters"]:
        if c["key"] == "cien":
            return c["member_fnames"]
    return []


def load_ocr():
    ocr = {}
    with open(OCR_FILE) as f:
        for line in f:
            d = json.loads(line)
            ocr[d["fname"]] = " ".join(t["text"] for t in d.get("texts", []))
    return ocr


def get_embeds(fnames):
    cache = OUT_DIR / "cien_dinov2.npy"
    if cache.exists():
        return np.load(cache)
    log("Computing DINOv2 embeddings...")
    processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL)
    model = AutoModel.from_pretrained(DINOV2_MODEL).to(DEVICE)
    model.eval()
    all_embeds = []
    for i in range(0, len(fnames), BATCH_SIZE):
        batch = fnames[i:i + BATCH_SIZE]
        images = []
        for fname in batch:
            path = CROP_DIR / fname
            if path.exists():
                try:
                    images.append(Image.open(path).convert("RGB"))
                except Exception:
                    pass
        if not images:
            continue
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        all_embeds.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
        if (i + BATCH_SIZE) % 32 == 0:
            log(f"  {min(i+BATCH_SIZE, len(fnames))}/{len(fnames)}")
    embeds = np.concatenate(all_embeds)
    np.save(cache, embeds)
    return embeds


def cluster_hdbscan(X):
    import hdbscan
    cl = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric="euclidean")
    labels = cl.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = sum(1 for l in labels if l == -1)
    probs = cl.probabilities_
    return labels, {"n_clusters": n_clusters, "n_noise": n_noise, "probs": probs}


def nearest_neighbors(embeds, crop_names, k=5):
    sim = cosine_similarity(embeds)
    nn = {}
    for i in range(len(crop_names)):
        scores = sim[i]
        indices = np.argsort(scores)[::-1][1:k+1]
        nn[crop_names[i]] = [(crop_names[j], float(scores[j])) for j in indices]
    return nn


def centroid(embeds, indices):
    return np.mean(embeds[indices], axis=0)


def inter_cluster_nn(embeds, fnames, labels, k=3):
    clusters = {}
    for i, lbl in enumerate(labels):
        clusters.setdefault(int(lbl), []).append(i)
    cross_nn = {}
    for lbl, indices in clusters.items():
        if lbl == -1:
            continue
        c = centroid(embeds, indices).reshape(1, -1)
        other_indices = [i for i in range(len(fnames)) if labels[i] != lbl]
        if not other_indices:
            continue
        other_embeds = embeds[other_indices]
        sim = cosine_similarity(c, other_embeds)[0]
        top_k = np.argsort(sim)[::-1][:k]
        cross_nn[lbl] = [(fnames[other_indices[j]], float(sim[j])) for j in top_k]
    return cross_nn


def make_html(fnames, embeds, labels, info, ocr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    nn = nearest_neighbors(embeds, fnames, k=5)
    reducer = umap.UMAP(n_components=2, random_state=43, n_neighbors=15)
    embeds_2d = reducer.fit_transform(embeds)

    clusters = {}
    for i, lbl in enumerate(labels):
        clusters.setdefault(int(lbl), []).append(i)
    cross_nn = inter_cluster_nn(embeds, fnames, labels, k=3)

    css = """
    <style>
        body{font-family:sans-serif;margin:20px;font-size:13px;color:#222;}
        .block{margin:24px 0;border:1px solid #ddd;border-radius:8px;overflow:hidden;}
        .hdr{padding:8px 14px;font-weight:bold;font-size:15px;background:#f0f0f0;}
        .grid{display:flex;flex-wrap:wrap;gap:3px;padding:6px;}
        .cell{text-align:center;font-size:10px;width:90px;}
        .cell img{width:84px;height:84px;object-fit:cover;border-radius:3px;}
        .cell .lbl{max-width:84px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:9px;}
        .nn-row{display:flex;gap:4px;padding:4px 14px 8px;overflow-x:auto;}
        .nn-item{text-align:center;font-size:9px;width:64px;flex-shrink:0;}
        .nn-item img{width:60px;height:60px;object-fit:cover;border-radius:3px;}
        .nn-item .score{color:#666;font-size:8px;}
        .stats{display:flex;gap:16px;margin:10px 0;flex-wrap:wrap;}
        .stat-card{background:#f9f9f9;border:1px solid #ddd;border-radius:6px;padding:6px 12px;}
        .stat-card .val{font-size:20px;font-weight:bold;}
        .stat-card .lbl{font-size:11px;color:#666;}
        .cross-nn{display:flex;gap:4px;padding:4px 14px 12px;overflow-x:auto;}
        .cross-nn-item{text-align:center;font-size:9px;width:64px;flex-shrink:0;}
        .cross-nn-item img{width:60px;height:60px;object-fit:cover;border-radius:3px;border:1px solid #ccc;}
        .cross-nn-item .score{color:#666;font-size:8px;}
        table{border-collapse:collapse;margin:10px 0;width:100%;}
        th,td{padding:4px 8px;border-bottom:1px solid #eee;text-align:left;font-size:12px;}
        th{background:#f0f0f0;}
        .noise .hdr{background:#eee;color:#999;}
        .summary{margin:20px 0;}
        .color-dot{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:4px;}
    </style>
    """

    colors = plt.cm.tab10(np.linspace(0, 1, info["n_clusters"]))

    body = f"<h2>Cien SKU-precision clusters — DINOv2 + UMAP + HDBSCAN</h2>"
    body += f"<p>{len(fnames)} crops · {info['n_clusters']} clusters · {info['n_noise']} noise · over-segmented for SKU purity</p>"

    # Stats summary
    body += '<div class="stats">'
    body += f'<div class="stat-card"><div class="val">{info["n_clusters"]}</div><div class="lbl">clusters</div></div>'
    body += f'<div class="stat-card"><div class="val">{info["n_noise"]}</div><div class="lbl">noise (unassigned)</div></div>'
    avg_size = (len(fnames) - info["n_noise"]) / max(info["n_clusters"], 1)
    body += f'<div class="stat-card"><div class="val">{avg_size:.1f}</div><div class="lbl">avg cluster size</div></div>'
    cluster_sizes = [len(v) for v in clusters.values() if len(clusters) > 0]
    if cluster_sizes:
        body += f'<div class="stat-card"><div class="val">{min(cluster_sizes)}-{max(cluster_sizes)}</div><div class="lbl">size range</div></div>'
    body += "</div>"

    # UMAP
    plt.figure(figsize=(8, 6))
    for lbl in sorted(set(labels)):
        mask = labels == lbl
        c = "gray" if lbl == -1 else colors[lbl % info["n_clusters"]]
        plt.scatter(embeds_2d[mask, 0], embeds_2d[mask, 1],
                    c=[c], label=f"Cl {lbl}" if lbl != -1 else "Noise",
                    alpha=0.7, s=30)
    plt.legend(fontsize=7, ncol=4)
    plt.title("Cien — DINOv2 + UMAP (color = cluster)")
    plot_path = OUT_DIR / "cien_umap.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    body += f'<img src="cien_umap.png" style="max-width:700px;margin:10px 0">'

    # Cluster detail
    body += '<h3>Per-cluster detail</h3>'
    for lbl in sorted(set(labels)):
        indices = clusters[int(lbl)]
        members = [fnames[i] for i in indices]
        is_noise = lbl == -1
        label_str = "Noise" if is_noise else f"Cluster {lbl}"
        color_style = f"background:{colors[lbl % info['n_clusters']]};color:#fff" if not is_noise else ""

        body += f'<div class="block {"noise" if is_noise else ""}">'
        body += f'<div class="hdr" style="{color_style}">{label_str} — {len(members)} crops</div>'

        # Purity table
        texts = [ocr.get(f, "") for f in members]
        wf = Counter(w.lower() for t in texts for w in t.strip().split())
        top_words = wf.most_common(5)

        # Intra-cluster distances
        cluster_embeds = embeds[indices]
        centroid_v = np.mean(cluster_embeds, axis=0)
        dists = np.linalg.norm(cluster_embeds - centroid_v, axis=1)
        mean_dist = float(np.mean(dists))
        std_dist = float(np.std(dists))

        body += f'<table><tr><th>Metric</th><th>Value</th></tr>'
        body += f'<tr><td>Intra-cluster mean dist</td><td>{mean_dist:.4f} ± {std_dist:.4f}</td></tr>'
        body += f'<tr><td>Top OCR words</td><td>{", ".join(f"{w}({c})" for w,c in top_words)}</td></tr>'
        body += f'<tr><td>Files</td><td style="font-size:10px">{" ".join(m[:20] for m in members[:4])}</td></tr>'
        body += "</table>"

        # Contact sheet
        body += '<div class="grid">'
        for fname in members:
            txt = ocr.get(fname, "")[:25]
            body += f'<div class="cell"><img src="{CROP_DIR / fname}" loading="lazy"><div class="lbl">{txt}</div></div>'
        body += "</div>"

        # Nearest neighbors within cluster (centroid)
        if not is_noise and len(members) > 1:
            closest_idx = indices[int(np.argmin(dists))]
            farthest_idx = indices[int(np.argmax(dists))]
            body += f'<div style="padding:0 14px 4px;font-size:11px;color:#666">Centroid nearest: {fnames[closest_idx][:30]} | Centroid farthest: {fnames[farthest_idx][:30]}</div>'

        # Cross-cluster nearest neighbors (suggest merges)
        if not is_noise and lbl in cross_nn:
            body += f'<div style="padding:4px 14px 0;font-size:11px;color:#666;font-weight:bold">Nearest from other clusters (potential merges):</div>'
            body += '<div class="cross-nn">'
            for nn_fname, nn_score in cross_nn[lbl]:
                nn_lbl = labels[fnames.index(nn_fname)]
                body += f'<div class="cross-nn-item"><img src="{CROP_DIR / nn_fname}" loading="lazy">'
                body += f'<div>Cl {nn_lbl}</div><div class="score">{nn_score:.3f}</div></div>'
            body += "</div>"

        body += "</div>"

    # Global nearest neighbors per crop (sampled)
    body += '<h3>Nearest neighbor examples (5 per cluster, shows embedding quality)</h3>'
    for lbl in sorted(set(labels)):
        if lbl == -1:
            continue
        indices = clusters[int(lbl)]
        if not indices:
            continue
        body += f'<div class="block"><div class="hdr">Cluster {lbl} — 5 example crops with their top-5 NN</div>'
        for idx in indices[:5]:
            fname = fnames[idx]
            body += f'<div style="padding:4px 14px;font-size:11px;border-top:1px solid #eee">'
            body += f'<b>{fname[:35]}</b> top-5 NN:'
            body += '<div class="nn-row">'
            for nn_fname, nn_score in nn[fname]:
                body += f'<div class="nn-item"><img src="{CROP_DIR / nn_fname}" loading="lazy">'
                body += f'<div class="score">{nn_score:.3f}</div></div>'
            body += "</div></div>"
        body += "</div>"

    (OUT_DIR / "cien_purity.html").write_text(
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>{css}</head><body>{body}</body></html>"
    )
    log(f"Saved: {OUT_DIR}/cien_purity.html")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fnames = load_cien()
    ocr = load_ocr()
    log(f"Cien crops: {len(fnames)}")

    X = get_embeds(fnames)
    log(f"Embeddings: {X.shape}")

    log("Clustering (DINOv2 + UMAP + HDBSCAN, over-segmented)...")
    labels, info = cluster_hdbscan(X)

    log(f"Result: {info['n_clusters']} clusters, {info['n_noise']} noise")

    clusters = {}
    for i, lbl in enumerate(labels):
        clusters.setdefault(int(lbl), []).append(fnames[i])
    for lbl in sorted(clusters.keys()):
        log(f"  {'Noise' if lbl==-1 else f'Cl {lbl}'}: {len(clusters[lbl])}")

    make_html(fnames, X, labels, info, ocr)
    log(f"\nSaved: {OUT_DIR}/cien_purity.html")
    log("Done.")


if __name__ == "__main__":
    main()
