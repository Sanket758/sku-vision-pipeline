#!/usr/bin/env python3
"""
Generate failure_mode_examples.png — a 3×2 montage showing 6 documented
failure modes from the thesis evaluation.

Strategy (dual):
  1. CLIP-based search over exemplar crops for SKU-120 (reflective) and
     SKU-067 (visual-similarity / coffee capsules).
  2. Intelligent region extraction from large curated shelf images for the
     other four modes (occlusion, dense shelf, lighting, novel).

Output: output/figures/failure_mode_examples.png  (< 1 MB, publication quality)
"""

import os
import sys
import gc
import warnings
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── paths ──────────────────────────────────────────────────────────────────
PROJ = Path("/home/sanket758/Education/BSBI/Masters-Thesis")
OUTPUT = PROJ / "output" / "figures" / "failure_mode_examples.png"

KAUFLAND = PROJ / "Dataset" / "raw" / "kaufland"
NETTO = PROJ / "Dataset" / "raw" / "netto"
EXEMPLARS = PROJ / "experiments" / "curated_pipeline" / "exemplars"

# Map failure-mode → (shelf image path, crop ROI (x1,y1,x2,y2) or None for auto,
#                      label, subtitle, fallback ROI)
FALLBACK_CONFIG = {
    "reflective": {
        "image": KAUFLAND / "IMG20260601170913.jpg",
        "roi": None,  # auto-detect shiny region
        "label": "Reflective Packaging",
        "subtitle": "SKU-120 | 11.1% top-1 acc",
        "query": "shiny reflective metallic wrapper glare specular highlights",
    },
    "visual_similarity": {
        "image": None,  # from exemplars
        "roi": None,
        "label": "Visual Similarity",
        "subtitle": "SKU-067 (Coffee Capsules) | 50.0% top-1 acc",
        "query": "two nearly identical coffee capsules different only by small text",
    },
    "occlusion": {
        "image": KAUFLAND / "IMG20260601170749.jpg",
        "roi": None,
        "label": "Partial Occlusion",
        "subtitle": "Products stacked behind each other",
        "query": "products stacked behind each other partially hidden overlapping",
    },
    "dense": {
        "image": NETTO / "IMG20260601100354.jpg",
        "roi": None,
        "label": "Dense Shelf / Extreme Density",
        "subtitle": "Tightly packed products, edge-to-edge",
        "query": "many products tightly packed together crowded shelf",
    },
    "lighting": {
        "image": KAUFLAND / "IMG20260601170302.jpg",
        "roi": None,
        "label": "Lighting Variation (Shadows/Glare)",
        "subtitle": "Uneven shelf illumination",
        "query": "uneven lighting shadow dark region on supermarket shelf",
    },
    "novel": {
        "image": NETTO / "IMG20260529095752.jpg",
        "roi": None,
        "label": "Novel / Unseen Product",
        "subtitle": "Cardboard pallet display, unusual format",
        "query": "cardboard display box new product unfamiliar packaging",
    },
}

# ── helpers ─────────────────────────────────────────────────────────────────

def _ensure_rgb(pil_img):
    return pil_img.convert("RGB")


def _entropy(gray_arr):
    """Compute 2D entropy of a grayscale image array."""
    h, w = gray_arr.shape
    hist = np.bincount(gray_arr.ravel(), minlength=256)
    hist = hist / hist.sum()
    return -np.sum(hist * np.log(hist + 1e-12))


def _edge_density(gray_arr):
    """Simple gradient-magnitude-based edge density."""
    gx = np.abs(np.diff(gray_arr, axis=1)).mean()
    gy = np.abs(np.diff(gray_arr, axis=0)).mean()
    return (gx + gy) / 2


def _find_brightest_region(img, crop_size=(800, 800), num_candidates=20):
    """Find the sub-window with highest mean intensity."""
    arr = np.array(_ensure_rgb(img))
    gray = np.mean(arr, axis=2).astype(np.uint8)
    h, w = gray.shape
    ch, cw = crop_size
    if h <= ch or w <= cw:
        return (0, 0, w, h)
    best_score = -1
    best_roi = (0, 0, cw, ch)
    for _ in range(num_candidates):
        y = np.random.randint(0, h - ch)
        x = np.random.randint(0, w - cw)
        tile = gray[y : y + ch, x : x + cw]
        score = tile.mean()
        if score > best_score:
            best_score = score
            best_roi = (x, y, x + cw, y + ch)
    return best_roi


def _find_darkest_region(img, crop_size=(800, 800), num_candidates=20):
    """Find the sub-window with lowest mean intensity (shadow)."""
    arr = np.array(_ensure_rgb(img))
    gray = np.mean(arr, axis=2).astype(np.uint8)
    h, w = gray.shape
    ch, cw = crop_size
    if h <= ch or w <= cw:
        return (0, 0, w, h)
    best_score = 256
    best_roi = (0, 0, cw, ch)
    for _ in range(num_candidates):
        y = np.random.randint(0, h - ch)
        x = np.random.randint(0, w - cw)
        tile = gray[y : y + ch, x : x + cw]
        score = tile.mean()
        if score < best_score:
            best_score = score
            best_roi = (x, y, x + cw, y + ch)
    return best_roi


def _find_highest_entropy_region(img, crop_size=(800, 800), num_candidates=30):
    """Find sub-window with highest entropy (= many different products/edges)."""
    arr = np.array(_ensure_rgb(img))
    gray = np.mean(arr, axis=2).astype(np.uint8)
    h, w = gray.shape
    ch, cw = crop_size
    if h <= ch or w <= cw:
        return (0, 0, w, h)
    best_score = -1
    best_roi = (0, 0, cw, ch)
    for _ in range(num_candidates):
        y = np.random.randint(0, h - ch)
        x = np.random.randint(0, w - cw)
        tile = gray[y : y + ch, x : x + cw]
        score = _entropy(tile)
        if score > best_score:
            best_score = score
            best_roi = (x, y, x + cw, y + ch)
    return best_roi


def _find_high_contrast_edge_region(img, crop_size=(800, 800), num_candidates=30):
    """Find region with highest edge density (= many object boundaries, overlap)."""
    arr = np.array(_ensure_rgb(img))
    gray = np.mean(arr, axis=2).astype(np.uint8)
    h, w = gray.shape
    ch, cw = crop_size
    if h <= ch or w <= cw:
        return (0, 0, w, h)
    best_score = -1
    best_roi = (0, 0, cw, ch)
    for _ in range(num_candidates):
        y = np.random.randint(0, h - ch)
        x = np.random.randint(0, w - cw)
        tile = gray[y : y + ch, x : x + cw]
        score = _edge_density(tile)
        if score > best_score:
            best_score = score
            best_roi = (x, y, x + cw, y + ch)
    return best_roi


# ── CLIP-based panel helpers ────────────────────────────────────────────────

def _load_clip():
    """Load open-clip model; return (model, preprocess, tokenizer) or None."""
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()
        return model, preprocess, tokenizer
    except Exception as exc:
        print(f"[CLIP] load failed: {exc}")
        return None, None, None


def _clip_best_match(crop_dir, text_query, top_k=1):
    """Score all JPEGs in crop_dir against text_query with CLIP, return best path + score."""
    model, preprocess, tokenizer = _load_clip()
    if model is None:
        return None, 0.0

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    crops = sorted(Path(crop_dir).glob("*.jpg"))
    if not crops:
        return None, 0.0

    texts = tokenizer([text_query]).to(device)
    text_feat = model.encode_text(texts)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    best_path = None
    best_score = -1.0

    with torch.no_grad():
        for cp in crops:
            try:
                img = _ensure_rgb(Image.open(cp))
                inp = preprocess(img).unsqueeze(0).to(device)
                img_feat = model.encode_image(inp)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                sim = (img_feat @ text_feat.T).item()
            except Exception:
                sim = -1.0
            if sim > best_score:
                best_score = sim
                best_path = cp

    del model, text_feat
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_path, best_score


# ── panel extraction ────────────────────────────────────────────────────────

def _make_composite_grid(image_paths, cols=3, target_w=400, target_h=300, bg=(255, 255, 255)):
    """Tile several images into a composite grid that fills the target size."""
    if not image_paths:
        return Image.new("RGB", (target_w, target_h), bg)
    # Load and resize each to fit
    tiles = []
    for p in image_paths[:6]:  # max 6
        try:
            img = _ensure_rgb(Image.open(p))
            # resize to fit a grid cell
            tw, th = target_w // cols, target_h // 2
            img.thumbnail((tw - 8, th - 8), Image.LANCZOS)
            # center-pad to uniform cell size
            canvas = Image.new("RGB", (tw, th), bg)
            x = (tw - img.size[0]) // 2
            y = (th - img.size[1]) // 2
            canvas.paste(img, (x, y))
            tiles.append(canvas)
        except Exception:
            pass
    if not tiles:
        return Image.new("RGB", (target_w, target_h), bg)
    # Arrange in 2 rows
    rows = []
    n = len(tiles)
    for r in range(2):
        row_tiles = tiles[r * cols : (r + 1) * cols]
        if not row_tiles:
            break
        row_img = Image.new("RGB", (sum(t.size[0] for t in row_tiles), row_tiles[0].size[1]), bg)
        x = 0
        for t in row_tiles:
            row_img.paste(t, (x, 0))
            x += t.size[0]
        rows.append(row_img)
    # Stack rows
    composite = Image.new("RGB", (rows[0].size[0], sum(r.size[1] for r in rows)), bg)
    y = 0
    for r in rows:
        composite.paste(r, (0, y))
        y += r.size[1]
    # Resize to target
    return composite.resize((target_w, target_h), Image.LANCZOS)


def _extract_reflective(force_fallback=False):
    """Use CLIP on SKU-120 exemplars to create a composite of best reflective crops."""
    sku120_dir = EXEMPLARS / "SKU-120"
    if not sku120_dir.exists():
        print("[reflective] SKU-120 dir not found")
        return _extract_reflective_fallback()

    if not force_fallback:
        # Use CLIP to rank all SKU-120 crops, take top-N for composite
        model, preprocess, tokenizer = _load_clip()
        if model is not None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            texts = tokenizer([
                "shiny reflective metallic wrapper on a supermarket shelf with glare and specular highlights"
            ]).to(device)
            text_feat = model.encode_text(texts)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            scores = []
            crops = sorted(Path(sku120_dir).glob("*.jpg"))
            with torch.no_grad():
                for cp in crops:
                    try:
                        img = _ensure_rgb(Image.open(cp))
                        inp = preprocess(img).unsqueeze(0).to(device)
                        img_feat = model.encode_image(inp)
                        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                        sim = (img_feat @ text_feat.T).item()
                        scores.append((sim, cp))
                    except Exception:
                        pass
            scores.sort(reverse=True, key=lambda x: x[0])
            top_paths = [s[1] for s in scores[:6]]
            print(f"[reflective] CLIP top scores: {[f'{s[0]:.3f}' for s in scores[:6]]}")
            del model, text_feat
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if top_paths:
                composite = _make_composite_grid(top_paths)
                return composite, f"composite_{len(top_paths)}_best"

    return _extract_reflective_fallback()


def _extract_reflective_fallback():
    """Fallback: use shelf image with glare, crop the brightest region."""
    cfg = FALLBACK_CONFIG["reflective"]
    img = Image.open(cfg["image"])
    roi = _find_brightest_region(img, crop_size=(1000, 800))
    crop = img.crop(roi).resize((400, 320), Image.LANCZOS)
    print(f"[reflective] fallback shelf crop roi={roi}")
    return crop, "shelf_glare_region"


def _extract_visual_similarity(force_fallback=False):
    """Use CLIP on SKU-067 exemplars to create composite showing coffee capsules."""
    sku067_dir = EXEMPLARS / "SKU-067"
    if not sku067_dir.exists():
        print("[vis-sim] SKU-067 dir not found")
        return _extract_visual_similarity_fallback()

    if not force_fallback:
        model, preprocess, tokenizer = _load_clip()
        if model is not None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            texts = tokenizer([
                "two nearly identical coffee capsules that differ only by small text on the packaging"
            ]).to(device)
            text_feat = model.encode_text(texts)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            scores = []
            crops = sorted(Path(sku067_dir).glob("*.jpg"))
            with torch.no_grad():
                for cp in crops:
                    try:
                        img = _ensure_rgb(Image.open(cp))
                        inp = preprocess(img).unsqueeze(0).to(device)
                        img_feat = model.encode_image(inp)
                        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                        sim = (img_feat @ text_feat.T).item()
                        scores.append((sim, cp))
                    except Exception:
                        pass
            scores.sort(reverse=True, key=lambda x: x[0])
            top_paths = [s[1] for s in scores[:6]]
            print(f"[vis-sim] CLIP top scores: {[f'{s[0]:.3f}' for s in scores[:6]]}")
            del model, text_feat
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if top_paths:
                composite = _make_composite_grid(top_paths, cols=3)
                return composite, f"composite_{len(top_paths)}_best"

    return _extract_visual_similarity_fallback()


def _extract_visual_similarity_fallback():
    """Fallback: composite of all SKU-067 crops."""
    sku067_dir = EXEMPLARS / "SKU-067"
    crops = sorted(Path(sku067_dir).glob("*.jpg"))
    if crops:
        composite = _make_composite_grid(crops, cols=3)
        return composite, f"composite_{len(crops)}_crops"
    return Image.new("RGB", (400, 300), (240, 240, 240)), "fallback_blank"


def _extract_occlusion():
    """Use shelf close-up image; crop region with highest edge density (= overlapping products)."""
    cfg = FALLBACK_CONFIG["occlusion"]
    img = Image.open(cfg["image"])
    roi = _find_high_contrast_edge_region(img, crop_size=(900, 800))
    crop = img.crop(roi).resize((400, 320), Image.LANCZOS)
    print(f"[occlusion] roi={roi}")
    return crop, "occlusion_region"


def _extract_dense():
    """Use dense-overlap shelf image; crop highest-entropy region (= many products)."""
    cfg = FALLBACK_CONFIG["dense"]
    img = Image.open(cfg["image"])
    roi = _find_highest_entropy_region(img, crop_size=(900, 700))
    crop = img.crop(roi).resize((400, 320), Image.LANCZOS)
    print(f"[dense] roi={roi}")
    return crop, "dense_region"


def _extract_lighting():
    """Use shadow shelf image; crop darkest region."""
    cfg = FALLBACK_CONFIG["lighting"]
    img = Image.open(cfg["image"])
    # Find a region that has both dark and bright areas (uneven lighting)
    roi = _find_darkest_region(img, crop_size=(900, 700))
    crop = img.crop(roi).resize((400, 320), Image.LANCZOS)
    print(f"[lighting] roi={roi}")
    return crop, "shadow_region"


def _extract_novel():
    """Use cardboard-pallet-display image; crop region showing the unusual display format."""
    cfg = FALLBACK_CONFIG["novel"]
    img = Image.open(cfg["image"])
    # For the novel/unseen failure mode, crop the center-ish region showing the display
    h, w = img.size[1], img.size[0]
    roi = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
    crop = img.crop(roi).resize((400, 320), Image.LANCZOS)
    print(f"[novel] roi={roi}")
    return crop, "novel_display_region"


# ── montage assembly ───────────────────────────────────────────────────────

def _resize_to_height(img, target_h=400):
    """Resize maintaining aspect ratio to target height."""
    w, h = img.size
    ratio = target_h / h
    new_w = int(w * ratio)
    return img.resize((new_w, target_h), Image.LANCZOS)


def make_montage(panels, rows=3, cols=2):
    """
    panels: list of dicts with keys 'img' (PIL), 'title', 'subtitle'
    Layout: 3 rows × 2 cols — 6 panels.
    Returns PIL Image.
    """
    TARGET_H = 280
    PAD = 14
    LABEL_H = 48  # space for two-line label below each panel

    # Resize all panels to uniform height
    resized = []
    for p in panels:
        resized.append(_resize_to_height(p["img"], TARGET_H))

    # Determine cell width (max of all resized widths)
    cell_w = max(r.size[0] for r in resized)
    cell_h = TARGET_H + LABEL_H

    # Canvas
    canvas_w = cols * cell_w + (cols + 1) * PAD
    canvas_h = rows * cell_h + (rows + 1) * PAD + 60  # 60 for title
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (IOError, OSError):
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()

    # Title
    title = "Failure Mode Analysis — Documented Failure Cases"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    tw = bbox[2] - bbox[0]
    draw.text(((canvas_w - tw) // 2, 12), title, fill=(30, 30, 30), font=title_font)

    for idx, (p, r) in enumerate(zip(panels, resized)):
        row = idx // cols
        col = idx % cols

        # Center image in cell
        x_off = PAD + col * cell_w + (cell_w - r.size[0]) // 2
        y_off = 60 + PAD + row * cell_h + (TARGET_H - r.size[1]) // 2

        canvas.paste(r, (x_off, y_off))

        # Draw border
        draw.rectangle(
            [x_off - 2, y_off - 2, x_off + r.size[0] + 2, y_off + r.size[1] + 2],
            outline=(200, 200, 200),
            width=1,
        )

        # Label
        lx = PAD + col * cell_w
        ly = 60 + PAD + row * cell_h + TARGET_H + 4

        # Title in bold
        draw.text((lx + 8, ly), p["title"], fill=(40, 40, 40), font=label_font)
        # Subtitle in smaller font
        draw.text((lx + 8, ly + 20), p["subtitle"], fill=(180, 50, 50), font=subtitle_font)

    return canvas


# ── main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Failure Mode Montage Generator")
    print("=" * 60)

    force_fallback = "--fallback" in sys.argv

    panels = []

    # (a) Reflective Packaging
    print("\n[1/6] Reflective Packaging ...")
    try:
        if force_fallback:
            img, src = _extract_reflective_fallback()
        else:
            img, src = _extract_reflective()
        panels.append({
            "img": img,
            "title": "Reflective Packaging",
            "subtitle": "SKU-120 | 11.1% top-1 acc (source: {})".format(src),
        })
        print(f"       -> selected: {src}")
    except Exception as e:
        print(f"       FAILED: {e}")
        traceback.print_exc()
        # emergency blank
        panels.append({
            "img": Image.new("RGB", (400, 300), (240, 240, 240)),
            "title": "Reflective Packaging",
            "subtitle": "SKU-120 | 11.1% top-1 acc",
        })

    # (b) Visual Similarity
    print("\n[2/6] Visual Similarity (Coffee Capsules) ...")
    try:
        if force_fallback:
            img, src = _extract_visual_similarity_fallback()
        else:
            img, src = _extract_visual_similarity()
        panels.append({
            "img": img,
            "title": "Visual Similarity",
            "subtitle": "SKU-067 (Coffee Capsules) | 50.0% top-1 acc (source: {})".format(src),
        })
        print(f"       -> selected: {src}")
    except Exception as e:
        print(f"       FAILED: {e}")
        traceback.print_exc()
        panels.append({
            "img": Image.new("RGB", (400, 300), (240, 240, 240)),
            "title": "Visual Similarity",
            "subtitle": "SKU-067 (Coffee Capsules) | 50.0% top-1 acc",
        })

    # (c) Partial Occlusion
    print("\n[3/6] Partial Occlusion ...")
    try:
        img, src = _extract_occlusion()
        panels.append({
            "img": img,
            "title": "Partial Occlusion",
            "subtitle": "Stacked / overlapping products (source: {})".format(src),
        })
        print(f"       -> selected: {src}")
    except Exception as e:
        print(f"       FAILED: {e}")
        traceback.print_exc()
        panels.append({
            "img": Image.new("RGB", (400, 300), (240, 240, 240)),
            "title": "Partial Occlusion",
            "subtitle": "Stacked / overlapping products",
        })

    # (d) Dense Shelf
    print("\n[4/6] Dense Shelf / Extreme Density ...")
    try:
        img, src = _extract_dense()
        panels.append({
            "img": img,
            "title": "Dense Shelf / Extreme Density",
            "subtitle": "Tightly packed products (source: {})".format(src),
        })
        print(f"       -> selected: {src}")
    except Exception as e:
        print(f"       FAILED: {e}")
        traceback.print_exc()
        panels.append({
            "img": Image.new("RGB", (400, 300), (240, 240, 240)),
            "title": "Dense Shelf / Extreme Density",
            "subtitle": "Tightly packed products",
        })

    # (e) Lighting Variation
    print("\n[5/6] Lighting Variation (Shadows/Glare) ...")
    try:
        img, src = _extract_lighting()
        panels.append({
            "img": img,
            "title": "Lighting Variation (Shadows/Glare)",
            "subtitle": "Uneven shelf illumination (source: {})".format(src),
        })
        print(f"       -> selected: {src}")
    except Exception as e:
        print(f"       FAILED: {e}")
        traceback.print_exc()
        panels.append({
            "img": Image.new("RGB", (400, 300), (240, 240, 240)),
            "title": "Lighting Variation (Shadows/Glare)",
            "subtitle": "Uneven shelf illumination",
        })

    # (f) Novel Product
    print("\n[6/6] Novel / Unseen Product ...")
    try:
        img, src = _extract_novel()
        panels.append({
            "img": img,
            "title": "Novel / Unseen Product",
            "subtitle": "Unusual display format (source: {})".format(src),
        })
        print(f"       -> selected: {src}")
    except Exception as e:
        print(f"       FAILED: {e}")
        traceback.print_exc()
        panels.append({
            "img": Image.new("RGB", (400, 300), (240, 240, 240)),
            "title": "Novel / Unseen Product",
            "subtitle": "Unusual display format",
        })

    # Build montage (3 rows × 2 cols)
    print("\n" + "=" * 60)
    print("Assembling montage ...")
    montage = make_montage(panels, rows=3, cols=2)

    # Save — use JPEG quality path for photo content, then convert to PNG
    print(f"Saving to {OUTPUT} ...")
    # First try PNG with max compression
    montage.save(OUTPUT, "PNG", optimize=True, compress_level=9)
    size_kb = os.path.getsize(OUTPUT) / 1024
    print(f"Output: {OUTPUT}  ({size_kb:.0f} KB)")

    # If still > 1 MB, save via JPEG at quality 85 to shrink (photographic content compresses better in JPEG)
    if size_kb > 1024:
        print("File > 1 MB, compressing via JPEG intermediate ...")
        tmp_jpeg = OUTPUT.with_suffix(".tmp.jpg")
        montage.save(tmp_jpeg, "JPEG", quality=80, optimize=True)
        # Convert back to PNG
        jpeg_img = Image.open(tmp_jpeg)
        jpeg_img.save(OUTPUT, "PNG", optimize=True, compress_level=9)
        tmp_jpeg.unlink()
        size_kb = os.path.getsize(OUTPUT) / 1024
        print(f"After JPEG compression: {size_kb:.0f} KB")

    # Final fallback: progressive reduction
    if size_kb > 1024:
        print("Still > 1 MB, applying progressive downsample ...")
        scale = 0.9
        while size_kb > 1024 and scale > 0.5:
            new_w = int(montage.width * scale)
            new_h = int(montage.height * scale)
            smaller = montage.resize((new_w, new_h), Image.LANCZOS)
            smaller.save(OUTPUT, "PNG", optimize=True, compress_level=9)
            size_kb = os.path.getsize(OUTPUT) / 1024
            scale -= 0.05
        print(f"After downscale: {size_kb:.0f} KB")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
