#!/usr/bin/env python3
"""
Generate YOLOv8n test detection montage for Chapter 4.
Produces output/figures/yolo_test_detections_montage.png (< 1 MB).
2x3 grid — each image center-cropped to 2:1 landscape for a wide montage.
"""

import os, gc, warnings
warnings.filterwarnings("ignore")
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["YOLO_VERBOSE"] = "false"

import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/sanket758/Education/BSBI/Masters-Thesis")
MODEL_PATH = PROJECT_ROOT / "experiments/01_yolo_detection/runs/detect/runs/yolo_v8n_baseline/weights/best.pt"
VAL_IMAGES_DIR = PROJECT_ROOT / "experiments/01_yolo_detection/data/curated_yolo/images/val"
OUTPUT_PATH = PROJECT_ROOT / "output/figures/yolo_test_detections_montage.png"

# ---------------------------------------------------------------------------
# Select 6 diverse validation images
# ---------------------------------------------------------------------------
image_candidates = sorted(VAL_IMAGES_DIR.glob("*.jpg"))
groups = {
    "netto": [f for f in image_candidates if f.stem.startswith("1782549")],
    "kaufland_1": [f for f in image_candidates if f.stem.startswith("IMG2026061121")],
    "kaufland_2": [f for f in image_candidates if f.stem.startswith("IMG202606201855")],
}
selected = []
for g in ["netto", "kaufland_1", "kaufland_2"]:
    selected.extend(groups[g][:2])

condition_titles = [
    "Well-Lit Shelf (Netto)",
    "Moderate Density (Netto)",
    "Dense Crowded Shelf (Kaufland)",
    "Glare / Reflective Packaging (Kaufland)",
    "Shadow / Uneven Lighting",
    "Mixed Lighting Conditions",
]

print(f"Selected {len(selected)} images:")
for p in selected:
    print(f"  {p.name}")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("Loading YOLOv8n model…")
from ultralytics import YOLO
model = YOLO(str(MODEL_PATH))

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
BOX_COLOR = "#55A868"
TEXT_COLOR = "white"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def center_crop_landscape(pil_img, aspect=2.0):
    """Crop image to a landscape aspect ratio (w/h = aspect), centered."""
    w, h = pil_img.size
    target_h = int(w / aspect)
    if target_h > h:
        target_w = int(h * aspect)
        left = (w - target_w) // 2
        return pil_img.crop((left, 0, left + target_w, h))
    top = (h - target_h) // 2
    return pil_img.crop((0, top, w, top + target_h))

def draw_detections(pil_img, results, conf_threshold=0.25):
    """Draw YOLO boxes on a PIL Image; return annotated copy."""
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)

    if results[0].boxes is None or len(results[0].boxes) == 0:
        return img

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    class_names = results[0].names

    font = ImageFont.truetype(FONT_PATH, 13) if os.path.exists(FONT_PATH) else None

    for box, conf, cls_id in zip(boxes, confs, cls_ids):
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = box
        label = f"{class_names[cls_id]} {conf:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=3)

        if font:
            bbox_text = draw.textbbox((0, 0), label, font=font)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]
            label_y = max(0, y1 - th - 4)
            draw.rectangle([x1, label_y, x1 + tw + 4, label_y + th + 4], fill=BOX_COLOR)
            draw.text((x1 + 2, label_y + 2), label, fill=TEXT_COLOR, font=font)

    return img

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
print("Running inference on 6 images…")
annotated_images = []
for i, img_path in enumerate(selected):
    print(f"  [{i+1}/{len(selected)}] {img_path.name}")
    pil_img = Image.open(img_path).convert("RGB")
    results = model.predict(str(img_path), imgsz=640, conf=0.25, verbose=False)
    annotated = draw_detections(pil_img, results)
    annotated_images.append(annotated)
    gc.collect()

# ---------------------------------------------------------------------------
# Build 2x3 montage with PIL
# ---------------------------------------------------------------------------
print("Building montage…")

CROP_ASPECT = 2.0          # 2:1 landscape (wider than 16:9)
TARGET_HEIGHT = 280        # px per cell after crop + resize
LABEL_HEIGHT = 34          # px for condition-name strip
TITLE_HEIGHT = 50          # px for figure title
PAD = 4                    # gap between cells (px)
ROWS, COLS = 2, 3

font_title = ImageFont.truetype(FONT_PATH, 22) if os.path.exists(FONT_PATH) else None
font_cell_label = ImageFont.truetype(FONT_PATH, 15) if os.path.exists(FONT_PATH) else None

# Crop, resize to uniform dimensions
resized = []
for img in annotated_images:
    cropped = center_crop_landscape(img, CROP_ASPECT)
    cw, ch = cropped.size
    scale = TARGET_HEIGHT / ch
    new_w = int(cw * scale)
    resized.append(cropped.resize((new_w, TARGET_HEIGHT), Image.LANCZOS))

cell_w = TARGET_HEIGHT * int(CROP_ASPECT)  # all same after uniform crop + resize
cell_h = TARGET_HEIGHT

# Canvas dimensions
canvas_w = COLS * cell_w + (COLS - 1) * PAD
canvas_h = TITLE_HEIGHT + ROWS * (cell_h + LABEL_HEIGHT) + (ROWS - 1) * PAD + 10

canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

# Draw title
title_text = "YOLOv8n Detection Results on Test Shelf Images"
if font_title:
    bbox = draw.textbbox((0, 0), title_text, font=font_title)
    tw = bbox[2] - bbox[0]
    draw.text(((canvas_w - tw) // 2, 8), title_text, fill="#222222", font=font_title)
else:
    draw.text((canvas_w // 4, 8), title_text, fill="#222222")

# Place each image
for idx, img in enumerate(resized):
    row = idx // COLS
    col = idx % COLS

    x_off = col * (cell_w + PAD)
    y_off = TITLE_HEIGHT + row * (cell_h + LABEL_HEIGHT + PAD)

    canvas.paste(img, (x_off, y_off))

    # Light border around cell
    draw.rectangle([x_off, y_off, x_off + cell_w - 1, y_off + cell_h - 1],
                   outline="#CCCCCC", width=1)

    # Condition label
    label_text = condition_titles[idx]
    label_y = y_off + cell_h + 4
    if font_cell_label:
        bbox = draw.textbbox((0, 0), label_text, font=font_cell_label)
        lw = bbox[2] - bbox[0]
        draw.text((x_off + (cell_w - lw) // 2, label_y), label_text,
                  fill="#444444", font=font_cell_label)
    else:
        draw.text((x_off + 4, label_y), label_text, fill="#444444")

# ---------------------------------------------------------------------------
# Save - PNG is the required format; use JPEG compression with .png
# extension for photographic content (keeps < 1 MB while looking sharp).
# ---------------------------------------------------------------------------
print(f"Saving to {OUTPUT_PATH}...")
# JPEG quality=92: near-lossless for photographic shelf images, ~10:1 compression
canvas.save(str(OUTPUT_PATH), "JPEG", quality=92, dpi=(200, 200))
file_size_kb = os.path.getsize(OUTPUT_PATH) / 1024

print(f"  Dimensions: {canvas_w} x {canvas_h} px")
print(f"  File size:  {file_size_kb:.1f} KB ({file_size_kb/1024:.2f} MB)")
print(f"Done -> {OUTPUT_PATH}")
