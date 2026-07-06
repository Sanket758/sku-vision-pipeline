#!/usr/bin/env python3
"""
Run YOLOv8n baseline inference on curated test images and generate detection data
for the demo webapp.

Usage: python scripts/run_inference.py
Output: Updates ../src/detections.js with real model predictions
"""

import json
import sys
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Paths
MODEL_PATH = Path("/home/sanket758/Education/BSBI/Masters-Thesis/experiments/01_yolo_detection/runs/detect/runs/yolo_v8n_baseline/weights/best.pt")
TEST_IMAGES_DIR = Path("/home/sanket758/Education/BSBI/Masters-Thesis/experiments/01_yolo_detection/data/curated_yolo/images/test")
DEMO_SHELVES_DIR = Path(__file__).resolve().parent.parent / "public" / "demo" / "shelves"
DETECTIONS_JS_PATH = Path(__file__).resolve().parent.parent / "src" / "detections.js"
MASTER_REFS_DIR = Path("/home/sanket758/Education/BSBI/Masters-Thesis/experiments/curated_pipeline/master_references")
EXEMPLARS_DIR = Path("/home/sanket758/Education/BSBI/Masters-Thesis/experiments/curated_pipeline/exemplars")

# SKU-to-product name mapping from class_catalogue.json
CATALOGUE_PATH = Path("/home/sanket758/Education/BSBI/Masters-Thesis/experiments/annotation_tool/data/class_catalogue.json")

def load_catalogue(catalogue_path):
    """Load SKU-to-product name mapping from class catalogue."""
    with open(catalogue_path) as f:
        data = json.load(f)
    
    # Map SKU codes (SKU-0001 format) to product names
    # The curated YOLO uses SKU-001 format (no leading zero)
    sku_names = {}
    for cls in data["classes"]:
        sku_code = cls["sku_code"]  # e.g., "SKU-0001"
        # Convert to demo format: "SKU-001" (remove leading zero after hyphen)
        parts = sku_code.split("-")
        num = int(parts[1])  # Remove leading zeros
        demo_sku = f"SKU-{num:03d}"  # Pad to 3 digits
        sku_names[demo_sku] = cls["class_name"]
    
    return sku_names

def extract_shelf_label(filename):
    """Generate a human-readable label from the filename."""
    if "11210028" in filename:
        return "Kaufland Shelf (Coffee Creamer)"
    if "11210046" in filename:
        return "Kaufland Shelf (Condensed Milk)"
    if "20185525" in filename:
        return "Aldi Shelf (Coffee & Tea)"
    if "20185548" in filename:
        return "Lidl Shelf (Coffee & Tea)"
    if "20185555" in filename:
        return "Aldi Shelf (Hot Beverages)"
    if "20185559" in filename:
        return "Lidl Shelf (Chocolate Spread)"
    if "20185605" in filename:
        return "Lidl Shelf (Packaged Treats)"
    if "1782549176660" in filename:
        return "Kaufland Aisle (Beverages & Spreads)"
    if "1782549177232" in filename:
        return "Kaufland Shelf (Chocolate Bars)"
    if "1782549183882" in filename:
        return "Aldi Shelf (Canned Goods)"
    if "1782549184954" in filename:
        return "Netto Shelf (Packet Mixes)"
    return "Shelf Image"

def extract_source(label):
    """Extract store source from label."""
    if "Kaufland" in label:
        return "Kaufland"
    if "Lidl" in label:
        return "Lidl"
    if "Aldi" in label:
        return "Aldi"
    if "Netto" in label:
        return "Netto"
    return "Retail Store"

def main():
    print("=" * 60)
    print("YOLOv8n Baseline Inference for Demo")
    print("=" * 60)
    
    # Import torch/ultralytics here so the script can be checked for syntax without dependencies
    import torch
    from ultralytics import YOLO
    
    # Load catalogue for product names
    print("\n[1/5] Loading product catalogue...")
    sku_names = load_catalogue(CATALOGUE_PATH)
    print(f"  Loaded {len(sku_names)} SKU-to-name mappings")
    
    # Load model
    print(f"\n[2/5] Loading YOLOv8n baseline from: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print(f"  ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)
    
    model = YOLO(str(MODEL_PATH))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Model loaded on {device}")
    
    # Get all test images
    print(f"\n[3/5] Finding test images in: {TEST_IMAGES_DIR}")
    test_images = sorted(TEST_IMAGES_DIR.glob("*.jpg")) + sorted(TEST_IMAGES_DIR.glob("*.png"))
    print(f"  Found {len(test_images)} test images")
    
    # Copy images to demo shelves directory and run inference
    print(f"\n[4/5] Running inference on {len(test_images)} images...")
    
    os.makedirs(DEMO_SHELVES_DIR, exist_ok=True)
    
    demo_images = []
    
    for img_path in test_images:
        filename = img_path.name
        dest_path = DEMO_SHELVES_DIR / filename
        
        # Copy image to demo dir (only if needed)
        if not dest_path.exists():
            import shutil
            shutil.copy2(img_path, dest_path)
            print(f"  Copied {filename}")
        else:
            print(f"  Already exists: {filename}")
        
        # Run inference
        results = model(str(img_path), conf=0.25, iou=0.5, device=device, verbose=False)
        result = results[0]
        
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xywhn.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.int().cpu().numpy()
            
            for i in range(len(boxes)):
                x, y, w, h = boxes[i]
                conf = float(confs[i])
                cls_id = int(cls_ids[i])
                
                # YOLO class IDs are 0-indexed; the curated YOLO uses SKU-001 -> class 0
                # But looking at the data.yaml, SKU-001 is class 0, SKU-002 is class 1, etc.
                # The curated pipeline uses SKU codes that may not be contiguous 1:1
                # Let's map from the names in data.yaml
                
                # The data.yaml has names as SKU-001 through SKU-148
                # So class ID 0 = SKU-001, class ID 147 = SKU-148
                sku_num = cls_id + 1
                
                # YOLO xywhn gives CENTER (x,y), but CSS needs TOP-LEFT corner
                # Convert: left = center_x - w/2, top = center_y - h/2
                left_pct = round((float(x) - float(w) / 2) * 100, 1)
                top_pct = round((float(y) - float(h) / 2) * 100, 1)
                width_pct = round(float(w) * 100, 1)
                height_pct = round(float(h) * 100, 1)
                bbox = {
                    "x": left_pct,
                    "y": top_pct,
                    "w": width_pct,
                    "h": height_pct
                }
                
                detection = {
                    "sku": f"SKU-{sku_num:03d}",
                    "confidence": round(conf, 3),
                    "bbox": bbox
                }
                detections.append(detection)
            
            print(f"  {filename}: {len(detections)} detections")
        else:
            print(f"  {filename}: No detections")
        
        # Create demo image entry
        label = extract_shelf_label(filename)
        source = extract_source(label)
        
        demo_images.append({
            "filename": f"demo/shelves/{filename}",
            "label": label,
            "source": source,
            "detections": detections
        })
    
    # Get model metrics from FINDINGS_YOLO.md and thesis pipeline reference data
    metrics = {
        "detection_map50": 0.347,
        "detection_map5095": 0.266,
        "detection_recall": 0.411,
        "detection_precision": 0.281,
        "recognition_top1": 0.791,
        "recognition_top3": 0.903,
        "retrieval_top1_dino": 0.514,
        "retrieval_top1_hybrid_k3": 0.791,
        "pipeline_latency_per_crop_ms": 38,
        "vlm_latency_per_image_s": 35,
        "total_skus": 110,
        "total_exemplars": 2790,
        "model": "YOLOv8n Baseline",
        "model_params": "3.34M",
        "test_images": len(test_images),
        "inference_device": device
    }
    
    experiments = [
        {
            "id": "YOLOv8n-DET",
            "name": "YOLOv8n — Product Detection",
            "precision": round(0.281, 3),
            "recall": round(0.411, 3),
            "f1": round(0.334, 3),
            "status": "completed"
        },
        {
            "id": "YOLOv10n-DET",
            "name": "YOLOv10n — Product Detection",
            "precision": 0.373,
            "recall": 0.322,
            "f1": 0.346,
            "status": "completed"
        },
        {
            "id": "DINO-K3-RET",
            "name": "DINOv3 Retrieval (3-shot)",
            "accuracy": 0.605,
            "status": "completed"
        },
        {
            "id": "Hybrid-K3-RET",
            "name": "DINOv3+MobileNetV2 Hybrid (3-shot)",
            "accuracy": 0.791,
            "status": "completed"
        },
        {
            "id": "CLIP-ZS-RET",
            "name": "CLIP Zero-Shot Retrieval",
            "top1_acc": 0.114,
            "top5_acc": 0.285,
            "status": "completed"
        },
        {
            "id": "VLM-LLAVA",
            "name": "LLaVA 7B VLM Annotation",
            "avg_latency_s": 35,
            "hallucination_rate": "high",
            "status": "failed"
        }
    ]
    
    # Build the full detection data object
    detection_data = {
        "images": demo_images,
        "metrics": metrics,
        "experiments": experiments
    }
    
    # Generate the JS file
    print(f"\n[5/5] Writing detections.js...")
    
    js_content = "const detectionData = " + json.dumps(detection_data, indent=2) + ";\nexport default detectionData;\n"
    
    with open(DETECTIONS_JS_PATH, "w") as f:
        f.write(js_content)
    
    print(f"  Written to: {DETECTIONS_JS_PATH}")
    
    # Summary
    total_detections = sum(len(img["detections"]) for img in demo_images)
    total_images = len(demo_images)
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {total_images} images, {total_detections} total detections")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
