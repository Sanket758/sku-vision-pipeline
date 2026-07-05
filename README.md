# SKU-Level Product Recognition in Retail Shelf Images: A Multi-Architecture Approach with Curated Pipeline Refinement

**Computer Vision · Retail AI · Fine-Grained Product Recognition · Object Detection**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.x-00C7B7.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-enabled-blueviolet.svg)](https://git-lfs.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Citation

```bibtex
@mastersthesis{gadge2026sku,
  author  = {Sanket Gadge},
  title   = {{SKU-Level Product Recognition in Retail Shelf Images: A Multi-Architecture Approach with Curated Pipeline Refinement}},
  school  = {BSBI / University for the Creative Arts},
  year    = {2026},
  type    = {Master's dissertation}
}
```

---

## Abstract

This repository contains the complete source code and trained model weights for a Master's dissertation investigating SKU-level product recognition in retail shelf images. The study evaluates four architectural paradigms—YOLO object detection baselines, CLIP/DINOv2 retrieval-based matching, Vision-Language Model (VLM) few-shot annotation, and a curated DINOv3+MobileNetV2 hybrid pipeline—on a dataset of 148 SKU classes spanning coffee, chocolate, and confectionery products from German discount supermarkets (Aldi, Kaufland, Lidl, Netto).

The curated pipeline achieves **93.7% top-1 retrieval accuracy** across 148 SKUs, outperforming CLIP ViT-B/32 zero-shot by 36.8 percentage points. At 5-way 1-shot classification—matching a typical shelf row with one reference image per product—accuracy reaches **96.0%**. Object detection baselines (YOLOv8n: mAP50 = 0.347) establish a complementary localisation capability. The code is designed for full reproducibility: all experiments, evaluation scripts, figure generators, and model weights are included.

---

## Repository Structure

```
Thesis/
├── experiments/
│   ├── 01_yolo_detection/           # Chapter 4 — YOLO Detection Baseline
│   │   ├── train.py                 #   YOLOv8/YOLOv10/YOLO11 training
│   │   ├── evaluate.py              #   mAP, confusion matrix, entropy analysis
│   │   ├── prepare_data.py          #   Build YOLO dataset from curated pipeline exports
│   │   ├── verify_export.py         #   Validate YOLO export integrity
│   │   └── configs/                 #   Hyperparameter YAML files
│   │       ├── yolov8_german_supermarket.yaml
│   │       ├── yolov10_german_supermarket.yaml
│   │       └── yolov8_tuned.yaml
│   │
│   ├── 02_retrieval_system/         # Chapter 4 — Retrieval-Based Approaches
│   │   ├── exp1_clip_retrieval.py   #   CLIP ViT-B/32 retrieval accuracy
│   │   ├── exp2_dinov2_retrieval.py #   DINOv2 ViT-B/14 retrieval accuracy
│   │   ├── exp3_fewshot.py          #   Few-shot N-way K-shot evaluation (CLIP)
│   │   ├── exp4_text_query.py       #   VLM text-query evaluation with prompt templates
│   │   ├── indexer.py               #   Build FAISS index from product images
│   │   ├── query.py                 #   FAISS index query + batch evaluation
│   │   ├── match_skus.py            #   Match retrieval crops to SKU reference images
│   │   └── results/                 #   Experiment result outputs
│   │
│   ├── 03_vlm_fewshot/             # Chapter 4 — VLM Few-Shot Detection
│   │   ├── vlm_annotation_pipeline.py # Ollama-based auto-annotation pipeline
│   │   ├── eval_vlm_ollama.py       #   Zero-shot / few-shot VLM evaluation
│   │   ├── vlm_converter.py         #   Convert VLM JSON to YOLO format
│   │   ├── configs/
│   │   │   └── vlm_annotation.yaml  #   Pipeline configuration
│   │   ├── prompts/
│   │   │   ├── zero_shot_detection.txt
│   │   │   └── few_shot_detection.txt
│   │   └── annotations/             #   Generated annotation outputs
│   │
│   ├── curated_pipeline/           # Chapter 4 — Curated Pipeline
│   │   ├── pipeline.py              #   Main pipeline: detect → embed → match → review
│   │   ├── config.py                #   Pipeline configuration constants
│   │   ├── review_server.py         #   Interactive visual review web server
│   │   ├── yolo_to_labelme.py       #   Convert YOLO exports to LabelMe JSON
│   │   ├── export_yolo_from_registry.py  # Export SKU registry to YOLO format
│   │   ├── auto_review_pipeline.py  #   Automated review workflow
│   │   └── pipeline_utils/
│   │       ├── detection.py         #   YOLOv5 object detection on shelf images
│   │       ├── embeddings.py        #   DINOv3 + MobileNetV2 feature extraction
│   │       ├── matching.py          #   SKU registry matching logic
│   │       ├── positional.py        #   Shelf-row grouping via DBSCAN
│   │       └── export.py            #   YOLO and retrieval format exporters
│   │
│   └── shared/                      # Shared utilities across all experiments
│       ├── hardware_utils.py        #   GPU/CPU detection, environment config
│       ├── metrics_logger.py        #   Metrics logging and visualisation
│       ├── dataset_utils.py         #   Dataset path management, YAML IO
│       ├── yolo_detector.py         #   YOLO inference wrapper
│       ├── train_recognition.py     #   Classification model training
│       └── ...                      #   Additional utility modules
│
├── logic/
│   └── dinov3_wrapper.py            # DINOv3 ViT-S/16+ feature extraction wrapper
│
├── scripts/
│   ├── generate_figures.py          # Generate 5 publication-quality thesis figures
│   ├── generate_yolo_montage.py     # YOLOv8n test detection montage
│   ├── generate_failure_montage.py  # Failure mode visualisation montage
│   └── populate_sku_catalogue.py    # Populate SKU catalogue from master/pre-master
│
├── demo-ui/                         # [DEPRECATED] Early prototype dashboard (mock data only).
                                 # The active annotation review tool is at:
                                 # experiments/curated_pipeline/review_server.py
│   ├── src/
│   │   ├── App.jsx                  #   Main application component
│   │   ├── App.css
│   │   └── main.jsx                 #   React entry point
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
│
├── configs/
│   ├── sources.yml                  # Literature search configuration
│   └── target_brands.yml            # Brand/product taxonomy (6 categories, 150+ brands)
│
├── models/                          # Trained model weights (Git LFS)
│   ├── SKU110K_V3.pt                #   YOLOv5 detection backbone (public SKU110K)
│   ├── yolov8n.pt                   #   YOLOv8n base pretrained
│   ├── yolov10n.pt                  #   YOLOv10n base pretrained
│   ├── yolov8n_best.pt              #   Fine-tuned YOLOv8n (mAP50 = 0.347)
│   ├── yolov10n_best.pt             #   Fine-tuned YOLOv10n (mAP50 = 0.287)
│   ├── yolov8n_curated_best.pt      #   Curated pipeline fine-tune
│   ├── recognition_resnet18.pt      #   ResNet18 classification head
│   └── recognition_vit.pt           #   ViT classification head
│
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project metadata
├── download_base_weights.sh         # Download base pretrained weights
├── download_gated_models.sh         # Instructions for Meta-gated DINOv3 weights
├── .gitattributes                   # Git LFS tracking configuration
└── .gitignore
```

---

## Prerequisites

- **Python** 3.10 or later
- **Git LFS** — for downloading large model weight files (`git lfs install`)
- **pip** and **venv** — standard Python package management
- **Node.js** 18+ (optional) — required only for the interactive demo UI
- **Ollama** (optional) — required only for VLM experiments (`ollama run qwen3-vl:4b`)
- **CUDA-capable GPU** with 6+ GB VRAM recommended (tested on RTX 3050 6GB laptop GPU)

---

## Setup

### 1. Clone and pull LFS weights

```bash
git clone <repository-url> thesis-sku-recognition
cd thesis-sku-recognition
git lfs pull
```

The `git lfs pull` command downloads all fine-tuned model weights (~750 MB total). If you skip this step, the model loading code will print clear error messages.

### 2. Create a virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or .venv\Scripts\activate  # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs all core dependencies: PyTorch (via Ultralytics), sentence-transformers, FAISS (CPU), Albumentations, Matplotlib, and supporting libraries.

### 4. Download base pretrained weights

```bash
bash download_base_weights.sh
```

Downloads YOLOv8n.pt and YOLOv10n.pt from the Ultralytics asset release to the `models/` directory. These are the base checkpoints used for fine-tuning. Weights already present are skipped.

### 5. Download gated model weights (DINOv3)

```bash
bash download_gated_models.sh
```

The curated pipeline uses **DINOv3 ViT-S/16+** for feature extraction. These weights are **gated by Meta (Facebook)** and require signing a research agreement at [github.com/facebookresearch/dinov3](https://github.com/facebookresearch/dinov3). The script prints instructions for manual download or accepts a `DINO_VITS16_URL` environment variable for automated download.

The expected location is `models/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` (~110 MB). The pipeline code auto-detects missing weights and prints a clear error message with download instructions.

---

## Dataset

The dataset used in this thesis consists of shelf images from German discount supermarkets (Aldi, Kaufland, Lidl, Netto), primarily covering the coffee, chocolate, and confectionery aisles. It contains **148 SKU classes** with **2,790 exemplar images** and associated bounding box annotations.

**The dataset is NOT included in this repository** due to licensing and size constraints. It is available separately on [Kaggle](https://kaggle.com/) (link placeholder — search for "SKU-Level Product Recognition Retail Shelf").

To reproduce the experiments, place the downloaded dataset such that:

```
Dataset/
├── raw/                     # Original shelf photographs
│   ├── Aldi/
│   ├── kaufland/
│   ├── Lidl/
│   └── netto/
├── recognition/             # Cropped product images by class
│   ├── train/
│   └── val/
└── processed_yolo/          # YOLO-format dataset (generated by prepare_data.py)
```

---

## Reproduction Commands

All commands should be run from the repository root unless otherwise noted.

### Chapter 4.2 — YOLO Detection Baseline

#### Data Preparation

```bash
python experiments/01_yolo_detection/prepare_data.py
```

Converts curated pipeline YOLO exports into a train/val/test split (stratified by store) with Albumentations augmentation (2× by default). Outputs to `experiments/01_yolo_detection/data/curated_yolo/`.

Options: `--dry-run` (preview), `--augment N` (override augmentation factor), `--force` (rebuild), `--seed N` (custom random seed).

#### Training

```bash
# YOLOv8n baseline (thesis primary model)
python experiments/01_yolo_detection/train.py \
    --model yolov8n.pt \
    --config experiments/01_yolo_detection/configs/yolov8_german_supermarket.yaml

# YOLOv10n baseline (speed comparison)
python experiments/01_yolo_detection/train.py \
    --model yolov10n.pt \
    --config experiments/01_yolo_detection/configs/yolov10_german_supermarket.yaml
```

The training script auto-detects the environment (local/Kaggle/Colab), configures hardware-appropriate batch sizes, and logs all metrics. Training on a 6GB GPU takes approximately 4–5 minutes for 100 epochs with nano variants.

#### Evaluation

```bash
python experiments/01_yolo_detection/evaluate.py \
    --model models/yolov8n_best.pt

python experiments/01_yolo_detection/evaluate.py \
    --model models/yolov10n_best.pt
```

Generates mAP50/mAP50-95 metrics, per-class breakdowns, confidence distributions, and entropy analysis. Use `--plot` for confusion matrices and `--save_json` for structured output.

#### Verification

```bash
python experiments/01_yolo_detection/verify_export.py
```

Validates YOLO export integrity: class ID ranges, bounding box coordinate validity, source image matching, and empty/duplicate file detection.

---

### Chapter 4.3 — Retrieval-Based Approaches

These experiments evaluate CLIP ViT-B/32 and DINOv2 ViT-B/14 for image-based and text-based SKU retrieval. Results are saved to `experiments/02_retrieval_system/results/`.

#### Experiment 1: CLIP Retrieval

```bash
python experiments/02_retrieval_system/exp1_clip_retrieval.py
```

Evaluates CLIP ViT-B/32 retrieval accuracy by querying each validation image against a FAISS index of 32,670 DETR-produced product crops. Reports top-1, top-5, and top-10 accuracy.

#### Experiment 2: DINOv2 Retrieval

```bash
python experiments/02_retrieval_system/exp2_dinov2_retrieval.py
```

Evaluates DINOv2 ViT-B/14 on a labeled-only subset (300 training images as database, 76 validation images as queries). Uses FAISS for nearest-neighbour search.

#### Experiment 3: Few-Shot N-Way K-Shot

```bash
python experiments/02_retrieval_system/exp3_fewshot.py
```

Episodic few-shot evaluation with CLIP ViT-B/32. Tests configurations: N = {5, 10, 20} ways, K = {1, 3, 5} shots, 100 episodes each. Reports mean ± std top-1 and top-5 accuracy.

#### Experiment 4: Text Query

```bash
python experiments/02_retrieval_system/exp4_text_query.py
```

Tests multiple CLIP text-prompt templates (`"a photo of {class_name}"`, `"a photo of {class_name} on a supermarket shelf"`, etc.) for zero-shot text-to-image retrieval across 52 SKU classes.

#### Supporting Tools

```bash
python experiments/02_retrieval_system/indexer.py     # Build FAISS index from product images
python experiments/02_retrieval_system/query.py        # Query FAISS index interactively
python experiments/02_retrieval_system/match_skus.py   # Match retrieval crops to SKU catalogue
```

---

### Chapter 4.4 — VLM Few-Shot Detection

These experiments evaluate local Vision-Language Models (Moondream 1.7B, LLaVA 7B, Qwen3-VL 4B) for zero-shot and few-shot product detection on dense shelf images.

#### VLM Annotation Pipeline

```bash
python experiments/03_vlm_fewshot/vlm_annotation_pipeline.py \
    --image_dir Dataset/labeling_set \
    --model qwen3-vl:4b \
    --max_products 20 \
    --german_text
```

Generates pseudo-labels with bounding boxes and generic SKU names using Ollama-hosted VLMs. Requires Ollama running locally. The pipeline produces structured JSON annotations with confidence scores and human-review workflow integration.

#### Zero-Shot / Few-Shot Evaluation

```bash
# Zero-shot detection on a single image
python experiments/03_vlm_fewshot/eval_vlm_ollama.py \
    --mode zero_shot \
    --image Dataset/raw/kaufland/IMG20260601171129.jpg

# Few-shot with reference images
python experiments/03_vlm_fewshot/eval_vlm_ollama.py \
    --mode few_shot \
    --image test.jpg \
    --shots Dataset/processed_yolo/images/train/*.jpg \
    --num_shots 3
```

**Note:** As documented in the thesis (§4.4.5), local VLMs exhibited significant hallucination, high latency (25–53 s/image), and poor performance on dense shelf configurations. These experiments serve as a negative baseline, confirming that lightweight local VLMs are not viable for fine-grained SKU recognition in their current form.

#### Conversion

```bash
python experiments/03_vlm_fewshot/vlm_converter.py \
    --input experiments/03_vlm_fewshot/annotations \
    --output Dataset/processed_yolo
```

Converts VLM-annotated JSON into YOLO-format training data.

---

### Chapter 4.5 — Curated Pipeline

The curated pipeline is the thesis's primary contribution: a comprehensive system that takes a shelf image through detection (YOLOv5 SKU110K), hybrid embedding (DINOv3 + MobileNetV2), exemplar-based matching, human-in-the-loop review, and multi-format export.

#### Main Pipeline

```bash
# Process a single shelf image through the full pipeline
python experiments/curated_pipeline/pipeline.py \
    --image experiments/curated_pipeline/data/input/aldi_001.jpg

# Process and export to YOLO + retrieval formats after review
python experiments/curated_pipeline/pipeline.py \
    --image experiments/curated_pipeline/data/input/aldi_001.jpg \
    --export yolo,retrieval

# Dry-run (detect + embed only, skip review)
python experiments/curated_pipeline/pipeline.py \
    --image experiments/curated_pipeline/data/input/aldi_001.jpg \
    --dry-run

# Resume most recent review session
python experiments/curated_pipeline/pipeline.py --resume

# List all SKUs in the registry
python experiments/curated_pipeline/pipeline.py --list-skus
```

#### Visual Review Server

```bash
python experiments/curated_pipeline/review_server.py
```

Launches an interactive web-based review tool (default: `http://127.0.0.1:8765`) where each detected product crop can be confirmed, renamed, skipped, or assigned to a new SKU. The review server enforces an explicit state model: crops start as `pending` → user confirms/skips → confirmed crops are committed to the SKU registry.

#### Label Conversion

```bash
# Convert YOLO .txt exports to LabelMe JSON for visual validation
python experiments/curated_pipeline/yolo_to_labelme.py

# Export SKU registry to YOLO training format
python experiments/curated_pipeline/export_yolo_from_registry.py
```

#### Pipeline Utilities

The pipeline is supported by a modular utility package at `experiments/curated_pipeline/pipeline_utils/`:

| Module | Purpose |
|--------|---------|
| `detection.py` | YOLOv5 object detection on full shelf images (SKU110K backbone) |
| `embeddings.py` | DINOv3 ViT-S/16+ (384-dim) + MobileNetV2 (1280-dim) hybrid embedding |
| `matching.py` | SKU registry management, similarity scoring, centroid computation |
| `positional.py` | DBSCAN-based shelf-row grouping, positional bonus scoring |
| `export.py` | YOLO-format and retrieval-format dataset export |

---

## Model Weights

### Fine-Tuned Weights (Git LFS)

| Model | Task | Size | mAP50 | mAP50:95 | Top-1 Acc. |
|-------|------|------|-------|----------|------------|
| `SKU110K_V3.pt` | Detection backbone | 353 MB | — | — | — |
| `yolov8n_best.pt` | Detection (YOLOv8n) | 6.7 MB | **0.347** | 0.266 | — |
| `yolov10n_best.pt` | Detection (YOLOv10n) | 5.8 MB | 0.287 | 0.229 | — |
| `yolov8n_curated_best.pt` | Detection (curated) | 6.7 MB | — | — | — |
| `recognition_resnet18.pt` | Classification | 43 MB | — | — | — |
| `recognition_vit.pt` | Classification | 328 MB | — | — | — |

**Note:** The curated pipeline retrieval accuracy (93.7% top-1) uses DINOv3+MobileNetV2 embeddings, not a classification head — no single weight file corresponds to the retrieval accuracy metric.

### Base Weights (Downloaded by `download_base_weights.sh`)

| Weight | Source | Size |
|--------|--------|------|
| `yolov8n.pt` | Ultralytics Assets | 6.2 MB |
| `yolov10n.pt` | Ultralytics Assets | 5.4 MB |

### Gated Weights (Not Included)

| Weight | Source | Access |
|--------|--------|--------|
| `models/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` | Meta Research | Requires signed research agreement at [github.com/facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) |

---

## Performance Summary

### Detection Baselines (Chapter 4.2)

| Metric | YOLOv8n | YOLOv10n |
|--------|:-------:|:--------:|
| **mAP50** | **0.347** | 0.287 |
| **mAP50-95** | **0.266** | 0.229 |
| Precision | 0.281 | **0.373** |
| Recall | **0.411** | 0.322 |
| Parameters | 3.34M | **2.33M** |
| GFLOPs | 9.6 | **6.9** |
| Inference (fused) | 33.7 ms/img | **4.1 ms/img** |
| VRAM peak | 3.0 GB | **2.0 GB** |

### Curated Pipeline Retrieval (Chapter 4.5)

| Metric | Hybrid (DINOv3+MNet) | CLIP ViT-B/32 (zero-shot) |
|--------|:--------------------:|:-------------------------:|
| **Top-1 accuracy** | **93.7%** | 56.9% |
| Top-5 accuracy | 99.3% | 82.4% |
| Top-10 accuracy | 99.5% | — |
| SKU classes | 148 | 148 |
| Query images | 2,790 | 940 |

### Few-Shot Retrieval (5-way K-shot, hybrid embeddings)

| Setting | Top-1 (mean ± std) | Top-5 |
|---------|:------------------:|:-----:|
| 5-way 1-shot | **96.0%** ± 8.5% | 100.0% |
| 5-way 3-shot | 99.4% ± 3.4% | 100.0% |
| 5-way 5-shot | 99.6% ± 2.8% | 100.0% |
| 10-way 1-shot | 93.4% ± 7.0% | 99.5% |
| 20-way 1-shot | 89.1% ± 7.5% | 98.7% |

### Gallery Size Scalability

| Gallery Size | Top-1 Accuracy |
|:-----------:|:--------------:|
| 10 SKUs | 100.0% |
| 20 SKUs | 95.0% |
| 30 SKUs | 96.7% |
| 50 SKUs | 98.0% |
| 80 SKUs | 90.0% |

### VLM Few-Shot Detection (Chapter 4.4)

| Model | Observations |
|-------|-------------|
| Moondream 1.7B | Failed on dense scenes; truncated/irrelevant outputs |
| LLaVA 7B | 25–53 s/image; severe prompt-induced hallucination |
| Qwen3-VL 4B | Multi-image API compatibility issues; dense object failure |

**Conclusion:** Lightweight local VLMs are not suitable for fine-grained SKU recognition in dense retail environments. The DINOv3+MobileNetV2 hybrid remains the superior architecture.

---

## Figure Generation

```bash
# Generate 5 publication-quality thesis figures (1200 DPI, <1 MB each)
python scripts/generate_figures.py

# Generate YOLOv8n test detection montage (2×3 grid)
python scripts/generate_yolo_montage.py

# Generate failure mode visualisation montage (6 documented failure modes)
python scripts/generate_failure_montage.py
```

Output directory: `output/figures/`

Figures generated:
1. **Hybrid Ablation** — Bar chart comparing DINOv3-only, MNet-only, and hybrid accuracy at K=1,3,5
2. **Few-Shot N-Way K-Shot** — Heatmap of accuracy across (N, K) configurations
3. **Gallery Size Scalability** — Line plot of top-1 accuracy vs. gallery size
4. **Per-SKU Retrieval Heatmap** — Confusion matrix of retrieval results
5. **CLIP vs Hybrid Comparison** — Side-by-side bar chart
6. **YOLO Test Detection Montage** — 2×3 grid of detection visualisations across diverse lighting/density conditions
7. **Failure Mode Montage** — 3×2 grid showing occlusion, reflection, lighting, and novel-variant failures

---

## Interactive Demo UI

**Note:** The `demo-ui/` directory contains only an early prototype dashboard (mock data, deprecated). It is **not** the primary annotation or review tool.

The **active** tools for interacting with the system are:

- **Visual Review Server** (recommended): `python experiments/curated_pipeline/review_server.py` — launches an interactive web UI for reviewing detected product crops, confirming or correcting SKU assignments, and managing the registry.
- **FAISS Query Interface**: `python experiments/02_retrieval_system/query.py` — query the retrieval index interactively from the command line.
- **Pipeline Batch Mode**: `python experiments/curated_pipeline/pipeline.py` — run the full detect-embed-match-review pipeline on new shelf images.

See the "Reproduction Commands" sections above for usage details.

---

## Results Logging

The repository uses a structured metrics logging system (`experiments/shared/metrics_logger.py`) that persists run metadata, hyperparameters, metrics, confusion matrices, and entropy distributions to timestamped JSON files. This enables post-hoc analysis and figure generation without re-running experiments.

---

## Extension Guide

The modular architecture supports several extension paths:

- **New detection backbones:** Add a new YOLO variant by extending `experiments/01_yolo_detection/train.py` and adding a hyperparameter config.
- **New embedding models:** Add feature extractors to `experiments/curated_pipeline/pipeline_utils/embeddings.py` following the `DINOv3Extractor` / `MobileNetV2Extractor` interface.
- **Additional SKU categories:** Extend `configs/target_brands.yml` with new brands; the SKU registry auto-assigns new IDs.
- **OCR integration:** The pipeline's current limitation is text-based disambiguation of visually similar SKUs. Future work should integrate a dedicated OCR module (EasyOCR, PaddleOCR) as a pre-processing or post-processing step.

---

## License

This work is licensed under the MIT License. See the `LICENSE` file for details.

**Third-party model licenses:**

- **YOLOv8, YOLOv10** — [AGPL-3.0](https://github.com/ultralytics/ultralytics) (Ultralytics)
- **DINOv3** — [CC-BY-NC-4.0](https://github.com/facebookresearch/dinov3) (Meta Research; research agreement required for weights)
- **CLIP** — [MIT](https://github.com/openai/CLIP) (OpenAI)
- **MobileNetV2** — [Apache 2.0](https://github.com/pytorch/vision) (PyTorch)
- **SKU110K** — [Custom (research)](https://github.com/eg4000/SKU110K_CVPR19) (CVPR 2019)

---

## Acknowledgments

This research was conducted at the Berlin School of Business and Innovation (BSBI) in collaboration with the University for the Creative Arts (UCA). The author thanks the academic supervisors and the research community for their guidance.

The SKU110K dataset [Goldman et al., CVPR 2019] provided the detection backbone pre-training. The DINOv3 self-supervised learning framework [Oquab et al., 2024] and the Ultralytics YOLO ecosystem [Jocher et al., 2023] were instrumental to the experimental design.

---

## References

Brandes, D. and Brandes, N. (2012). *BARE ESSENTIALS the ALDI Way to Retail Success*. Books on Demand.

Cai, Y., Wen, L., Zhang, L., Du, D. and Wang, W. (2021). Rethinking Object Detection in Retail Stores. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(2), pp.947 954. doi:10.1609/aaai.v35i2.16178.

Cao, M. et al. (2023). Recognition of Occluded Goods under Prior Inference Based on Generative Adversarial Network. *Sensors*, 23(6), pp.3355. doi:10.3390/s23063355.

Deng, J. et al. (2022). ArcFace: Additive Angular Margin Loss for Deep Face Recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(10), pp.5962 5979. doi:10.1109/TPAMI.2021.3087709.

Desmarescaux, M. et al. (2025). A Review: One-Shot Object Detection Methods for Conditional Detection of Retail and Warehouse Products. *Neural Processing Letters*, 57(2). doi:10.1007/s11063-025-11740-4.

Dong, H. et al. (2023). Detection of Occluded Small Commodities Based on Feature Enhancement under Super-Resolution. *Sensors*, 23(5), pp.2439. doi:10.3390/s23052439.

Follmann, P. et al. (2018). MVTec D2S: Densely Segmented Supermarket Dataset. *Proceedings of the European Conference on Computer Vision (ECCV)*, pp.569 585.

Fu, Y. et al. (2025). NTIRE 2025 Challenge on Cross-Domain Few-Shot Object Detection: Methods and Results. *arXiv preprint*, arXiv:2504.10685.

Ge, Z. et al. (2021). YOLOX: Exceeding YOLO Series in 2021. *arXiv preprint*, arXiv:2107.08430.

Geng, W. et al. (2018). Fine-Grained Grocery Product Recognition by One-Shot Learning. *Proceedings of ACM Multimedia*.

George, M. and Floerkemeier, C. (2014). Recognizing Products: a Per-exemplar Multi-label Image Classification Approach. *Proceedings of the European Conference on Computer Vision (ECCV)*, pp.440 455.

Goldman, E. et al. (2019). Precise Detection in Densely Packed Scenes. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp.5227 5236.

Guirguis, K. et al. (2022). Few-Shot Object Detection in Unseen Domains. *arXiv preprint*, arXiv:2204.05072.

Jund, P. et al. (2016). The Freiburg Groceries Dataset. *arXiv preprint*, arXiv:1611.05799.

Klasson, M. et al. (2019). A Hierarchical Grocery Store Image Dataset with Visual and Semantic Labels. *Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV)*.

Laitala, J. and Ruotsalainen, L. (2023). Computer Vision Based Planogram Compliance Evaluation. *Applied Sciences*, 13(18), 10145. doi:10.3390/app131810145.

Lamm, B. and Keuper, J. (2024). Retail-786k: a Large-Scale Dataset for Visual Entity Matching. *arXiv preprint*, arXiv:2309.17164.

Oquab, M. et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. *arXiv preprint*, arXiv:2304.07193.

Ou, T.-Y. et al. (2025). Real-time retail planogram compliance application using computer vision and virtual shelves. *Scientific Reports*, 15(1). doi:10.1038/s41598-025-86026-3.

Peng, J., Xiao, C. and Li, Y. (2020). RP2K: A Large-Scale Retail Product Dataset for Fine-Grained Image Classification. *arXiv preprint*, arXiv:2006.12634.

Pietrini, R. et al. (2024). Shelf Management: A deep learning-based system for shelf visual monitoring. *Expert Systems with Applications*, 255, 124635. doi:10.1016/j.eswa.2024.124635.

Saleh, K., Szénási, S. and Vámossy, Z. (2021). Occlusion Handling in Generic Object Detection: A Review. *IEEE International Conference on Systems, Automation, and Measurement (SAMI)*.

Schroff, F., Kalenichenko, D. and Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp.815 823.

Šikić, F. et al. (2024). Enhanced Out-of-Stock Detection in Retail Shelf Images Based on Deep Learning. *Sensors*, 24(2), 693. doi:10.3390/s24020693.

Siméoni, O. et al. (2025). DINOv3. *arXiv preprint*, arXiv:2508.10104.

Srivastava, M.M. (2020). Bag of Tricks for Retail Product Image Classification. *arXiv preprint*, arXiv:2001.03992.

Srivastava, M.M. (2023). RetailKLIP: Finetuning OpenCLIP backbone using metric learning on a single GPU for Zero-shot retail product image classification. *arXiv preprint*, arXiv:2312.10282.

Tan, L. et al. (2024). Enhanced Self-Checkout System for Retail Based on Improved YOLOv10. *Journal of Imaging*, 10(10), 248. doi:10.3390/jimaging10100248.

Tur, A.O. et al. (2024). Exploring Fine-grained Retail Product Discrimination with Zero-shot Object Classification Using Vision-Language Models. *IEEE International Conference on Recent Trends in Systems Innovation (RTSI)*, pp.97 102.

Wang, A. et al. (2020). Robust Object Detection under Occlusion with Context-Aware CompositionalNets. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

Wei, X.-S. et al. (2019). RPC: A Large-Scale Retail Product Checkout Dataset. *arXiv preprint*, arXiv:1901.07249.

Yazdanjouei, H. et al. (2025). A Co-Training Semi-Supervised Framework Using Faster R-CNN and YOLO Networks for Object Detection in Densely Packed Retail Images. *arXiv preprint*, arXiv:2509.09750.

Yücel, M.E. and Ünsalan, C. (2022). Planogram Compliance Control via Object Detection, Sequence Alignment, and Focused Iterative Search. *arXiv preprint*, arXiv:2212.01004.

Zhang, Z. et al. (2021). ViT-YOLO: Transformer-Based YOLO for Object Detection. *Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)*, pp.2799 2808.

Zhao, C., Wan, J. and Chan, A.B. (2025). Density-based Object Detection in Crowded Scenes. *arXiv preprint*, arXiv:2504.09819.

Zhao, Y. et al. (2025). LSR-YOLO: A lightweight and fast model for retail products detection. *PLOS One*, 20(10), e0334216. doi:10.1371/journal.pone.0334216.
