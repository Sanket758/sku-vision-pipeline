# VLM Annotation Pipeline - README

## Overview

This repository contains a **Vision-Language Model (VLM) annotation pipeline** for German supermarket shelf images, designed to generate pseudo-labels for SKU recognition experiments.

The pipeline implements the **Phase 1** of the game plan described in the advice, transforming Experiment 03 from a simple evaluation tool into a **data annotation engine** that feeds all other experiments.

## Key Features

### 1. VLM-Powered Annotation
- **Model**: qwen3-vl:4b (Ollama) for annotation generation
- **Output**: Structured JSON with bounding boxes and generic SKU names
- **Language**: Handles German text on packaging
- **Format**: YOLO-compatible annotations

### 2. Human-in-the-Loop Workflow
- **Spot-check**: You verify, don't create
- **Active learning**: Iterative improvement loop
- **Quality control**: Confidence thresholds and validation

### 3. Integration with Existing Experiments
- **Experiment 01**: YOLO Detection (Phase 2)
- **Experiment 02**: Retrieval System (Phase 3)
- **Experiment 03**: Comparative Evaluation (Phase 4)

### 4. Flexible SKU Naming
- **Generic SKUs**: SKU-0001, SKU-0002, etc.
- **No exact product names required**
- **Focus on product categories**

## Directory Structure

```
experiments/03_vlm_fewshot/
├── vlm_annotation_pipeline.py          # Main annotation pipeline
├── vlm_converter.py                    # Annotation converter and integrator
├── configs/                            # Configuration files
│   └── vlm_annotation.yaml             # Pipeline configuration
├── prompts/                            # Prompt templates
├── annotations/                        # Output annotations (created)
├── test_pipeline.py                    # Test script
└── README.md                            # This file
```

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start Ollama with qwen3-vl:4b
ollama run qwen3-vl:4b
```

### 2. Run the VLM Annotation Pipeline

```bash
# Run full annotation pipeline
python vlm_annotation_pipeline.py \
  --image_dir Dataset/labeling_set \
  --output_dir experiments/03_vlm_fewshot/annotations \
  --model qwen3-vl:4b \
  --max_products 20 \
  --german_text

# Or run with custom configuration
python vlm_annotation_pipeline.py \
  --config configs/custom_config.yaml
```

### 3. Convert Annotations to YOLO Format

```bash
# Convert VLM annotations to YOLO format
python vlm_converter.py \
  --input experiments/03_vlm_fewshot/annotations \
  --output Dataset/processed_yolo \
  --yolo-config experiments/01_yolo_detection/configs/yolov8_german_supermarket.yaml \
  --retrieval-config experiments/02_retrieval_system/config.yaml
```

### 4. Test the Pipeline

```bash
# Run test script
python test_pipeline.py
```

## Configuration

### Main Configuration (`configs/vlm_annotation.yaml`)

```yaml
# Ollama Configuration
ollama_url: "http://localhost:11434"
model_name: "qwen3-vl:4b"

# Annotation Parameters
max_products: 20
confidence_threshold: 0.7
use_german_text: true
output_format: "yolo"

# Processing Configuration
batch_size: 5
retry_attempts: 3
retry_delay: 5

# Output Configuration
output_dir: "experiments/03_vlm_fewshot/annotations"
save_detailed: true
save_yolo_format: true
save_summary: true

# German Text Handling
german_text_handling:
  preserve_case: true
  transliterate: true
  category_mapping:
    "getränk": "beverage"
    "milch": "dairy"
    "brot": "bakery"
    # ... more mappings
```

### Converter Configuration (`configs/converter_config.yaml`)

```yaml
# Dataset Configuration
input_dir: "experiments/03_vlm_fewshot/annotations"
output_dir: "Dataset/processed_yolo"
train_split: 0.7
val_split: 0.2
test_split: 0.1

# Class Configuration
class_mapping:
  "SKU-0001": 0
  "SKU-0002": 1
  "SKU-0003": 2
  # ... more classes

# Processing Options
generate_yolo_dataset: true
generate_retrieval_data: true
create_splits: true
backup_existing: true
```

## How It Works

### Phase 1: VLM Annotation Pipeline

1. **Image Processing**: Load shelf images from `Dataset/labeling_set`
2. **Prompt Generation**: Create German-aware prompts
3. **VLM Inference**: Call Ollama with qwen3-vl:4b
4. **Response Parsing**: Extract bounding boxes and SKU names
5. **Human Review**: You spot-check and validate
6. **Output**: Save annotations in multiple formats

### Phase 2: Annotation Conversion

1. **Format Conversion**: Convert VLM JSON to YOLO format
2. **Dataset Creation**: Create YOLO-compatible dataset structure
3. **Split Generation**: Create train/val/test splits
4. **Retrieval Data**: Generate crops for retrieval system
5. **Integration**: Connect with existing experiments

### Phase 3: Experiment Integration

1. **YOLO Detection**: Use converted annotations for training
2. **Retrieval System**: Use crops for similarity matching
3. **Comparative Evaluation**: Compare all three systems

## Expected Outputs

### VLM Annotations (`experiments/03_vlm_fewshot/annotations/`)

```
experiments/03_vlm_fewshot/annotations/
├── vlm_annotations.json              # Detailed VLM results
├── yolo_annotations.json             # YOLO format annotations
├── annotation_summary.json            # Processing summary
└── metrics/                          # Performance metrics
```

### YOLO Dataset (`Dataset/processed_yolo/`)

```
Dataset/processed_yolo/
├── data.yaml                          # YOLO dataset config
├── images/                            # Image symlinks
│   ├── train/
│   ├── val/
│   └── test/
├── labels/                            # Label files
│   ├── train/
│   ├── val/
│   └── test/
├── crops/                             # Product crops for retrieval
│   ├── train/
│   ├── val/
│   └── test/
├── metadata/                          # Dataset metadata
├── splits/                            # Train/val/test splits
└── retrieval_index.json               # Retrieval system index
```

## Usage Examples

### Example 1: Basic Usage

```bash
# Process 10 images with default settings
python vlm_annotation_pipeline.py \
  --image_dir Dataset/labeling_set \
  --output_dir experiments/03_vlm_fewshot/annotations \
  --max_products 10
```

### Example 2: German Text Handling

```bash
# Enable German text handling
python vlm_annotation_pipeline.py \
  --image_dir Dataset/labeling_set \
  --german_text \
  --confidence 0.8
```

### Example 3: Custom Configuration

```bash
# Use custom configuration
python vlm_annotation_pipeline.py \
  --config configs/custom_config.yaml
```

## Troubleshooting

### Common Issues

1. **Ollama Not Running**
   ```bash
   # Start Ollama
   ollama serve
   ollama run qwen3-vl:4b
   ```

2. **Model Not Found**
   ```bash
   # Pull the model
   ollama pull qwen3-vl:4b
   ```

3. **Permission Errors**
   ```bash
   # Check file permissions
   chmod -R 755 Dataset/labeling_set
   ```

4. **Memory Issues**
   ```bash
   # Reduce batch size
   python vlm_annotation_pipeline.py --batch_size 1
   ```

### Debugging

```bash
# Run test script
python test_pipeline.py

# Check logs
tail -f experiments/03_vlm_fewshot/annotations/metrics.log

# View output
less experiments/03_vlm_fewshot/annotations/vlm_annotations.json
```

## Integration with Existing Experiments

### Experiment 01: YOLO Detection

The converted annotations are automatically integrated with Experiment 01:

```bash
# Train YOLO model on annotated data
cd experiments/01_yolo_detection
python train.py --model yolov8m.pt --config configs/yolov8_german_supermarket.yaml
```

### Experiment 02: Retrieval System

The generated crops are used by Experiment 02:

```bash
# Run retrieval system evaluation
cd experiments/02_retrieval_system
python exp1_clip_retrieval.py
```

### Experiment 03: Comparative Evaluation

All three systems are compared using the same test set:

```bash
# Run comparative evaluation
cd experiments/03_vlm_fewshot
python comparative_evaluation.py
```

## Future Enhancements

### 1. Active Learning

- Implement active learning loop
- Use YOLO predictions to select images for VLM review
- Iteratively improve both VLM and YOLO models

### 2. Multi-Model Support

- Support for other VLMs (llava, gemini, etc.)
- Model comparison and selection
- Performance benchmarking

### 3. Advanced German Text Processing

- OCR integration for text extraction
- Translation and normalization
- Brand name recognition

### 4. Quality Assurance

- Automated quality checks
- Human feedback integration
- Continuous improvement loops

## License

This project is part of the BSBI Masters Thesis project. All rights reserved.

## Contact

For questions or issues, please refer to the thesis documentation or contact the project maintainer.
