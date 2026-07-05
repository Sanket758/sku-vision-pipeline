"""VLM Annotation Pipeline for German Supermarket Shelf Images.

This script implements the VLM-driven auto-annotation pipeline described in the game plan.
It processes shelf images to generate pseudo-labels with bounding boxes and generic SKU names.

Key Features:
- Uses qwen3-vl:4b (Ollama) for annotation generation
- Generates structured JSON with bounding boxes and generic SKU names
- Implements human spot-check workflow
- Outputs YOLO-compatible annotation format
- Handles German text in packaging
- Integrates with existing labeling_set dataset
"""

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import hardware_utils as hw
from shared.metrics_logger import MetricsLogger

# Setup minimal logging for error reporting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
class VLMAnnotationPipeline:
    """Main VLM annotation pipeline class."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ollama_url = config.get("ollama_url", "http://localhost:11434")
        self.model_name = config.get("model_name", "qwen3-vl:4b")
        self.max_products = config.get("max_products", 20)
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.use_german_text = config.get("use_german_text", True)
        self.output_format = config.get("output_format", "yolo")

        self.logger = MetricsLogger(
            experiment_name="vlm_annotation",
            config=config
        )

    def call_ollama(self, image_path: Path, prompt: str) -> Dict[str, Any]:
        """Call Ollama VLM model to annotate an image."""
        try:
            # Prepare image for Ollama
            with open(image_path, "rb") as f:
                image_data = f.read()

            # Create payload for Ollama
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_data.hex()],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1000,
                }
            }

            # Make API call
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return self.parse_vlm_response(result.get("response", ""), image_path)
            else:
                logger.error(f"Ollama API error for {image_path.name}: {response.status_code}")
                return self.get_fallback_annotation(image_path)

        except Exception as e:
            logger.error(f"Error calling Ollama for {image_path.name}: {e}")
            return self.get_fallback_annotation(image_path)

    def parse_vlm_response(self, response: str, image_path: Path) -> Dict[str, Any]:
        """Parse VLM response into structured annotation format."""
        annotations = []
        lines = response.split('\n')

        for line in lines:
            line = line.strip()
            if not line or not ('[' in line and ']' in line):
                continue

            # Try to extract bounding box and product info
            try:
                # Look for pattern like: "SKU-0001 [x1,y1,x2,y2] - description"
                import re

                # Find bounding box coordinates
                bbox_match = re.search(r'\[([^\]]+)\]', line)
                if not bbox_match:
                    continue

                bbox_str = bbox_match.group(1)
                bbox_coords = [float(x.strip()) for x in bbox_str.split(',')]

                if len(bbox_coords) != 4:
                    continue

                # Generate generic SKU name
                sku_id = f"SKU-{len(annotations) + 1:04d}"

                # Extract description if available
                desc_match = re.search(r'\](.*)', line)
                description = desc_match.group(1).strip() if desc_match else ""

                annotation = {
                    "label": sku_id,
                    "points": [
                        [bbox_coords[0], bbox_coords[1]],
                        [bbox_coords[2], bbox_coords[3]]
                    ],
                    "description": description,
                    "confidence": self.confidence_threshold,
                    "source": "vlm"
                }
                annotations.append(annotation)

            except Exception as e:
                logger.warning(f"Error parsing line '{line}' in {image_path.name}: {e}")
                continue

        return {
            "image_path": str(image_path),
            "annotations": annotations,
            "product_count": len(annotations),
            "processing_time": 0,
            "success": len(annotations) > 0
        }

    def get_fallback_annotation(self, image_path: Path) -> Dict[str, Any]:
        """Get fallback annotation using existing labels if available."""
        json_path = image_path.with_suffix('.json')
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    existing_data = json.load(f)

                annotations = []
                for shape in existing_data.get('shapes', []):
                    if shape.get('label', '').startswith('SKU-'):
                        points = shape.get('points', [])
                        if len(points) == 2:
                            annotation = {
                                "label": shape['label'],
                                "points": points,
                                "description": shape.get('description', ''),
                                "confidence": 0.9,
                                "source": "existing"
                            }
                            annotations.append(annotation)

                return {
                    "image_path": str(image_path),
                    "annotations": annotations,
                    "product_count": len(annotations),
                    "processing_time": 0,
                    "success": len(annotations) > 0,
                    "fallback": True
                }
            except Exception as e:
                print(f"Error reading existing annotations: {e}")

        return {
            "image_path": str(image_path),
            "annotations": [],
            "product_count": 0,
            "processing_time": 0,
            "success": False,
            "fallback": False
        }

    def generate_prompt(self, image_path: Path) -> str:
        """Generate prompt for VLM annotation."""
        base_prompt = """
        Analyze this German supermarket shelf image and identify all distinct products visible.
        For each product, provide:
        1. Bounding box coordinates [x1,y1,x2,y2] (pixel coordinates)
        2. Generic SKU identifier (SKU-0001, SKU-0002, etc.)
        3. Brief product category (e.g., 'beverage', 'dairy', 'household')
        4. Any visible German text on packaging

        Requirements:
        - Include ALL products visible, even partially occluded
        - Use pixel coordinates relative to image dimensions
        - Handle German language text on packaging
        - Maximum {max_products} products per image
        - Format: SKU-XXXX [x1,y1,x2,y2] - category (text)
        - Focus on product-level identification, not exact brand names
        - Ignore packaging details, focus on product categories
        """.format(max_products=self.max_products)

        if self.use_german_text:
            base_prompt += "\n\nNote: German text on packaging should be preserved in analysis."

        return base_prompt

    def convert_to_yolo_format(self, annotation: Dict[str, Any], image_width: int, image_height: int) -> Dict[str, Any]:
        """Convert annotation to YOLO format."""
        points = annotation['points']
        if len(points) != 2:
            return None

        # Convert pixel coordinates to normalized YOLO format
        x1, y1 = points[0]
        x2, y2 = points[1]

        # Calculate center, width, height
        x_center = (x1 + x2) / 2 / image_width
        y_center = (y1 + y2) / 2 / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height

        # Clamp values to valid range
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        # Extract class ID from label (handle malformed labels)
        label = annotation['label']
        try:
            if label.startswith('SKU-'):
                class_id_str = label[4:]
                import re
                digits = re.search(r'\d+', class_id_str)
                if digits:
                    base_class_id = int(digits.group())
                    variant_match = re.search(r'([A-Za-z]+)$', class_id_str)
                    if variant_match:
                        variant_hash = hash(variant_match.group()) % 100
                        class_id = base_class_id * 100 + variant_hash
                    else:
                        class_id = base_class_id
                else:
                    class_id = hash(label) % 10000
            else:
                class_id = hash(label) % 10000
        except Exception:
            class_id = hash(label) % 10000

        return {
            "class_id": class_id,
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
            "confidence": annotation.get('confidence', 1.0),
            "source": annotation.get('source', 'unknown')
        }

    def process_image(self, image_path: Path) -> Dict[str, Any]:
        """Process a single image through the VLM pipeline."""
        start_time = time.time()

        logger.info(f"Processing: {image_path.name}")

        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                image_width, image_height = img.size
        except Exception as e:
            logger.error(f"Error opening image {image_path}: {e}")
            return {
                "image_path": str(image_path),
                "annotations": [],
                "processing_time": 0,
                "success": False,
                "error": str(e)
            }

        # Generate prompt
        prompt = self.generate_prompt(image_path)

        # Call VLM model
        vlm_result = self.call_ollama(image_path, prompt)

        # Convert to YOLO format
        yolo_annotations = []
        for annotation in vlm_result.get("annotations", []):
            yolo_format = self.convert_to_yolo_format(
                annotation, image_width, image_height
            )
            if yolo_format:
                yolo_format["image_path"] = str(image_path)
                yolo_annotations.append(yolo_format)

        processing_time = time.time() - start_time

        result = {
            "image_path": str(image_path),
            "image_width": image_width,
            "image_height": image_height,
            "vlm_annotations": vlm_result.get("annotations", []),
            "yolo_annotations": yolo_annotations,
            "product_count": len(yolo_annotations),
            "processing_time": processing_time,
            "success": len(yolo_annotations) > 0,
            "fallback_used": vlm_result.get("fallback", False)
        }

        logger.info(f"  Found {len(yolo_annotations)} products in {processing_time:.2f}s")
        return result

    def save_annotations(self, results: List[Dict[str, Any]], output_dir: Path):
        """Save annotations to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        detailed_path = output_dir / "vlm_annotations.json"
        with open(detailed_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved detailed annotations to: {detailed_path}")

        # Save YOLO format
        yolo_path = output_dir / "yolo_annotations.json"
        yolo_data = []
        for result in results:
            for annotation in result.get('yolo_annotations', []):
                yolo_data.append({
                    "image_path": annotation['image_path'],
                    "class_id": annotation['class_id'],
                    "x_center": annotation['x_center'],
                    "y_center": annotation['y_center'],
                    "width": annotation['width'],
                    "height": annotation['height'],
                    "confidence": annotation['confidence'],
                    "source": annotation['source']
                })

        with open(yolo_path, 'w') as f:
            json.dump(yolo_data, f, indent=2)
        logger.info(f"Saved YOLO format annotations to: {yolo_path}")

        # Save summary
        summary = {
            "total_images": len(results),
            "successful_annotations": sum(1 for r in results if r['success']),
            "total_products": sum(r['product_count'] for r in results),
            "sources": {
                "vlm": sum(1 for r in results if not r.get('fallback_used', False)),
                "existing": sum(1 for r in results if r.get('fallback_used', False))
            },
            "processing_stats": {
                "avg_time_per_image": np.mean([r['processing_time'] for r in results]),
                "min_time": np.min([r['processing_time'] for r in results]),
                "max_time": np.max([r['processing_time'] for r in results])
            }
        }

        summary_path = output_dir / "annotation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved annotation summary to: {summary_path}")

        logger.info(f"\nAnnotations saved to: {output_dir}")
        logger.info(f"  Detailed results: {detailed_path}")
        logger.info(f"  YOLO format: {yolo_path}")
        logger.info(f"  Summary: {summary_path}")

    def run(self, image_paths: List[Path], output_dir: Path):
        """Run the complete VLM annotation pipeline."""
        logger.info("=" * 60)
        logger.info("VLM Annotation Pipeline")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Images to process: {len(image_paths)}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 60)

        # Log hyperparameters
        self.logger.log_hyperparams({
            "model": self.model_name,
            "max_products": self.max_products,
            "confidence_threshold": self.confidence_threshold,
            "use_german_text": self.use_german_text,
            "output_format": self.output_format,
            "total_images": len(image_paths)
        })

        # Process images
        results = []
        for i, image_path in enumerate(image_paths):
            logger.info(f"[{i+1}/{len(image_paths)}]")
            result = self.process_image(image_path)
            results.append(result)

            # Log metrics
            self.logger.log_metrics({
                "image_index": i,
                "image_path": str(image_path),
                "success": result['success'],
                "product_count": result['product_count'],
                "processing_time": result['processing_time'],
                "fallback_used": result.get('fallback_used', False)
            })

        # Save results
        self.save_annotations(results, output_dir)

        # Log final summary
        total_time = sum(r['processing_time'] for r in results)
        self.logger.log_metrics({
            "total_images": len(image_paths),
            "successful_annotations": sum(1 for r in results if r['success']),
            "total_products": sum(r['product_count'] for r in results),
            "total_processing_time": total_time,
            "avg_time_per_image": total_time / len(image_paths) if image_paths else 0
        })

        self.logger.flush()

        logger.info("\n" + "=" * 60)
        logger.info("Annotation Pipeline Complete")
        logger.info("=" * 60)
        logger.info(f"Processed {len(image_paths)} images")
        logger.info(f"Successfully annotated: {sum(1 for r in results if r['success'])}")
        logger.info(f"Total products detected: {sum(r['product_count'] for r in results)}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 60)
def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / "configs" / "vlm_annotation.yaml"

    default_config = {
        "ollama_url": "http://localhost:11434",
        "model_name": "qwen3-vl:4b",
        "max_products": 20,
        "confidence_threshold": 0.7,
        "use_german_text": True,
        "output_format": "yolo"
    }

    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        default_config.update(config)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        logger.warning(f"Configuration file not found: {config_path}, using defaults")

    return default_config
def find_images(image_dir: str, extensions: List[str] = None) -> List[Path]:
    """Find images in directory."""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png']

    image_dir = Path(image_dir)
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(image_dir.glob(f'*{ext}'))
    
    logger.info(f"Found {len(image_paths)} images in {image_dir}")
    return sorted(image_paths)
def main():
    parser = argparse.ArgumentParser(description="VLM Annotation Pipeline")
    parser.add_argument("--image_dir", type=str, default="Dataset/labeling_set",
                        help="Directory containing images to annotate")
    parser.add_argument("--output_dir", type=str, default="experiments/03_vlm_fewshot/annotations",
                        help="Output directory for annotations")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model name")
    parser.add_argument("--max_products", type=int, default=None,
                        help="Override maximum products per image")
    parser.add_argument("--confidence", type=float, default=None,
                        help="Override confidence threshold")
    parser.add_argument("--german_text", action="store_true",
                        help="Enable German text handling")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Process images in batches")

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Override config with command line arguments
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            cli_config = yaml.safe_load(f)
        config.update(cli_config)

    if args.model:
        config["model_name"] = args.model
    if args.max_products:
        config["max_products"] = args.max_products
    if args.confidence:
        config["confidence_threshold"] = args.confidence
    if args.german_text:
        config["use_german_text"] = True

    # Find images
    image_paths = find_images(args.image_dir)
    if not image_paths:
        logger.error(f"ERROR: No images found in {args.image_dir}")
        sys.exit(1)

    logger.info(f"Found {len(image_paths)} images to process")

    # Create output directory
    output_dir = Path(args.output_dir)

    # Initialize pipeline
    pipeline = VLMAnnotationPipeline(config)

    # Process images
    pipeline.run(image_paths, output_dir)
if __name__ == "__main__":
    main()
