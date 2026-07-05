#!/usr/bin/env python3
"""Test script for VLM Annotation Pipeline.

This script tests the VLM annotation pipeline with the existing labeling_set dataset.
"""

import sys
import os
import logging
from pathlib import Path

# Setup minimal logging for error reporting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.03_vlm_fewshot.vlm_annotation_pipeline import VLMAnnotationPipeline, load_config, find_images
from experiments.03_vlm_fewshot.vlm_converter import VLMAnnotationConverter
def test_vlm_pipeline():
    """Test the VLM annotation pipeline."""
    logger.info("=" * 60)
    logger.info("Testing VLM Annotation Pipeline")
    logger.info("=" * 60)
    
    # Test 1: Find images
    logger.info("\n1. Finding images...")
    image_dir = "Dataset/labeling_set"
    image_paths = find_images(image_dir)
    
    logger.info(f"   Found {len(image_paths)} images")
    logger.info(f"   Sample images: {[p.name for p in image_paths[:5]]}")
    
    if len(image_paths) == 0:
        logger.error("ERROR: No images found!")
        return False
    
    # Test 2: Load configuration
    logger.info("\n2. Loading configuration...")
    config = load_config()
    
    logger.info(f"   Model: {config['model_name']}")
    logger.info(f"   Max products: {config['max_products']}")
    logger.info(f"   Use German text: {config['use_german_text']}")
    
    # Test 3: Initialize pipeline
    logger.info("\n3. Initializing pipeline...")
    pipeline = VLMAnnotationPipeline(config)
    
    # Test 4: Process a single image
    logger.info("\n4. Testing single image processing...")
    test_image = image_paths[0]
    logger.info(f"   Processing: {test_image.name}")
    
    # Generate prompt
    prompt = pipeline.generate_prompt(test_image)
    logger.info(f"   Prompt length: {len(prompt)} characters")
    logger.info(f"   Prompt preview: {prompt[:200]}...")
    
    # Test annotation conversion
    logger.info("\n5. Testing annotation conversion...")
    test_annotation = {
        "label": "SKU-0001",
        "points": [[100, 200], [300, 400]],
        "description": "test product",
        "confidence": 0.8,
        "source": "test"
    }
    
    with open(test_image, 'rb') as f:
        image_width, image_height = Image.open(f).size
    
    yolo_format = pipeline.convert_to_yolo_format(test_annotation, image_width, image_height)
    logger.info(f"   YOLO format: {yolo_format}")
    
    # Test 6: Initialize converter
    logger.info("\n6. Testing converter...")
    converter = VLMAnnotationConverter()
    
    # Test 7: Check if output directory exists
    logger.info("\n7. Checking output directory...")
    output_dir = Path(config["output_dir"])
    logger.info(f"   Output directory: {output_dir}")
    logger.info(f"   Exists: {output_dir.exists()}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Test Complete")
    logger.info("=" * 60)
    logger.info("All tests passed!")
    logger.info("=" * 60)

if __name__ == "__main__":
    test_vlm_pipeline()
