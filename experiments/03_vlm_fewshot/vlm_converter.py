"""VLM Annotation Converter and Integration Script.

This script converts VLM annotations to YOLO format and integrates with the existing
experiment pipeline. It handles:
1. Converting VLM JSON annotations to YOLO format
2. Integrating with existing YOLO detection experiment
3. Preparing data for retrieval system
4. Generating training/validation splits
5. Creating annotation metadata for downstream tasks
"""

import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import yaml
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import hardware_utils as hw

# Setup minimal logging for error reporting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
class VLMAnnotationConverter:
    """Convert VLM annotations to YOLO format and integrate with experiments."""

    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path)
        self.setup_directories()

    def load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        default_config = {
            "input_dir": "experiments/03_vlm_fewshot/annotations",
            "output_dir": "Dataset/processed_yolo",
            "yolo_config_path": "experiments/01_yolo_detection/configs/yolov8_german_supermarket.yaml",
            "retrieval_config_path": "experiments/02_retrieval_system/config.yaml",
            "train_split": 0.7,
            "val_split": 0.2,
            "test_split": 0.1,
            "min_annotations_per_image": 1,
            "max_annotations_per_image": 50,
            "class_mapping": {
                "SKU-0001": 0,
                "SKU-0002": 1,
                "SKU-0003": 2,
                "SKU-0004": 3,
                "SKU-0005": 4,
                "SKU-0006": 5,
                "SKU-0007": 6,
                "SKU-0008": 7,
                "SKU-0009": 8,
                "SKU-0010": 9,
            },
            "generate_yolo_dataset": True,
            "generate_retrieval_data": True,
            "create_splits": True,
            "backup_existing": True
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            default_config.update(user_config)

        return default_config

    def setup_directories(self):
        """Create necessary output directories."""
        directories = [
            self.config["output_dir"],
            f"{self.config['output_dir']}/images/train",
            f"{self.config['output_dir']}/images/val",
            f"{self.config['output_dir']}/images/test",
            f"{self.config['output_dir']}/labels/train",
            f"{self.config['output_dir']}/labels/val",
            f"{self.config['output_dir']}/labels/test",
            f"{self.config['output_dir']}/crops/train",
            f"{self.config['output_dir']}/crops/val",
            f"{self.config['output_dir']}/crops/test",
            f"{self.config['output_dir']}/metadata",
            f"{self.config['output_dir']}/splits",
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def load_vlm_annotations(self) -> List[Dict[str, Any]]:
        """Load VLM annotations from JSON file."""
        input_file = Path(self.config["input_dir"]) / "vlm_annotations.json"
        
        if not input_file.exists():
            raise FileNotFoundError(f"VLM annotations not found: {input_file}")

        with open(input_file, 'r') as f:
            data = json.load(f)

        return data

    def convert_to_yolo_format(self, vlm_annotation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert VLM annotation to YOLO format."""
        yolo_annotations = []
        
        image_width = vlm_annotation.get("image_width", 1920)
        image_height = vlm_annotation.get("image_height", 1080)
        
        for yolo_anno in vlm_annotation.get("yolo_annotations", []):
            class_id = yolo_anno.get("class_id")
            
            # Check if class_id is in mapping
            if class_id not in self.config["class_mapping"].values():
                # Create new class ID
                new_class_id = len(self.config["class_mapping"].values())
                self.config["class_mapping"][f"SKU-{new_class_id+1:04d}"] = new_class_id
            
            yolo_annotations.append({
                "image_path": yolo_anno["image_path"],
                "class_id": class_id,
                "x_center": yolo_anno["x_center"],
                "y_center": yolo_anno["y_center"],
                "width": yolo_anno["width"],
                "height": yolo_anno["height"],
                "confidence": yolo_anno["confidence"],
                "source": yolo_anno["source"]
            })
        
        logger.info(f"Converted {len(yolo_annotations)} annotations from {vlm_annotation['image_path']}")
        return yolo_annotations

    def create_yolo_dataset(self, all_annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create YOLO dataset structure."""
        print("Creating YOLO dataset structure...")
        
        # Create train/val/test splits
        if self.config["create_splits"]:
            self.create_dataset_splits(all_annotations)
        
        # Create data.yaml for YOLO
        data_yaml = self.create_data_yaml()
        
        # Create image and label symlinks
        self.create_dataset_symlinks(all_annotations)
        
        # Create metadata
        metadata = self.create_dataset_metadata(all_annotations)
        
        return {
            "data_yaml": data_yaml,
            "metadata": metadata,
            "total_images": len(all_annotations),
            "total_annotations": sum(len(ann.get("yolo_annotations", [])) for ann in all_annotations)
        }

    def create_dataset_splits(self, all_annotations: List[Dict[str, Any]]):
        """Create train/val/test splits."""
        np.random.seed(42)  # For reproducibility
        
        # Shuffle annotations
        shuffled = np.random.permutation(all_annotations)
        
        # Calculate split sizes
        total = len(shuffled)
        train_size = int(total * self.config["train_split"])
        val_size = int(total * self.config["val_split"])
        
        # Create splits
        train_annotations = shuffled[:train_size]
        val_annotations = shuffled[train_size:train_size + val_size]
        test_annotations = shuffled[train_size + val_size:]
        
        # Save split information
        splits = {
            "train": [ann["image_path"] for ann in train_annotations],
            "val": [ann["image_path"] for ann in val_annotations],
            "test": [ann["image_path"] for ann in test_annotations]
        }
        
        splits_path = Path(self.config["output_dir"]) / "splits" / "dataset_splits.json"
        with open(splits_path, 'w') as f:
            json.dump(splits, f, indent=2)
        
        # Save annotations for each split
        self.save_split_annotations("train", train_annotations)
        self.save_split_annotations("val", val_annotations)
        self.save_split_annotations("test", test_annotations)

    def save_split_annotations(self, split_name: str, annotations: List[Dict[str, Any]]):
        """Save annotations for a specific split."""
        split_dir = Path(self.config["output_dir"]) / "labels" / split_name
        
        for annotation in annotations:
            image_name = Path(annotation["image_path"]).name
            label_filename = image_name.replace('.jpg', '.txt').replace('.png', '.txt')
            
            label_path = split_dir / label_filename
            
            with open(label_path, 'w') as f:
                for yolo_anno in annotation.get("yolo_annotations", []):
                    line = f"{yolo_anno['class_id']} {yolo_anno['x_center']} {yolo_anno['y_center']} {yolo_anno['width']} {yolo_anno['height']}\n"
                    f.write(line)

    def create_data_yaml(self) -> Dict[str, Any]:
        """Create data.yaml file for YOLO."""
        data_yaml = {
            "path": str(Path(self.config["output_dir"]).parent),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(self.config["class_mapping"]),
            "names": list(self.config["class_mapping"].keys())
        }
        
        data_yaml_path = Path(self.config["output_dir"]) / "data.yaml"
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        return data_yaml

    def create_dataset_symlinks(self, all_annotations: List[Dict[str, Any]]):
        """Create symlinks for images and labels."""
        # Create symlinks for each split
        for split in ["train", "val", "test"]:
            split_dir = Path(self.config["output_dir"]) / split
            
            # Create images symlink directory
            images_dir = Path(self.config["output_dir"]) / "images" / split
            labels_dir = Path(self.config["output_dir"]) / "labels" / split
            
            # Copy images to images directory
            for annotation in all_annotations:
                if annotation.get("split") != split:
                    continue
                
                image_path = Path(annotation["image_path"])
                if image_path.exists():
                    # Copy image to images directory
                    dest_image = images_dir / image_path.name
                    if not dest_image.exists():
                        shutil.copy2(image_path, dest_image)

    def create_dataset_metadata(self, all_annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create metadata for the dataset."""
        metadata = {
            "dataset_name": "german_supermarket_shelves",
            "description": "German supermarket shelf images with SKU annotations",
            "version": "1.0",
            "created_date": time.strftime("%Y-%m-%d"),
            "total_images": len(all_annotations),
            "total_annotations": sum(len(ann.get("yolo_annotations", [])) for ann in all_annotations),
            "class_distribution": {},
            "image_statistics": {
                "mean_width": np.mean([ann.get("image_width", 1920) for ann in all_annotations]),
                "mean_height": np.mean([ann.get("image_height", 1080) for ann in all_annotations]),
                "std_width": np.std([ann.get("image_width", 1920) for ann in all_annotations]),
                "std_height": np.std([ann.get("image_height", 1080) for ann in all_annotations])
            },
            "annotation_sources": {
                "vlm": sum(1 for ann in all_annotations if not ann.get("fallback_used", False)),
                "existing": sum(1 for ann in all_annotations if ann.get("fallback_used", False))
            },
            "quality_metrics": {
                "avg_confidence": np.mean([ann.get("avg_confidence", 0) for ann in all_annotations]),
                "min_confidence": np.min([ann.get("avg_confidence", 0) for ann in all_annotations]),
                "max_confidence": np.max([ann.get("avg_confidence", 0) for ann in all_annotations])
            }
        }
        
        # Calculate class distribution
        class_dist = {}
        for annotation in all_annotations:
            for yolo_anno in annotation.get("yolo_annotations", []):
                class_id = yolo_anno["class_id"]
                class_dist[class_id] = class_dist.get(class_id, 0) + 1
        
        metadata["class_distribution"] = class_dist
        
        # Save metadata
        metadata_path = Path(self.config["output_dir"]) / "metadata" / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata

    def generate_retrieval_data(self, all_annotations: List[Dict[str, Any]]):
        """Generate data for retrieval system."""
        print("Generating retrieval system data...")
        
        # Create crops directory structure
        crops_dir = Path(self.config["output_dir"]) / "crops"
        
        for split in ["train", "val", "test"]:
            split_crops_dir = crops_dir / split
            split_crops_dir.mkdir(parents=True, exist_ok=True)
            
            for annotation in all_annotations:
                if annotation.get("split") != split:
                    continue
                
                image_path = Path(annotation["image_path"])
                if not image_path.exists():
                    continue
                
                # Extract crops based on annotations
                with Image.open(image_path) as img:
                    for yolo_anno in annotation.get("yolo_annotations", []):
                        # Calculate crop coordinates
                        x_center = yolo_anno["x_center"]
                        y_center = yolo_anno["y_center"]
                        width = yolo_anno["width"]
                        height = yolo_anno["height"]
                        
                        image_width = annotation.get("image_width", 1920)
                        image_height = annotation.get("image_height", 1080)
                        
                        # Convert normalized coordinates to pixel coordinates
                        x1 = int((x_center - width / 2) * image_width)
                        y1 = int((y_center - height / 2) * image_height)
                        x2 = int((x_center + width / 2) * image_width)
                        y2 = int((y_center + height / 2) * image_height)
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(image_width, x2)
                        y2 = min(image_height, y2)
                        
                        if x2 > x1 and y2 > y1:
                            # Crop and save
                            crop = img.crop((x1, y1, x2, y2))
                            
                            # Create filename
                            crop_filename = f"{image_path.stem}_crop_{yolo_anno['class_id']}_{yolo_anno['confidence']:.2f}.jpg"
                            crop_path = split_crops_dir / crop_filename
                            
                            crop.save(crop_path)
        
        # Generate retrieval index
        self.generate_retrieval_index(crops_dir)

    def generate_retrieval_index(self, crops_dir: Path):
        """Generate index for retrieval system."""
        index_data = {
            "crops": [],
            "metadata": {
                "total_crops": 0,
                "crops_per_split": {},
                "class_distribution": {}
            }
        }
        
        for split in ["train", "val", "test"]:
            split_crops_dir = crops_dir / split
            if not split_crops_dir.exists():
                continue
            
            crops = list(split_crops_dir.glob("*.jpg"))
            index_data["crops"].extend([
                {
                    "path": str(crop.relative_to(crops_dir)),
                    "split": split,
                    "filename": crop.name
                }
                for crop in crops
            ])
            
            index_data["metadata"]["crops_per_split"][split] = len(crops)
        
        # Calculate class distribution (simplified)
        class_dist = {}
        for crop_info in index_data["crops"]:
            class_id = int(crop_info["filename"].split('_')[-3])
            class_dist[class_id] = class_dist.get(class_id, 0) + 1
        
        index_data["metadata"]["class_distribution"] = class_dist
        index_data["metadata"]["total_crops"] = len(index_data["crops"])
        
        # Save index
        index_path = Path(self.config["output_dir"]) / "retrieval_index.json"
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)

    def run(self):
        """Run the complete annotation conversion pipeline."""
        print("=" * 60)
        print("VLM Annotation Converter")
        print("=" * 60)
        
        # Load VLM annotations
        print("Loading VLM annotations...")
        vlm_annotations = self.load_vlm_annotations()
        
        # Convert to YOLO format
        print("Converting to YOLO format...")
        all_yolo_annotations = []
        
        for vlm_annotation in vlm_annotations:
            yolo_annotations = self.convert_to_yolo_format(vlm_annotation)
            vlm_annotation["yolo_annotations"] = yolo_annotations
            all_yolo_annotations.append(vlm_annotation)
        
        # Create YOLO dataset
        if self.config["generate_yolo_dataset"]:
            dataset_info = self.create_yolo_dataset(all_yolo_annotations)
            print(f"Created YOLO dataset with {dataset_info['total_images']} images and {dataset_info['total_annotations']} annotations")
        
        # Generate retrieval data
        if self.config["generate_retrieval_data"]:
            self.generate_retrieval_data(all_yolo_annotations)
        
        print("\n" + "=" * 60)
        print("Annotation Conversion Complete")
        print("=" * 60)
        print(f"Input annotations: {len(vlm_annotations)}")
        print(f"Total YOLO annotations: {sum(len(ann.get('yolo_annotations', [])) for ann in all_yolo_annotations)}")
        print(f"Output directory: {self.config['output_dir']}")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="VLM Annotation Converter")
    parser.add_argument("--config", type=str, default="configs/converter_config.yaml",
                        help="Path to converter configuration file")
    parser.add_argument("--input", type=str, default="experiments/03_vlm_fewshot/annotations",
                        help="Input directory for VLM annotations")
    parser.add_argument("--output", type=str, default="Dataset/processed_yolo",
                        help="Output directory for YOLO dataset")
    parser.add_argument("--yolo-config", type=str, default="experiments/01_yolo_detection/configs/yolov8_german_supermarket.yaml",
                        help="Path to YOLO configuration file")
    parser.add_argument("--retrieval-config", type=str, default="experiments/02_retrieval_system/config.yaml",
                        help="Path to retrieval system configuration file")

    args = parser.parse_args()

    # Update config
    config = {
        "input_dir": args.input,
        "output_dir": args.output,
        "yolo_config_path": args.yolo_config,
        "retrieval_config_path": args.retrieval_config
    }

    logger.info(f"Starting VLM annotation conversion")
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output directory: {args.output}")

    # Create converter and run
    converter = VLMAnnotationConverter(config)
    converter.run()
if __name__ == "__main__":
    main()
