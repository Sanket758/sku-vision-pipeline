#!/usr/bin/env python3
"""
Convert VLM annotations to clean YOLO format for Labelme.
Generates a simple YOLO format that Labelme can easily import.
"""

import json
from pathlib import Path

def load_yolo_annotations(yolo_path):
    """Load YOLO annotations from JSON file."""
    with open(yolo_path, 'r') as f:
        return json.load(f)

def convert_to_labelme_format(yolo_annotations):
    """Convert YOLO annotations to Labelme format."""
    # Group annotations by image
    images_data = {}
    for annotation in yolo_annotations:
        image_path = annotation['image_path']
        if image_path not in images_data:
            images_data[image_path] = []
        images_data[image_path].append(annotation)
    
    # Generate Labelme format
    labelme_data = {}
    
    for image_path, annotations in images_data.items():
        # Extract image filename
        image_filename = Path(image_path).name
        
        # Create Labelme format annotations
        labelme_annotations = []
        for ann in annotations:
            # Convert normalized coordinates to pixel coordinates
            # (Labelme typically uses pixel coordinates)
            labelme_annotation = {
                "label": f"SKU{ann['class_id']}",
                "coordinates": {
                    "x": ann['x_center'] - ann['width'] / 2,
                    "y": ann['y_center'] - ann['height'] / 2,
                    "width": ann['width'],
                    "height": ann['height']
                },
                "confidence": ann['confidence'],
                "source": ann['source']
            }
            labelme_annotations.append(labelme_annotation)
        
        labelme_data[image_filename] = labelme_annotations
    
    return labelme_data

def save_labelme_format(labelme_data, output_path):
    """Save Labelme format to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(labelme_data, f, indent=2)
    print(f"Labelme format saved to: {output_path}")

def main():
    # Paths
    annotations_dir = Path("experiments/03_vlm_fewshot/annotations")
    yolo_path = annotations_dir / "yolo_annotations.json"
    output_path = annotations_dir / "labelme_annotations.json"
    
    if not yolo_path.exists():
        print(f"Error: YOLO annotations file not found: {yolo_path}")
        return
    
    # Load YOLO annotations
    yolo_annotations = load_yolo_annotations(yolo_path)
    
    if not yolo_annotations:
        print("Error: No annotations found in YOLO file")
        return
    
    print(f"Loaded {len(yolo_annotations)} YOLO annotations")
    print(f"Processing {len(set(a['image_path'] for a in yolo_annotations))} unique images")
    
    # Convert to Labelme format
    labelme_data = convert_to_labelme_format(yolo_annotations)
    
    # Save Labelme format
    save_labelme_format(labelme_data, output_path)
    
    print(f"\n✅ Labelme format conversion complete!")
    print(f"📄 Labelme format saved: {output_path}")
    print(f"📊 Converted {len(yolo_annotations)} annotations for {len(labelme_data)} images")
    
    # Print sample of converted data
    print(f"\n📋 Sample of converted data:")
    for i, (image_name, annotations) in enumerate(list(labelme_data.items())[:3]):
        print(f"\nImage: {image_name}")
        print(f"  Annotations: {len(annotations)}")
        for j, ann in enumerate(annotations[:2]):
            print(f"    {j+1}. {ann['label']} - {ann['coordinates']}")

if __name__ == "__main__":
    main()
