#!/usr/bin/env python3
"""
Simple visual verification tool for VLM annotation pipeline results.
Generates an HTML page that displays images with bounding boxes from YOLO annotations.
"""

import json
from pathlib import Path
import base64
from PIL import Image
import io

def load_yolo_annotations(yolo_path):
    """Load YOLO annotations from JSON file."""
    with open(yolo_path, 'r') as f:
        return json.load(f)

def load_image_as_base64(image_path):
    """Load image and convert to base64 for embedding in HTML."""
    try:
        with Image.open(image_path) as img:
            # Resize to a reasonable size for display
            img.thumbnail((800, 600))
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def generate_bbox_coordinates(x_center, y_center, width, height, img_width, img_height):
    """Convert YOLO format to bounding box coordinates."""
    # YOLO format: x_center, y_center, width, height (normalized 0-1)
    # Convert to pixel coordinates
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    
    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width - 1, x2)
    y2 = min(img_height - 1, y2)
    
    return x1, y1, x2, y2

def generate_simple_html(annotations, output_path):
    """Generate simple HTML page with image visualizations using Tailwind CSS."""
    
    # Group annotations by image
    images_data = {}
    for annotation in annotations:
        image_path = annotation['image_path']
        if image_path not in images_data:
            images_data[image_path] = {
                'annotations': [],
                'image_base64': None,
                'width': 0,
                'height': 0
            }
        images_data[image_path]['annotations'].append(annotation)
    
    # Load images
    for image_path, data in images_data.items():
        data['image_base64'] = load_image_as_base64(image_path)
        try:
            with Image.open(image_path) as img:
                data['width'], data['height'] = img.size
        except Exception as e:
            print(f"Error getting image dimensions for {image_path}: {e}")
    
    # Generate HTML with Tailwind CSS - using string concatenation to avoid f-string issues
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VLM Annotation Pipeline - Visual Verification</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* Custom animations and transitions */
        .image-container {
            margin-bottom: 2rem;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            background-color: #f9f9f9;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .image-header {
            background-color: #4CAF50;
            color: white;
            padding: 1rem 1.5rem;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .image-content {
            padding: 1.5rem;
            text-align: center;
        }
        .image-wrapper {
            position: relative;
            display: block;
            margin: 1rem auto;
            text-align: center;
        }
        .image-wrapper img {
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 4px;
            display: block;
            margin: 0 auto;
        }
        .bbox {
            position: absolute;
            border: 2px solid #ff4444;
            background-color: rgba(255, 68, 68, 0.2);
            pointer-events: none;
            box-sizing: border-box;
        }
        .bbox-label {
            position: absolute;
            top: -25px;
            left: 0;
            background-color: #ff4444;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
            white-space: nowrap;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .stat-item {
            background-color: #e8f5e8;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2e7d32;
        }
        .stat-label {
            font-size: 0.875rem;
            color: #666;
        }
        .legend {
            margin: 2rem 0;
            padding: 1.5rem;
            background-color: #f0f0f0;
            border-radius: 8px;
        }
        .legend h3 {
            margin-top: 0;
            color: #333;
        }
        .legend-item {
            display: inline-block;
            margin: 0.5rem 1rem;
            padding: 0.5rem 1rem;
            background-color: #fff;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        .legend-color {
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 8px;
            border-radius: 3px;
        }
        @media (max-width: 768px) {
            .container {
                margin: 1rem;
                padding: 1rem;
            }
            .stats {
                grid-template-columns: 1fr;
                gap: 0.5rem;
            }
        }
    </style>
</head>
<body class="bg-gray-100 font-sans">
    <div class="container mx-auto p-6 max-w-6xl">
        <h1 class="text-3xl font-bold text-center mb-8 text-gray-800">
            🎯 VLM Annotation Pipeline - Visual Verification
        </h1>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-value">""" + str(len(images_data)) + """
                </div>
                <div class="stat-label">Images Processed</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">""" + str(len(annotations)) + """
                </div>
                <div class="stat-label">Total Annotations</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">""" + str(len(annotations) // len(images_data)) + """
                </div>
                <div class="stat-label">Avg Annotations per Image</div>
            </div>
        </div>
        
        <div class="legend">
            <h3>Legend</h3>
            <div class="flex flex-wrap">
                <div class="legend-item">
                    <div class="legend-color bg-red-500"></div>
                    <span>Bounding Box</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color bg-green-500"></div>
                    <span>SKU Class</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color bg-blue-500"></div>
                    <span>Confidence: """ + (str(annotations[0]['confidence']) if annotations else '0') + """
                </div>
            </div>
        </div>
        
""")
    
    # Add image containers
    for image_path, data in images_data.items():
        if data['image_base64']:
            html_parts.append(f'''
        <div class="image-container">
            <div class="image-header">
                <span>📷 {Path(image_path).name}</span>
                <span class="text-sm bg-green-700 px-2 py-1 rounded-full">{len(data['annotations'])} annotations</span>
            </div>
            <div class="image-content">
                <div class="image-wrapper">
                    <img src="data:image/jpeg;base64,{data['image_base64']}" alt="{Path(image_path).name}">
                </div>
            </div>
        </div>
        
        ''' + ''.join(f'''
            <div class="bbox" style="left: {x1}px; top: {y1}px; width: {x2 - x1}px; height: {y2 - y1}px;"><div class="bbox-label">SKU{data.get('class_id', 'N/A')}</div></div>
        ''' for annotation in data['annotations'] 
           for (x1, y1, x2, y2) in [generate_bbox_coordinates(
               annotation['x_center'], annotation['y_center'], 
               annotation['width'], annotation['height'], 
               data['width'], data['height']
           )]))
    
    # Add image containers for failed loads
    for image_path, data in images_data.items():
        if not data['image_base64']:
            html_parts.append(f'''
        <div class="image-container">
            <div class="image-header">
                <span>📷 {Path(image_path).name}</span>
                <span class="text-sm bg-red-500 px-2 py-1 rounded-full">{len(data['annotations'])} annotations</span>
            </div>
            <div class="image-content">
                <p class="text-red-500">❌ Could not load image: {image_path}</p>
            </div>
        </div>
        ''')
    
    # Close HTML
    html_parts.append("""
    </div>
</body>
</html>
""")
    
    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(''.join(html_parts))
    
    print(f"HTML visualization generated: {output_path}")

def main():
    # Paths
    annotations_dir = Path("experiments/03_vlm_fewshot/annotations")
    yolo_path = annotations_dir / "yolo_annotations.json"
    output_path = annotations_dir / "visual_verification.html"
    
    if not yolo_path.exists():
        print(f"Error: YOLO annotations file not found: {yolo_path}")
        return
    
    # Load annotations
    annotations = load_yolo_annotations(yolo_path)
    
    if not annotations:
        print("Error: No annotations found in YOLO file")
        return
    
    print(f"Loaded {len(annotations)} annotations")
    print(f"Processing {len(set(a['image_path'] for a in annotations))} unique images")
    
    # Generate HTML visualization
    generate_simple_html(annotations, output_path)
    
    print(f"\n✅ Visual verification complete!")
    print(f"📄 HTML file generated: {output_path}")
    print(f"🌐 Open this file in your web browser to view the annotations visually")

if __name__ == "__main__":
    main()
    
    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML visualization generated: {output_path}")

def main():
    # Paths
    annotations_dir = Path("experiments/03_vlm_fewshot/annotations")
    yolo_path = annotations_dir / "yolo_annotations.json"
    output_path = annotations_dir / "visual_verification.html"
    
    if not yolo_path.exists():
        print(f"Error: YOLO annotations file not found: {yolo_path}")
        return
    
    # Load annotations
    annotations = load_yolo_annotations(yolo_path)
    
    if not annotations:
        print("Error: No annotations found in YOLO file")
        return
    
    print(f"Loaded {len(annotations)} annotations")
    print(f"Processing {len(set(a['image_path'] for a in annotations))} unique images")
    
    # Generate HTML visualization
    generate_simple_html(annotations, output_path)
    
    print(f"\n✅ Visual verification complete!")
    print(f"📄 HTML file generated: {output_path}")
    print(f"🌐 Open this file in your web browser to view the annotations visually")

if __name__ == "__main__":
    main()

