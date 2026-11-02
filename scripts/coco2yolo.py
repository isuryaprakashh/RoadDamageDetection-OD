import os
import json
import shutil
from pathlib import Path

def convert_coco_bbox_to_yolo(size, box):
    # COCO bbox format is [top_left_x, top_left_y, width, height]
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    # Normalize
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def coco2yolo(json_file, images_dir):
    print(f"Processing {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Categories mapping
    # Ensure category ID starts from 0 for YOLO
    # Some COCO datasets have categories starting from 1
    category_map = {}
    for i, category in enumerate(data['categories']):
        category_map[category['id']] = i
    
    # Save classes.txt for reference
    classes_file = os.path.join(os.path.dirname(json_file), "classes.txt")
    with open(classes_file, 'w', encoding='utf-8') as f:
        for category in data['categories']:
            f.write(f"{category['name']}\n")
    print(f"Found categories: {[c['name'] for c in data['categories']]}")

    # Images mapping
    images = {}
    for img in data['images']:
        images[img['id']] = img

    # Group annotations by image_id
    from collections import defaultdict
    ann_by_image = defaultdict(list)
    for ann in data['annotations']:
        ann_by_image[ann['image_id']].append(ann)

    # Create YOLO labels
    for img_id, img_info in images.items():
        filename = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        # Determine the label file path (same directory as image)
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(images_dir, label_filename)

        with open(label_path, 'w', encoding='utf-8') as f:
            for ann in ann_by_image.get(img_id, []):
                category_id = ann['category_id']
                yolo_class_id = category_map[category_id]
                bbox = ann['bbox'] # [x, y, w, h]
                yolo_bbox = convert_coco_bbox_to_yolo((width, height), bbox)
                
                f.write(f"{yolo_class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")

    print(f"Successfully converted annotations for {len(images)} images in {images_dir}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / "dataset"
    
    train_json = dataset_dir / "train.json"
    train_dir = dataset_dir / "train"
    if train_json.exists() and train_dir.exists():
        coco2yolo(train_json, train_dir)
        
    val_json = dataset_dir / "val.json"
    val_dir = dataset_dir / "val"
    if val_json.exists() and val_dir.exists():
        coco2yolo(val_json, val_dir)
