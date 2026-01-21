import os
import cv2
import numpy as np
import pandas as pd
import yaml
import torch
from ultralytics import YOLO
from sklearn.metrics import classification_report
from tqdm import tqdm

def slice_image(image, tile_size=320, overlap=0.2):
    """
    Slices an image into tiles with overlap.
    Returns a list of (tile, x, y) tuples.
    """
    img_h, img_w = image.shape[:2]
    step = int(tile_size * (1 - overlap))
    
    tiles = []
    # If image is smaller than tile size, pad it? Or just resize?
    # Test images are massive (2048x1000), so usually not an issue.
    
    y_starts = range(0, img_h, step)
    x_starts = range(0, img_w, step)
    
    for y in y_starts:
        for x in x_starts:
            # Adjust if we go out of bounds
            # For the last tile, we might want to shift back to fit exactly
            y_curr = y
            x_curr = x
            
            if y_curr + tile_size > img_h:
                y_curr = max(0, img_h - tile_size)
            
            if x_curr + tile_size > img_w:
                x_curr = max(0, img_w - tile_size)
                
            tile = image[y_curr:y_curr+tile_size, x_curr:x_curr+tile_size]
            tiles.append((tile, x_curr, y_curr))
            
            # Simple deduplication check for last tiles in row/col
            # But the max(0, ...) logic handles the boundary.
            # However, range() might produce duplicates if step is small or image dim matches perfectly.
            # Using specific logic to avoid infinite loops if step is 0 (not possible here).
            
    # Remove exact duplicates if any (e.g. from the max() adjustment at end of row/col)
    unique_tiles = []
    seen_coords = set()
    for t, x, y in tiles:
        if (x, y) not in seen_coords:
            unique_tiles.append((t, x, y))
            seen_coords.add((x, y))
            
    return unique_tiles

def non_max_suppression_fast(boxes, probs, classes, overlapThresh):
    """
    Standard NMS. 
    boxes: [[x1, y1, x2, y2], ...]
    probs: [conf, ...]
    classes: [cls_id, ...]
    """
    if len(boxes) == 0:
        return [], [], []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probs)

    while len(idxs) > 0:
        last = idxs.shape[0] - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        intersection = w * h
        # IoU
        union = area[i] + area[idxs[:last]] - intersection
        
        # Avoid division by zero
        iou = intersection / np.maximum(1e-6, union) 

        idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > overlapThresh)[0])))

    return boxes[pick], probs[pick], classes[pick]

def main():
    # 1. Load Training Configuration
    with open("data.yaml", "r") as f:
        data_config = yaml.safe_load(f)
    class_names = data_config['names']
    print(f"Loaded training classes: {class_names}")

    # 2. Load Ground Truth from Excel
    excel_path = "test/Defects Description.xlsx"
    print(f"Loading ground truth from {excel_path}...")
    
    df_desc = pd.read_excel(excel_path, header=None)
    
    cn_to_en = {
        '换卷冲孔': 'punching_hole',
        '换卷焊缝 焊缝': 'welding_line',
        '换卷月牙弯': 'crescent_gap',
        '斑迹-水斑': 'water_spot',
        '斑迹-油斑': 'oil_spot',
        '斑迹-丝斑': 'silk_spot',
        '异物压入': 'inclusion',
        '压痕': 'rolled_pit',
        '严重折痕': 'crease',
        '腰折': 'waist_folding'
    }

    folder_to_label = {}
    for i in range(10):
        folder_id = str(i + 1)
        excel_idx = 3 + i
        if excel_idx < len(df_desc):
            chinese_label = df_desc.iloc[excel_idx, 1]
            if pd.isna(chinese_label):
                english_label = "Unknown"
            else:
                # Fuzzy match or strip?
                chinese_label = str(chinese_label).strip()
                english_label = cn_to_en.get(chinese_label, f"Unknown_({chinese_label})")
            folder_to_label[folder_id] = english_label
        else:
            folder_to_label[folder_id] = "Unknown"

    print("Folder to Label Mapping:")
    for k, v in folder_to_label.items():
        print(f"  Folder {k}: {v}")

    # 3. Initialize Model
    model_path = "results/yolov8s_defect_detector_aug_finetune/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # 4. Run Tiled Inference
    results_list = []
    test_root = "test"
    
    TILE_SIZE = 320
    OVERLAP = 0.2
    CONF_THRESH = 0.25 
    
    print("Starting tiled inference...")
    
    for folder_id in folder_to_label.keys():
        folder_path = os.path.join(test_root, folder_id)
        if not os.path.exists(folder_path):
            continue
            
        true_label = folder_to_label[folder_id]
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for img_file in tqdm(image_files, desc=f"Folder {folder_id} ({true_label})"):
            img_path = os.path.join(folder_path, img_file)
            im0 = cv2.imread(img_path)
            if im0 is None:
                continue
            
            tile_data = slice_image(im0, TILE_SIZE, OVERLAP)
            
            all_boxes = []
            all_confs = []
            all_cls = []
            
            for tile, x_off, y_off in tile_data:
                results = model.predict(tile, verbose=False, conf=CONF_THRESH)
                res = results[0]
                
                if len(res.boxes) > 0:
                    boxes = res.boxes
                    coords = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clss = boxes.cls.cpu().numpy()
                    
                    coords[:, 0] += x_off
                    coords[:, 1] += y_off
                    coords[:, 2] += x_off
                    coords[:, 3] += y_off
                    
                    all_boxes.append(coords)
                    all_confs.append(confs)
                    all_cls.append(clss)
            
            if len(all_boxes) > 0:
                all_boxes = np.vstack(all_boxes)
                all_confs = np.concatenate(all_confs)
                all_cls = np.concatenate(all_cls)
                
                final_boxes, final_confs, final_cls = non_max_suppression_fast(
                    all_boxes, all_confs, all_cls, overlapThresh=0.45
                )
                
                if len(final_confs) > 0:
                    best_idx = final_confs.argmax()
                    best_cls_id = int(final_cls[best_idx])
                    best_conf = final_confs[best_idx]
                    predicted_label = class_names[best_cls_id]
                else:
                    predicted_label = "No Detection"
                    best_conf = 0.0
            else:
                predicted_label = "No Detection"
                best_conf = 0.0
            
            results_list.append({
                'filename': img_file,
                'folder_id': folder_id,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'confidence': best_conf
            })

    # 5. Save Results
    os.makedirs("predictions", exist_ok=True)
    results_df = pd.DataFrame(results_list)
    output_path = "predictions/results_tiled.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # 6. Analysis
    print("\n--- Analysis Report (Tiled Inference - Corrected Labels) ---")
    
    # Full Confusion Matrix
    print("\nConfusion Matrix (True Label vs Predicted Label):")
    confusion_matrix = pd.crosstab(results_df['true_label'], results_df['predicted_label'])
    print(confusion_matrix)

    # Specific check for Inclusion (Folder 7)
    # The only class where we might expect a match
    if 'inclusion' in results_df['true_label'].unique():
        inclusion_df = results_df[results_df['true_label'] == 'inclusion']
        print("\nAnalysis for 'inclusion' (Folder 7):")
        print(inclusion_df['predicted_label'].value_counts(normalize=True))
    
    print("\nDone.")

if __name__ == "__main__":
    main()
