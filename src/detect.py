import os
import pandas as pd
import yaml
from ultralytics import YOLO
from sklearn.metrics import classification_report
from tqdm import tqdm

def main():
    # 1. Load Training Configuration
    with open("data.yaml", "r") as f:
        data_config = yaml.safe_load(f)
    class_names = data_config['names']
    print(f"Loaded training classes: {class_names}")

    # 2. Load Ground Truth from Excel
    excel_path = "test/Defects Description.xlsx"
    print(f"Loading ground truth from {excel_path}...")
    
    # Read rows 3-12 (indices 2-11 in 0-indexed pandas, but header is row 2 so skip rows?)
    # header=None reads all. 
    # Row 3 in Excel is index 2 in df if header=None.
    # We want rows 3-12 corresponding to folders 1-10.
    
    df_desc = pd.read_excel(excel_path, header=None)
    
    # Mapping based on observation:
    # Row 3 (Index 2): Folder 1
    # ...
    # Row 12 (Index 11): Folder 10
    
    folder_mapping = {}
    
    # Chinese to English mapping based on order in data.yaml and excel file
    # data.yaml: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    # Excel: 3:裂纹, 4:杂质, 5:斑块, 6:麻点面, 7:压痕, 8:刮伤
    
    cn_to_en = {
        '裂纹': 'crazing',
        '杂质': 'inclusion',
        '斑块': 'patches',
        '麻点面': 'pitted_surface',
        '压痕': 'rolled-in_scale',
        '刮伤': 'scratches'
    }

    # Extract labels for folders 1-10
    folder_to_label = {}
    
    for i in range(10):
        folder_id = str(i + 1)
        row_idx = 3 + i # Excel row 3 starts at index 3 in our manual logic? 
        # API output showed:
        # 3      裂纹 (Folder 1)
        # However, earlier `df.iloc[3:13, 1]` showed the labels.
        # Let's rely on that index.
        
        # Row index in the full dataframe (header=None)
        # Excel Row 4 is index 3. Wait.
        # User prompt showed:
        # 3      裂纹
        # This implies df index 3 matches Folder 1? 
        # Let's verify: row 3 in Excel is usually where data starts if header is 2 rows.
        # The tool output showed:
        # 3      裂纹
        # 4      杂质
        # ...
        # 8      刮伤
        # 9      NaN
        
        # So Index 3 corresponds to Folder 1.
        
        excel_idx = 3 + i
        if excel_idx < len(df_desc):
            chinese_label = df_desc.iloc[excel_idx, 1]
            if pd.isna(chinese_label):
                english_label = "Unknown"
            else:
                english_label = cn_to_en.get(chinese_label, "Unknown")
            
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

    # 4. Run Inference
    results_list = []
    test_root = "test"
    
    print("Starting inference...")
    
    # Folders 1 to 10
    for folder_id in folder_to_label.keys():
        folder_path = os.path.join(test_root, folder_id)
        if not os.path.exists(folder_path):
            continue
            
        true_label = folder_to_label[folder_id]
        
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for img_file in tqdm(image_files, desc=f"Folder {folder_id} ({true_label})"):
            img_path = os.path.join(folder_path, img_file)
            
            # Run prediction
            results = model.predict(img_path, verbose=False)
            result = results[0]
            
            if len(result.boxes) > 0:
                # Get the class with highest confidence
                # Sorting by confidence just in case, though usually sorted
                boxes = result.boxes
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                
                best_idx = confidences.argmax()
                best_cls_id = int(classes[best_idx])
                best_conf = confidences[best_idx]
                predicted_label = class_names[best_cls_id]
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
    output_path = "predictions/results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # 6. Analysis
    print("\n--- Analysis Report ---")
    
    # Filter for known classes for classification report
    known_df = results_df[results_df['true_label'] != "Unknown"]
    if not known_df.empty:
        print("\nClassification Report (Known Classes):")
        print(classification_report(known_df['true_label'], known_df['predicted_label'], zero_division=0))
    else:
        print("\nNo known classes found to analyze.")

    # Analysis of Unknown folders
    unknown_df = results_df[results_df['true_label'] == "Unknown"]
    if not unknown_df.empty:
        print("\nAnalysis of Unknown/New Folders (7-10):")
        print(unknown_df.groupby(['folder_id', 'predicted_label']).size().unstack(fill_value=0))
    
    print("\nDone.")

if __name__ == "__main__":
    main()
