
import os
import pandas as pd
import glob

base_dir = "optuna_results_focused"
best_map = -1.0
best_trial = None

# We look for phase2 results as that's the fine-tuned model
for result_file in glob.glob(os.path.join(base_dir, "*_phase2", "results.csv")):
    try:
        df = pd.read_csv(result_file)
        # Check column names, they usually have spaces
        df.columns = [c.strip() for c in df.columns]
        
        if "metrics/mAP50-95(B)" in df.columns:
            max_map = df["metrics/mAP50-95(B)"].max()
            if max_map > best_map:
                best_map = max_map
                best_trial = os.path.dirname(result_file)
    except Exception as e:
        print(f"Error reading {result_file}: {e}")

print(f"Best Trial Directory: {best_trial}")
print(f"Best mAP50-95: {best_map}")
