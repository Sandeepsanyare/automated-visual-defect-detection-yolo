import optuna
from ultralytics import YOLO
import torch

def objective(trial):
    # --- The search space is focused on high-impact parameters ---
    
    # 1. Phase 1: Only tune the optimizer and its core learning parameters.
    params_phase1 = {
        "optimizer": trial.suggest_categorical("optimizer", ["SGD", "AdamW"]),
        "lr0": trial.suggest_float("lr0", 1e-4, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
    }

    # 2. Phase 2: Tune fine-tuning LR and image size.
    params_phase2 = {
        "imgsz": trial.suggest_categorical("imgsz_phase2", [416, 448, 512, 640]),
        "lr0": trial.suggest_float("lr0_phase2", 5e-5, 5e-3, log=True),
        "lrf": trial.suggest_float("lrf_phase2", 1e-5, 1e-3, log=True),
    }

    # --- Execute Phase 1 Training ---
    try:
        trial_name_phase1 = f"trial_{trial.number}_phase1"
        project_name = "optuna_results_focused"

        model = YOLO("../models/yolov8s.pt")
        model.train(
            data="data.yaml",
            epochs=40,
            patience=10,
            batch=64,
            imgsz=320,
            cos_lr=True,
            project=project_name,
            name=trial_name_phase1,
            workers=2,
            # Use the focused parameter set
            **params_phase1,
            
            # --- FIXED PARAMETERS ---
            # Using strong, fixed augmentations from your original train file
            degrees=10.0, translate=0.10, scale=0.50, shear=2.0, fliplr=0.5,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            mosaic=1.0, mixup=0.10, copy_paste=0.10,
        )
    except Exception as e:
        print(f"Trial {trial.number} Phase 1 failed with error: {e}")
        raise optuna.exceptions.TrialPruned()

    # --- Execute Phase 2 Fine-tuning ---
    try:
        best_model_path = f"{project_name}/{trial_name_phase1}/weights/best.pt"
        trial_name_phase2 = f"trial_{trial.number}_phase2"

        results = YOLO(best_model_path).train(
            data="data.yaml",
            epochs=15,
            patience=5,
            batch=64,
            cos_lr=True,
            # Light, fixed augmentation for fine-tuning
            mosaic=0.0, mixup=0.0, copy_paste=0.0,
            degrees=0.0, translate=0.02, scale=0.20, shear=0.0, fliplr=0.5,
            project=project_name,
            name=trial_name_phase2,
            workers=2,
            **params_phase2
        )
    except Exception as e:
        print(f"Trial {trial.number} Phase 2 failed with error: {e}")
        raise optuna.exceptions.TrialPruned()

    # --- Return Final Metric ---
    metric = results.box.map
    
    torch.cuda.empty_cache()
    return metric


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_min_trials=3),
    )
    import argparse
    parser = argparse.ArgumentParser()
    # CHANGED: The number of trials is now halved to reduce total time.
    parser.add_argument("--trials", type=int, default=25, help="Number of Optuna trials to run.")
    args = parser.parse_args()
    
    study.optimize(objective, n_trials=args.trials)
    
    print("✅ Best Trial:")
    print(f"  Value (mAP50-95): {study.best_value}")
    print("  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")