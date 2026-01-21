"""
Industrial Robustness & Sensitivity Testing Suite

Evaluates the robustness of a YOLOv8 model against synthetic industrial noise:
- High-Frequency Noise (oil mist, scale dust)
- Motion Blur (conveyor vibration)
- Lighting Variability (sensor aging, ambient light shifts)

Uses Trial 20 Phase 2 best weights from Optuna optimization.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_WEIGHTS = PROJECT_ROOT / "optuna_results_focused" / "trial_20_phase2" / "weights" / "best.pt"
VALIDATION_IMAGES = PROJECT_ROOT / "data" / "images" / "val"
VALIDATION_LABELS = PROJECT_ROOT / "data" / "labels" / "val"
DATA_YAML = PROJECT_ROOT / "data.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"

CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# Define Albumentations noise pipelines
NOISE_PIPELINES: Dict[str, A.Compose] = {
    "high_frequency_noise": A.Compose([
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.8),
    ]),
    "motion_blur": A.Compose([
        A.MotionBlur(blur_limit=(5, 15), p=1.0),
    ]),
    "lighting_variability": A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        A.Solarize(threshold=128, p=0.5),
    ]),
}


def create_corrupted_dataset(
    source_images_dir: Path,
    source_labels_dir: Path,
    output_dir: Path,
    transform: A.Compose,
    noise_name: str
) -> Tuple[Path, Path]:
    """
    Create a corrupted version of the validation dataset.
    
    Args:
        source_images_dir: Path to original validation images
        source_labels_dir: Path to original validation labels
        output_dir: Base output directory
        transform: Albumentations transform to apply
        noise_name: Name of the noise type for folder naming
        
    Returns:
        Tuple of (corrupted_images_path, corrupted_labels_path)
    """
    corrupted_images_dir = output_dir / "images" / noise_name
    corrupted_labels_dir = output_dir / "labels" / noise_name
    
    corrupted_images_dir.mkdir(parents=True, exist_ok=True)
    corrupted_labels_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(source_images_dir.glob("*.jpg")) + list(source_images_dir.glob("*.png"))
    
    for img_path in tqdm(image_files, desc=f"Creating {noise_name} dataset"):
        # Read and transform image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transformed = transform(image=image)
        corrupted_image = transformed["image"]
        
        # Convert back to BGR and save
        corrupted_image = cv2.cvtColor(corrupted_image, cv2.COLOR_RGB2BGR)
        output_img_path = corrupted_images_dir / img_path.name
        cv2.imwrite(str(output_img_path), corrupted_image)
        
        # Copy corresponding label file
        label_name = img_path.stem + ".txt"
        source_label = source_labels_dir / label_name
        if source_label.exists():
            shutil.copy(source_label, corrupted_labels_dir / label_name)
    
    return corrupted_images_dir, corrupted_labels_dir


def create_temp_data_yaml(images_dir: Path, labels_dir: Path, temp_dir: Path, noise_name: str) -> Path:
    """Create a temporary data.yaml for validation with corrupted data."""
    yaml_content = f"""train: {VALIDATION_IMAGES}
val: {images_dir}

nc: 6
names: {CLASS_NAMES}
"""
    yaml_path = temp_dir / f"data_{noise_name}.yaml"
    yaml_path.write_text(yaml_content)
    return yaml_path


def run_validation(model: YOLO, data_yaml: Path, noise_name: str) -> Dict:
    """
    Run validation and extract metrics.
    
    Returns:
        Dictionary with mAP@50, mAP@50-95, and per-class AP values
    """
    print(f"\n{'='*60}")
    print(f"Running validation for: {noise_name}")
    print(f"{'='*60}")
    
    results = model.val(data=str(data_yaml), verbose=False)
    
    metrics = {
        "noise_type": noise_name,
        "mAP@50": results.box.map50,
        "mAP@50-95": results.box.map,
    }
    
    # Per-class AP50 values
    per_class_ap50 = results.box.ap50
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(per_class_ap50):
            metrics[f"AP50_{class_name}"] = per_class_ap50[i]
        else:
            metrics[f"AP50_{class_name}"] = 0.0
    
    return metrics


def generate_comparison_dataframe(all_metrics: List[Dict]) -> pd.DataFrame:
    """Generate a pandas DataFrame comparing metrics across noise types."""
    df = pd.DataFrame(all_metrics)
    df = df.set_index("noise_type")
    return df


def plot_sensitivity_chart(df: pd.DataFrame, output_path: Path):
    """
    Plot a bar chart showing sensitivity of each class to noise types.
    """
    # Extract per-class columns
    class_columns = [col for col in df.columns if col.startswith("AP50_")]
    
    # Prepare data for plotting
    class_names_short = [col.replace("AP50_", "") for col in class_columns]
    noise_types = df.index.tolist()
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(class_names_short))
    width = 0.2
    multiplier = 0
    
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']  # green for clean, others for noise
    
    for i, noise_type in enumerate(noise_types):
        offset = width * multiplier
        values = df.loc[noise_type, class_columns].values
        bars = ax.bar(x + offset, values, width, label=noise_type.replace("_", " ").title(), 
                     color=colors[i % len(colors)], alpha=0.85)
        multiplier += 1
    
    # Formatting
    ax.set_xlabel('Defect Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('AP@50', fontsize=12, fontweight='bold')
    ax.set_title('Model Sensitivity to Industrial Noise Types\n(Per-Class AP@50 Comparison)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([name.replace("_", " ").title() for name in class_names_short], 
                       rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Sensitivity chart saved to: {output_path}")


def main():
    """Main function to run the robustness testing suite."""
    print("=" * 70)
    print("🔧 Industrial Robustness & Sensitivity Testing Suite")
    print("=" * 70)
    
    # Verify paths exist
    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(f"Model weights not found: {MODEL_WEIGHTS}")
    if not VALIDATION_IMAGES.exists():
        raise FileNotFoundError(f"Validation images not found: {VALIDATION_IMAGES}")
    
    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Load model
    print(f"\n📦 Loading model from: {MODEL_WEIGHTS}")
    model = YOLO(str(MODEL_WEIGHTS))
    
    all_metrics = []
    
    # Create temporary directory for corrupted datasets
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 1. Run baseline validation (clean data)
        print("\n" + "=" * 60)
        print("📊 Running BASELINE validation (clean data)")
        print("=" * 60)
        baseline_metrics = run_validation(model, DATA_YAML, "clean_baseline")
        all_metrics.append(baseline_metrics)
        print(f"   mAP@50: {baseline_metrics['mAP@50']:.4f}")
        print(f"   mAP@50-95: {baseline_metrics['mAP@50-95']:.4f}")
        
        # 2. Run validation for each noise type
        for noise_name, transform in NOISE_PIPELINES.items():
            print(f"\n{'='*60}")
            print(f"🔄 Processing noise type: {noise_name}")
            print(f"{'='*60}")
            
            # Create corrupted dataset
            corrupted_images, corrupted_labels = create_corrupted_dataset(
                VALIDATION_IMAGES,
                VALIDATION_LABELS,
                temp_path,
                transform,
                noise_name
            )
            
            # Create temporary data.yaml
            temp_yaml = create_temp_data_yaml(
                corrupted_images, 
                corrupted_labels, 
                temp_path, 
                noise_name
            )
            
            # Run validation
            metrics = run_validation(model, temp_yaml, noise_name)
            all_metrics.append(metrics)
            
            print(f"   mAP@50: {metrics['mAP@50']:.4f}")
            print(f"   mAP@50-95: {metrics['mAP@50-95']:.4f}")
    
    # Generate comparison DataFrame
    print("\n" + "=" * 70)
    print("📈 RESULTS SUMMARY")
    print("=" * 70)
    
    df = generate_comparison_dataframe(all_metrics)
    
    # Display DataFrame
    print("\n📊 Metrics Comparison (mAP@50 and mAP@50-95):")
    print("-" * 50)
    summary_df = df[["mAP@50", "mAP@50-95"]].copy()
    
    # Calculate degradation from baseline
    baseline_map50 = df.loc["clean_baseline", "mAP@50"]
    baseline_map = df.loc["clean_baseline", "mAP@50-95"]
    
    summary_df["Δ mAP@50"] = df["mAP@50"] - baseline_map50
    summary_df["Δ mAP@50-95"] = df["mAP@50-95"] - baseline_map
    
    print(summary_df.to_string())
    
    # Save full DataFrame to CSV
    csv_path = RESULTS_DIR / "robustness_comparison.csv"
    df.to_csv(csv_path)
    print(f"\n✅ Full metrics saved to: {csv_path}")
    
    # Generate sensitivity bar chart
    chart_path = RESULTS_DIR / "robustness_sensitivity_chart.png"
    plot_sensitivity_chart(df, chart_path)
    
    # Print per-class sensitivity analysis
    print("\n📊 Per-Class Sensitivity Analysis:")
    print("-" * 50)
    class_columns = [col for col in df.columns if col.startswith("AP50_")]
    class_df = df[class_columns].copy()
    class_df.columns = [col.replace("AP50_", "") for col in class_df.columns]
    
    # Calculate average degradation per class
    print("\nAverage AP@50 degradation by class (vs baseline):")
    for class_name in class_df.columns:
        baseline_val = class_df.loc["clean_baseline", class_name]
        avg_degradation = class_df.loc[class_df.index != "clean_baseline", class_name].mean() - baseline_val
        print(f"   {class_name:20s}: {avg_degradation:+.4f}")
    
    print("\n" + "=" * 70)
    print("✅ Robustness testing complete!")
    print("=" * 70)
    
    return df


if __name__ == "__main__":
    main()
