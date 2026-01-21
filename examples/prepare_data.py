"""
Example: Data Preparation
MSc Thesis - Arden University Berlin

This example demonstrates how to prepare and split your dataset for training.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import DataPreprocessor


def main():
    """Main function for data preparation example."""
    
    print("="*60)
    print("Data Preparation Example")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config_path='configs/config.yaml')
    
    # Example 1: Split dataset
    print("\n1. Splitting dataset into train/val/test sets...")
    print("-" * 60)
    
    # Note: Replace these paths with your actual data paths
    images_dir = 'data/raw/images'
    labels_dir = 'data/raw/labels'
    
    # Check if directories exist
    if Path(images_dir).exists() and Path(labels_dir).exists():
        splits = preprocessor.split_dataset(
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_dir='data/processed',
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            seed=42
        )
        
        print("\nDataset split summary:")
        for split_name, files in splits.items():
            print(f"  {split_name}: {len(files)} images")
    else:
        print(f"Please place your images in {images_dir}")
        print(f"Please place your labels in {labels_dir}")
        print("\nNote: Labels should be in YOLO format (class x_center y_center width height)")
    
    # Example 2: Create dataset YAML
    print("\n2. Creating dataset configuration YAML...")
    print("-" * 60)
    
    dataset_yaml = preprocessor.create_dataset_yaml(
        output_path='data/dataset.yaml',
        train_path='data/processed/train',
        val_path='data/processed/val',
        test_path='data/processed/test'
    )
    
    print(f"Dataset YAML created at: {dataset_yaml}")
    
    # Example 3: Validate dataset
    print("\n3. Validating dataset...")
    print("-" * 60)
    
    train_images = 'data/processed/train/images'
    train_labels = 'data/processed/train/labels'
    
    if Path(train_images).exists() and Path(train_labels).exists():
        stats = preprocessor.validate_dataset(train_images, train_labels)
        preprocessor.print_dataset_stats(stats)
    else:
        print("Training data not found. Please run dataset splitting first.")
    
    print("\n" + "="*60)
    print("Data preparation example completed!")
    print("="*60)


if __name__ == '__main__':
    main()
