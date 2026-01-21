"""
Data Preprocessing Utilities
MSc Thesis - Arden University Berlin

This module provides utilities for preprocessing and preparing defect detection datasets.
"""

import os
import shutil
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Class for preprocessing defect detection datasets.
    
    Handles data splitting, augmentation, and format conversion for YOLO models.
    """
    
    def __init__(self, config_path: str = 'configs/config.yaml'):
        """
        Initialize the data preprocessor.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.classes = self.config['classes']
        self.data_root = Path(self.config['paths']['data_root'])
        
    def create_dataset_yaml(self, 
                           output_path: str = 'data/dataset.yaml',
                           train_path: str = 'data/processed/train',
                           val_path: str = 'data/processed/val',
                           test_path: Optional[str] = 'data/processed/test') -> str:
        """
        Create a YOLO dataset configuration file.
        
        Args:
            output_path: Path to save the dataset YAML file
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Path to test data (optional)
            
        Returns:
            Path to created YAML file
        """
        dataset_config = {
            'path': str(Path(train_path).parent.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'names': {i: name for i, name in enumerate(self.classes)}
        }
        
        if test_path:
            dataset_config['test'] = 'test/images'
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Dataset YAML created at: {output_path}")
        return output_path
    
    def split_dataset(self,
                     images_dir: str,
                     labels_dir: str,
                     output_dir: str = 'data/processed',
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.2,
                     test_ratio: float = 0.1,
                     seed: int = 42) -> Dict[str, List[str]]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing YOLO format labels
            output_dir: Output directory for split data
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with file lists for each split
        """
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        output_dir = Path(output_dir)
        
        # Get all image files
        image_files = sorted(list(images_dir.glob('*.jpg')) + 
                           list(images_dir.glob('*.jpeg')) + 
                           list(images_dir.glob('*.png')))
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {images_dir}")
        
        print(f"Found {len(image_files)} images")
        
        # Verify ratios sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # Split data
        train_files, temp_files = train_test_split(
            image_files, train_size=train_ratio, random_state=seed
        )
        
        val_size = val_ratio / (val_ratio + test_ratio)
        val_files, test_files = train_test_split(
            temp_files, train_size=val_size, random_state=seed
        )
        
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        # Copy files to respective directories
        for split_name, files in splits.items():
            print(f"Processing {split_name} split: {len(files)} images")
            
            split_img_dir = output_dir / split_name / 'images'
            split_lbl_dir = output_dir / split_name / 'labels'
            
            split_img_dir.mkdir(parents=True, exist_ok=True)
            split_lbl_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in tqdm(files, desc=f"Copying {split_name} files"):
                # Copy image
                shutil.copy2(img_path, split_img_dir / img_path.name)
                
                # Copy corresponding label file
                label_path = labels_dir / (img_path.stem + '.txt')
                if label_path.exists():
                    shutil.copy2(label_path, split_lbl_dir / label_path.name)
                else:
                    # Create empty label file if not exists
                    (split_lbl_dir / label_path.name).touch()
        
        print("Dataset split completed successfully!")
        return splits
    
    def get_augmentation_pipeline(self) -> A.Compose:
        """
        Create an augmentation pipeline using albumentations.
        
        Returns:
            Composed augmentation pipeline
        """
        aug_config = self.config['augmentation']
        
        transforms = []
        
        if aug_config.get('horizontal_flip', False):
            transforms.append(A.HorizontalFlip(p=0.5))
        
        if aug_config.get('vertical_flip', False):
            transforms.append(A.VerticalFlip(p=0.5))
        
        if aug_config.get('rotation', 0) > 0:
            transforms.append(A.Rotate(limit=aug_config['rotation'], p=0.5))
        
        if aug_config.get('brightness', 0) > 0:
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=aug_config['brightness'],
                contrast_limit=aug_config.get('contrast', 0),
                p=0.5
            ))
        
        transforms.extend([
            A.Blur(blur_limit=3, p=0.1),
            A.GaussNoise(p=0.1),
        ])
        
        # Add bbox transformation
        transform = A.Compose(
            transforms,
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )
        
        return transform
    
    def validate_dataset(self, 
                        images_dir: str,
                        labels_dir: str) -> Dict:
        """
        Validate dataset integrity and statistics.
        
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing labels
            
        Returns:
            Dictionary with validation statistics
        """
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        
        image_files = list(images_dir.glob('*.jpg')) + \
                     list(images_dir.glob('*.jpeg')) + \
                     list(images_dir.glob('*.png'))
        
        stats = {
            'total_images': len(image_files),
            'images_with_labels': 0,
            'images_without_labels': 0,
            'total_objects': 0,
            'class_distribution': {cls: 0 for cls in self.classes},
            'image_sizes': []
        }
        
        for img_path in tqdm(image_files, desc="Validating dataset"):
            # Check for corresponding label
            label_path = labels_dir / (img_path.stem + '.txt')
            
            # Get image size
            img = cv2.imread(str(img_path))
            if img is not None:
                stats['image_sizes'].append(img.shape[:2])
            
            if label_path.exists():
                stats['images_with_labels'] += 1
                
                # Count objects per class
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    stats['total_objects'] += len(lines)
                    
                    for line in lines:
                        class_id = int(line.split()[0])
                        if 0 <= class_id < len(self.classes):
                            stats['class_distribution'][self.classes[class_id]] += 1
            else:
                stats['images_without_labels'] += 1
        
        # Calculate average image size
        if stats['image_sizes']:
            avg_height = np.mean([s[0] for s in stats['image_sizes']])
            avg_width = np.mean([s[1] for s in stats['image_sizes']])
            stats['avg_image_size'] = (int(avg_height), int(avg_width))
        
        return stats
    
    def print_dataset_stats(self, stats: Dict) -> None:
        """
        Print dataset statistics in a readable format.
        
        Args:
            stats: Statistics dictionary from validate_dataset
        """
        print("\n" + "="*50)
        print("Dataset Statistics")
        print("="*50)
        print(f"Total images: {stats['total_images']}")
        print(f"Images with labels: {stats['images_with_labels']}")
        print(f"Images without labels: {stats['images_without_labels']}")
        print(f"Total defect instances: {stats['total_objects']}")
        
        if 'avg_image_size' in stats:
            print(f"Average image size: {stats['avg_image_size']}")
        
        print("\nClass Distribution:")
        for cls, count in stats['class_distribution'].items():
            percentage = (count / stats['total_objects'] * 100) if stats['total_objects'] > 0 else 0
            print(f"  {cls}: {count} ({percentage:.1f}%)")
        print("="*50 + "\n")
