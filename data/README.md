# Data Directory

This directory contains all data related to the defect detection project.

## Directory Structure

```
data/
├── raw/                    # Raw, unprocessed data
│   ├── images/            # Original images
│   └── labels/            # Original annotation files (YOLO format)
│
├── processed/             # Processed and split data
│   ├── train/            # Training set
│   │   ├── images/       # Training images
│   │   └── labels/       # Training labels
│   ├── val/              # Validation set
│   │   ├── images/       # Validation images
│   │   └── labels/       # Validation labels
│   └── test/             # Test set
│       ├── images/       # Test images
│       └── labels/       # Test labels
│
├── annotations/           # Additional annotation files
│
└── dataset.yaml          # YOLO dataset configuration
```

## Data Format

### Image Format
- Supported formats: JPG, JPEG, PNG
- Recommended size: 640x640 pixels (will be resized automatically)
- RGB color space

### Label Format (YOLO)
Each image should have a corresponding `.txt` file with the same name containing annotations in YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: Integer representing the defect class (0-indexed)
- `x_center`, `y_center`: Center coordinates of the bounding box (normalized 0-1)
- `width`, `height`: Width and height of the bounding box (normalized 0-1)

Example:
```
0 0.5 0.5 0.3 0.4
1 0.7 0.3 0.2 0.2
```

## Dataset Preparation

1. **Place raw data**: Put your images and labels in `data/raw/images/` and `data/raw/labels/`

2. **Run preprocessing**: Use the data preparation script
   ```bash
   python examples/prepare_data.py
   ```

3. **Verify dataset**: The script will validate your data and show statistics

## Dataset Requirements

- Minimum 100 images per class (recommended)
- Balanced class distribution (if possible)
- High-quality images with clear defects
- Accurate annotations
- Consistent image quality and lighting

## Tips for Good Dataset

1. **Diverse examples**: Include various defect types, sizes, and positions
2. **Background variation**: Different backgrounds and lighting conditions
3. **Multiple angles**: Capture defects from different perspectives
4. **Quality control**: Verify annotations are accurate
5. **Augmentation**: Use data augmentation to increase dataset size

## Example Datasets

Public datasets that can be used for defect detection:
- MVTec AD Dataset
- Steel Defect Dataset
- PCB Defect Dataset
- Custom manufacturing datasets

## Notes

- Keep raw data intact as a backup
- The processed data is generated automatically
- Update `configs/config.yaml` if you change class names
- The `dataset.yaml` file is required for training
