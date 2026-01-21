"""
Utility Functions
MSc Thesis - Arden University Berlin

This module provides various utility functions for the defect detection system.
"""

import os
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging


def setup_logging(log_file: str = 'logs/training.log') -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file
        
    Returns:
        Configured logger
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_directory_structure(base_dir: str = '.') -> None:
    """
    Create the required directory structure for the project.
    
    Args:
        base_dir: Base directory for the project
    """
    directories = [
        'data/raw',
        'data/processed/train/images',
        'data/processed/train/labels',
        'data/processed/val/images',
        'data/processed/val/labels',
        'data/processed/test/images',
        'data/processed/test/labels',
        'data/annotations',
        'models/pretrained',
        'models/trained',
        'results/predictions',
        'results/visualizations',
        'results/evaluation',
        'logs',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_dir, directory), exist_ok=True)
    
    print("Directory structure created successfully!")


def get_device() -> str:
    """
    Get available device (CUDA or CPU).
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    return device


def print_system_info() -> None:
    """Print system and environment information."""
    print("\n" + "="*60)
    print("System Information")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*60 + "\n")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def check_dataset_exists(data_yaml: str) -> bool:
    """
    Check if dataset exists and is properly configured.
    
    Args:
        data_yaml: Path to dataset YAML file
        
    Returns:
        True if dataset exists, False otherwise
    """
    if not os.path.exists(data_yaml):
        print(f"Dataset YAML not found: {data_yaml}")
        return False
    
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    required_keys = ['path', 'train', 'val', 'names']
    for key in required_keys:
        if key not in data_config:
            print(f"Missing required key in dataset YAML: {key}")
            return False
    
    # Check if directories exist
    base_path = Path(data_config['path'])
    train_path = base_path / data_config['train']
    val_path = base_path / data_config['val']
    
    if not train_path.exists():
        print(f"Training data not found: {train_path}")
        return False
    
    if not val_path.exists():
        print(f"Validation data not found: {val_path}")
        return False
    
    return True
