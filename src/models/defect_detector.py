"""
YOLO Defect Detection Model
MSc Thesis - Arden University Berlin

This module provides a wrapper around YOLO models for industrial defect detection.
"""

from ultralytics import YOLO
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np


class DefectDetectionModel:
    """
    Wrapper class for YOLO-based defect detection model.
    
    This class handles model initialization, training, inference, and model management
    for automated visual defect detection in manufacturing.
    """
    
    def __init__(self, config_path: str = 'configs/config.yaml'):
        """
        Initialize the defect detection model.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.device = self._get_device()
        self.class_names = self.config['classes']
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _get_device(self) -> str:
        """Get the device for model training/inference."""
        device = self.config['training']['device']
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, using CPU instead")
            device = 'cpu'
        return device
    
    def load_pretrained(self, model_name: Optional[str] = None) -> None:
        """
        Load a pretrained YOLO model.
        
        Args:
            model_name: Name of the YOLO model (e.g., 'yolov8n', 'yolov8s')
        """
        if model_name is None:
            model_name = self.config['model']['name']
        
        print(f"Loading pretrained model: {model_name}")
        self.model = YOLO(f"{model_name}.pt")
        print(f"Model loaded successfully on device: {self.device}")
    
    def load_custom(self, model_path: str) -> None:
        """
        Load a custom trained model.
        
        Args:
            model_path: Path to the custom model weights
        """
        print(f"Loading custom model from: {model_path}")
        self.model = YOLO(model_path)
        print(f"Model loaded successfully on device: {self.device}")
    
    def train(self, 
              data_yaml: str,
              epochs: Optional[int] = None,
              batch_size: Optional[int] = None,
              imgsz: Optional[int] = None,
              save_dir: str = 'models/trained') -> Dict:
        """
        Train the model on defect detection dataset.
        
        Args:
            data_yaml: Path to dataset configuration YAML file
            epochs: Number of training epochs
            batch_size: Batch size for training
            imgsz: Input image size
            save_dir: Directory to save trained models
            
        Returns:
            Dictionary containing training results
        """
        if self.model is None:
            self.load_pretrained()
        
        # Use config values if not provided
        epochs = epochs or self.config['training']['epochs']
        batch_size = batch_size or self.config['training']['batch_size']
        imgsz = imgsz or self.config['model']['input_size']
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Batch size: {batch_size}, Image size: {imgsz}")
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=self.device,
            patience=self.config['training']['patience'],
            save_period=self.config['training']['save_period'],
            workers=self.config['training']['workers'],
            optimizer=self.config['training']['optimizer'],
            lr0=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            project=save_dir,
            name='defect_detection',
            exist_ok=True
        )
        
        print("Training completed successfully!")
        return results
    
    def predict(self, 
                source: Union[str, np.ndarray],
                conf: Optional[float] = None,
                iou: Optional[float] = None,
                save: bool = False,
                save_txt: bool = False,
                save_dir: str = 'results/predictions') -> List:
        """
        Run inference on images/videos to detect defects.
        
        Args:
            source: Path to image/video or numpy array
            conf: Confidence threshold
            iou: IoU threshold for NMS
            save: Save prediction visualizations
            save_txt: Save predictions to text files
            save_dir: Directory to save results
            
        Returns:
            List of detection results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_pretrained() or load_custom() first.")
        
        # Use config values if not provided
        conf = conf or self.config['model']['confidence_threshold']
        iou = iou or self.config['model']['iou_threshold']
        
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            device=self.device,
            save=save,
            save_txt=save_txt,
            project=save_dir,
            name='detect',
            exist_ok=True
        )
        
        return results
    
    def validate(self, data_yaml: str) -> Dict:
        """
        Validate the model on a validation/test dataset.
        
        Args:
            data_yaml: Path to dataset configuration YAML file
            
        Returns:
            Dictionary containing validation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_pretrained() or load_custom() first.")
        
        print("Running validation...")
        results = self.model.val(
            data=data_yaml,
            device=self.device,
            project='results/validation',
            name='val',
            exist_ok=True
        )
        
        print("Validation completed!")
        return results
    
    def export(self, format: str = 'onnx', save_dir: str = 'models/exported') -> str:
        """
        Export the model to different formats for deployment.
        
        Args:
            format: Export format (onnx, torchscript, coreml, etc.)
            save_dir: Directory to save exported model
            
        Returns:
            Path to exported model
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_pretrained() or load_custom() first.")
        
        print(f"Exporting model to {format} format...")
        exported_path = self.model.export(format=format)
        print(f"Model exported successfully to: {exported_path}")
        return exported_path
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
        info = {
            "model_type": self.config['model']['name'],
            "num_classes": len(self.class_names),
            "classes": self.class_names,
            "device": self.device,
            "input_size": self.config['model']['input_size'],
            "confidence_threshold": self.config['model']['confidence_threshold'],
            "iou_threshold": self.config['model']['iou_threshold']
        }
        
        return info
