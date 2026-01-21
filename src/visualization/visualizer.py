"""
Visualization Utilities
MSc Thesis - Arden University Berlin

This module provides visualization tools for defect detection results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import yaml


class DefectVisualizer:
    """
    Visualizer class for defect detection results.
    
    Handles visualization of detection results with bounding boxes and labels.
    """
    
    def __init__(self, config_path: str = 'configs/config.yaml'):
        """
        Initialize visualizer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.class_names = self.config['classes']
        self.colors = self._generate_colors(len(self.class_names))
        
        # Visualization settings
        viz_config = self.config.get('visualization', {})
        self.show_labels = viz_config.get('show_labels', True)
        self.show_confidence = viz_config.get('show_confidence', True)
        self.bbox_thickness = viz_config.get('bbox_thickness', 2)
        self.font_scale = viz_config.get('font_scale', 0.5)
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """
        Generate distinct colors for each class.
        
        Args:
            num_classes: Number of classes
            
        Returns:
            List of BGR color tuples
        """
        np.random.seed(42)
        colors = []
        for i in range(num_classes):
            # Generate colors in HSV and convert to BGR
            hue = int(180 * i / num_classes)
            color = cv2.cvtColor(
                np.uint8([[[hue, 255, 255]]]), 
                cv2.COLOR_HSV2BGR
            )[0][0]
            colors.append(tuple(map(int, color)))
        return colors
    
    def draw_detections(self,
                       image: np.ndarray,
                       boxes: np.ndarray,
                       classes: np.ndarray,
                       confidences: np.ndarray) -> np.ndarray:
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: Input image (BGR format)
            boxes: Bounding boxes in xyxy format
            classes: Class IDs
            confidences: Confidence scores
            
        Returns:
            Image with drawn detections
        """
        img_copy = image.copy()
        
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(cls)
            
            # Get color for this class
            color = self.colors[cls_id % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, self.bbox_thickness)
            
            # Prepare label
            if self.show_labels:
                label = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Class {cls_id}"
                if self.show_confidence:
                    label += f" {conf:.2f}"
                
                # Draw label background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
                )
                
                cv2.rectangle(img_copy, 
                            (x1, y1 - text_height - baseline - 5),
                            (x1 + text_width, y1),
                            color, -1)
                
                # Draw label text
                cv2.putText(img_copy, label,
                          (x1, y1 - baseline - 2),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          self.font_scale,
                          (255, 255, 255),
                          1,
                          cv2.LINE_AA)
        
        return img_copy
    
    def visualize_yolo_results(self,
                               results,
                               save_path: Optional[str] = None,
                               show: bool = False) -> np.ndarray:
        """
        Visualize YOLO detection results.
        
        Args:
            results: YOLO results object
            save_path: Path to save visualization
            show: Whether to display the image
            
        Returns:
            Visualized image
        """
        # Get the first result
        result = results[0] if isinstance(results, list) else results
        
        # Get original image
        img = result.orig_img.copy()
        
        # Get detections
        if len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            img = self.draw_detections(img, boxes, classes, confidences)
        
        # Save if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(save_path, img)
            print(f"Visualization saved to: {save_path}")
        
        # Show if requested
        if show:
            self._show_image(img)
        
        return img
    
    def visualize_batch(self,
                       results_list: List,
                       output_dir: str = 'results/visualizations',
                       grid_size: Optional[Tuple[int, int]] = None) -> None:
        """
        Visualize a batch of detection results in a grid.
        
        Args:
            results_list: List of YOLO results
            output_dir: Directory to save visualizations
            grid_size: Grid size (rows, cols). Auto-calculated if None
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        num_images = len(results_list)
        
        # Auto-calculate grid size if not provided
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(num_images)))
            rows = int(np.ceil(num_images / cols))
            grid_size = (rows, cols)
        
        rows, cols = grid_size
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)
        
        for idx, result in enumerate(results_list):
            row = idx // cols
            col = idx % cols
            
            if row >= rows:
                break
            
            # Get visualized image
            img = self.visualize_yolo_results(result)
            
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Display
            axes[row, col].imshow(img_rgb)
            axes[row, col].axis('off')
            axes[row, col].set_title(f"Detection {idx + 1}")
        
        # Hide empty subplots
        for idx in range(num_images, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        output_path = output_dir / 'batch_visualization.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Batch visualization saved to: {output_path}")
    
    def create_detection_summary(self,
                                results_list: List,
                                output_path: str = 'results/detection_summary.png') -> None:
        """
        Create a summary visualization showing detection statistics.
        
        Args:
            results_list: List of YOLO results
            output_path: Path to save summary
        """
        # Collect statistics
        class_counts = {cls: 0 for cls in self.class_names}
        total_detections = 0
        avg_confidence = []
        
        for result in results_list:
            if len(result.boxes) > 0:
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                total_detections += len(classes)
                avg_confidence.extend(confidences)
                
                for cls in classes:
                    cls_id = int(cls)
                    if cls_id < len(self.class_names):
                        class_counts[self.class_names[cls_id]] += 1
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Class distribution bar chart
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        bars = ax1.bar(classes, counts, color=self.colors[:len(classes)], alpha=0.7)
        ax1.set_xlabel('Defect Class', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Defect Class Distribution', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # Statistics text
        stats_text = f"Detection Summary\n"
        stats_text += f"{'='*30}\n"
        stats_text += f"Total Images: {len(results_list)}\n"
        stats_text += f"Total Detections: {total_detections}\n"
        
        if avg_confidence:
            stats_text += f"Average Confidence: {np.mean(avg_confidence):.3f}\n"
            stats_text += f"Min Confidence: {np.min(avg_confidence):.3f}\n"
            stats_text += f"Max Confidence: {np.max(avg_confidence):.3f}\n"
        
        stats_text += f"\nDetections per Image: {total_detections / len(results_list):.2f}\n"
        
        ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes,
                fontsize=12, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax2.axis('off')
        
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Detection summary saved to: {output_path}")
    
    def _show_image(self, image: np.ndarray, window_name: str = "Detection") -> None:
        """
        Display image in a window.
        
        Args:
            image: Image to display
            window_name: Name of display window
        """
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def compare_predictions(self,
                          image: np.ndarray,
                          predictions1: Dict,
                          predictions2: Dict,
                          labels1: str = "Model 1",
                          labels2: str = "Model 2",
                          output_path: str = 'results/comparison.png') -> None:
        """
        Compare predictions from two models side by side.
        
        Args:
            image: Original image
            predictions1: First model predictions (boxes, classes, confidences)
            predictions2: Second model predictions
            labels1: Label for first model
            labels2: Label for second model
            output_path: Path to save comparison
        """
        # Draw detections for both models
        img1 = self.draw_detections(
            image.copy(),
            predictions1['boxes'],
            predictions1['classes'],
            predictions1['confidences']
        )
        
        img2 = self.draw_detections(
            image.copy(),
            predictions2['boxes'],
            predictions2['classes'],
            predictions2['confidences']
        )
        
        # Create side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"{labels1}\n{len(predictions1['boxes'])} detections", 
                     fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        ax2.set_title(f"{labels2}\n{len(predictions2['boxes'])} detections", 
                     fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison saved to: {output_path}")
