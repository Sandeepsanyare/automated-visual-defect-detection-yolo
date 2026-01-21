"""
Evaluation Metrics Module
MSc Thesis - Arden University Berlin

This module provides comprehensive evaluation metrics for defect detection models.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class DefectDetectionEvaluator:
    """
    Evaluator class for defect detection model performance.
    
    Computes and visualizes various metrics for object detection models.
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize evaluator with class names.
        
        Args:
            class_names: List of defect class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def calculate_metrics_from_yolo_results(self, results) -> Dict:
        """
        Calculate metrics from YOLO validation results.
        
        Args:
            results: YOLO validation results object
            
        Returns:
            Dictionary containing various metrics
        """
        metrics = {
            'precision': float(results.box.mp),  # Mean precision
            'recall': float(results.box.mr),  # Mean recall
            'mAP50': float(results.box.map50),  # mAP at IoU=0.5
            'mAP50-95': float(results.box.map),  # mAP at IoU=0.5:0.95
        }
        
        # Calculate F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                                 (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0
        
        # Per-class metrics if available
        if hasattr(results.box, 'ap_class_index'):
            metrics['per_class'] = {}
            for i, cls_idx in enumerate(results.box.ap_class_index):
                if cls_idx < len(self.class_names):
                    cls_name = self.class_names[int(cls_idx)]
                    metrics['per_class'][cls_name] = {
                        'precision': float(results.box.p[i]) if i < len(results.box.p) else 0.0,
                        'recall': float(results.box.r[i]) if i < len(results.box.r) else 0.0,
                        'mAP50': float(results.box.ap50[i]) if i < len(results.box.ap50) else 0.0,
                        'mAP50-95': float(results.box.ap[i]) if i < len(results.box.ap) else 0.0,
                    }
        
        return metrics
    
    def print_metrics(self, metrics: Dict) -> None:
        """
        Print evaluation metrics in a formatted way.
        
        Args:
            metrics: Dictionary containing metrics
        """
        print("\n" + "="*60)
        print("Model Evaluation Metrics")
        print("="*60)
        print(f"Overall Precision:    {metrics['precision']:.4f}")
        print(f"Overall Recall:       {metrics['recall']:.4f}")
        print(f"Overall F1-Score:     {metrics['f1_score']:.4f}")
        print(f"mAP@0.5:             {metrics['mAP50']:.4f}")
        print(f"mAP@0.5:0.95:        {metrics['mAP50-95']:.4f}")
        
        if 'per_class' in metrics:
            print("\n" + "-"*60)
            print("Per-Class Metrics:")
            print("-"*60)
            print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'mAP50':<12}")
            print("-"*60)
            
            for cls_name, cls_metrics in metrics['per_class'].items():
                print(f"{cls_name:<20} "
                      f"{cls_metrics['precision']:<12.4f} "
                      f"{cls_metrics['recall']:<12.4f} "
                      f"{cls_metrics['mAP50']:<12.4f}")
        
        print("="*60 + "\n")
    
    def save_metrics_to_csv(self, 
                           metrics: Dict,
                           output_path: str = 'results/metrics.csv') -> None:
        """
        Save metrics to a CSV file.
        
        Args:
            metrics: Dictionary containing metrics
            output_path: Path to save CSV file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Overall metrics
        overall_df = pd.DataFrame([{
            'Metric': 'Overall',
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'mAP50': metrics['mAP50'],
            'mAP50-95': metrics['mAP50-95']
        }])
        
        # Per-class metrics if available
        if 'per_class' in metrics:
            per_class_data = []
            for cls_name, cls_metrics in metrics['per_class'].items():
                per_class_data.append({
                    'Metric': cls_name,
                    'Precision': cls_metrics['precision'],
                    'Recall': cls_metrics['recall'],
                    'F1-Score': 2 * (cls_metrics['precision'] * cls_metrics['recall']) / 
                               (cls_metrics['precision'] + cls_metrics['recall'])
                               if (cls_metrics['precision'] + cls_metrics['recall']) > 0 else 0,
                    'mAP50': cls_metrics['mAP50'],
                    'mAP50-95': cls_metrics['mAP50-95']
                })
            
            per_class_df = pd.DataFrame(per_class_data)
            df = pd.concat([overall_df, per_class_df], ignore_index=True)
        else:
            df = overall_df
        
        df.to_csv(output_path, index=False)
        print(f"Metrics saved to: {output_path}")
    
    def plot_metrics_comparison(self,
                               metrics: Dict,
                               output_path: str = 'results/metrics_comparison.png') -> None:
        """
        Create a bar plot comparing different metrics.
        
        Args:
            metrics: Dictionary containing metrics
            output_path: Path to save plot
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = ['Precision', 'Recall', 'F1-Score', 'mAP50', 'mAP50-95']
        metric_values = [
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score'],
            metrics['mAP50'],
            metrics['mAP50-95']
        ]
        
        bars = ax.bar(metric_names, metric_values, color='steelblue', alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Metrics comparison plot saved to: {output_path}")
    
    def plot_per_class_metrics(self,
                              metrics: Dict,
                              output_path: str = 'results/per_class_metrics.png') -> None:
        """
        Create a grouped bar plot for per-class metrics.
        
        Args:
            metrics: Dictionary containing metrics with per_class data
            output_path: Path to save plot
        """
        if 'per_class' not in metrics:
            print("No per-class metrics available")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        classes = list(metrics['per_class'].keys())
        precision = [metrics['per_class'][c]['precision'] for c in classes]
        recall = [metrics['per_class'][c]['recall'] for c in classes]
        map50 = [metrics['per_class'][c]['mAP50'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, map50, width, label='mAP50', alpha=0.8)
        
        ax.set_xlabel('Defect Classes', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Per-class metrics plot saved to: {output_path}")
    
    def create_confusion_matrix(self,
                               predictions: List[int],
                               ground_truth: List[int],
                               output_path: str = 'results/confusion_matrix.png') -> np.ndarray:
        """
        Create and plot a confusion matrix.
        
        Args:
            predictions: List of predicted class IDs
            ground_truth: List of ground truth class IDs
            output_path: Path to save plot
            
        Returns:
            Confusion matrix as numpy array
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cm = confusion_matrix(ground_truth, predictions, 
                            labels=list(range(self.num_classes)))
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('True Class', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {output_path}")
        return cm
    
    def generate_evaluation_report(self,
                                  metrics: Dict,
                                  output_dir: str = 'results') -> None:
        """
        Generate a comprehensive evaluation report with all metrics and plots.
        
        Args:
            metrics: Dictionary containing all metrics
            output_dir: Directory to save report files
        """
        print("\nGenerating comprehensive evaluation report...")
        
        # Print metrics
        self.print_metrics(metrics)
        
        # Save to CSV
        self.save_metrics_to_csv(metrics, f"{output_dir}/metrics.csv")
        
        # Generate plots
        self.plot_metrics_comparison(metrics, f"{output_dir}/metrics_comparison.png")
        
        if 'per_class' in metrics:
            self.plot_per_class_metrics(metrics, f"{output_dir}/per_class_metrics.png")
        
        print(f"\nEvaluation report generated in: {output_dir}")
