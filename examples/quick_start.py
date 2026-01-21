"""
Example: Quick Start Guide
MSc Thesis - Arden University Berlin

This example demonstrates basic usage of the defect detection system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.defect_detector import DefectDetectionModel
from src.visualization.visualizer import DefectVisualizer
from src.evaluation.metrics import DefectDetectionEvaluator


def example_basic_inference():
    """Example: Basic inference on an image."""
    print("\n" + "="*60)
    print("Example 1: Basic Inference")
    print("="*60)
    
    # Initialize model
    model = DefectDetectionModel(config_path='configs/config.yaml')
    
    # Load a trained model (replace with your model path)
    model_path = 'models/trained/defect_detection/weights/best.pt'
    
    if Path(model_path).exists():
        model.load_custom(model_path)
        
        # Run inference on an image
        image_path = 'path/to/your/image.jpg'  # Replace with actual path
        
        if Path(image_path).exists():
            results = model.predict(
                source=image_path,
                conf=0.25,
                save=True,
                save_dir='results/examples'
            )
            
            print(f"Detections: {len(results[0].boxes)}")
            print("Results saved to: results/examples")
        else:
            print(f"Image not found: {image_path}")
    else:
        print(f"Model not found: {model_path}")
        print("Please train a model first or provide a valid model path.")


def example_batch_inference():
    """Example: Batch inference on a directory."""
    print("\n" + "="*60)
    print("Example 2: Batch Inference")
    print("="*60)
    
    # Initialize model
    model = DefectDetectionModel(config_path='configs/config.yaml')
    
    # Load trained model
    model_path = 'models/trained/defect_detection/weights/best.pt'
    
    if Path(model_path).exists():
        model.load_custom(model_path)
        
        # Run inference on a directory
        images_dir = 'data/processed/test/images'  # Replace with actual path
        
        if Path(images_dir).exists():
            results = model.predict(
                source=images_dir,
                conf=0.25,
                save=True,
                save_txt=True,
                save_dir='results/batch_inference'
            )
            
            print(f"Processed {len(results)} images")
            print("Results saved to: results/batch_inference")
        else:
            print(f"Directory not found: {images_dir}")
    else:
        print(f"Model not found: {model_path}")


def example_visualization():
    """Example: Visualize detection results."""
    print("\n" + "="*60)
    print("Example 3: Visualization")
    print("="*60)
    
    # Initialize model and visualizer
    model = DefectDetectionModel(config_path='configs/config.yaml')
    visualizer = DefectVisualizer(config_path='configs/config.yaml')
    
    # Load trained model
    model_path = 'models/trained/defect_detection/weights/best.pt'
    
    if Path(model_path).exists():
        model.load_custom(model_path)
        
        # Run inference
        image_path = 'path/to/your/image.jpg'  # Replace with actual path
        
        if Path(image_path).exists():
            results = model.predict(source=image_path, conf=0.25)
            
            # Visualize results
            visualizer.visualize_yolo_results(
                results[0],
                save_path='results/examples/visualization.jpg'
            )
            
            # Create detection summary
            visualizer.create_detection_summary(
                results,
                output_path='results/examples/summary.png'
            )
            
            print("Visualizations saved to: results/examples/")
        else:
            print(f"Image not found: {image_path}")
    else:
        print(f"Model not found: {model_path}")


def example_evaluation():
    """Example: Evaluate model performance."""
    print("\n" + "="*60)
    print("Example 4: Model Evaluation")
    print("="*60)
    
    # Initialize model and evaluator
    model = DefectDetectionModel(config_path='configs/config.yaml')
    evaluator = DefectDetectionEvaluator(model.class_names)
    
    # Load trained model
    model_path = 'models/trained/defect_detection/weights/best.pt'
    dataset_yaml = 'data/dataset.yaml'
    
    if Path(model_path).exists() and Path(dataset_yaml).exists():
        model.load_custom(model_path)
        
        # Validate model
        print("Running validation...")
        val_results = model.validate(data_yaml=dataset_yaml)
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics_from_yolo_results(val_results)
        
        # Print and save metrics
        evaluator.print_metrics(metrics)
        evaluator.save_metrics_to_csv(metrics, 'results/examples/metrics.csv')
        
        # Generate plots
        evaluator.plot_metrics_comparison(metrics, 'results/examples/metrics.png')
        
        if 'per_class' in metrics:
            evaluator.plot_per_class_metrics(
                metrics, 
                'results/examples/per_class_metrics.png'
            )
        
        print("Evaluation results saved to: results/examples/")
    else:
        if not Path(model_path).exists():
            print(f"Model not found: {model_path}")
        if not Path(dataset_yaml).exists():
            print(f"Dataset YAML not found: {dataset_yaml}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Defect Detection System - Quick Start Guide")
    print("MSc Thesis - Arden University Berlin")
    print("="*60)
    
    # Run examples
    example_basic_inference()
    example_batch_inference()
    example_visualization()
    example_evaluation()
    
    print("\n" + "="*60)
    print("Quick start guide completed!")
    print("="*60)
    print("\nNote: Make sure to:")
    print("1. Prepare your dataset using examples/prepare_data.py")
    print("2. Train your model using train.py")
    print("3. Then run these examples with your trained model")


if __name__ == '__main__':
    main()
