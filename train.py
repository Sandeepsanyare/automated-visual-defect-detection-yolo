"""
Training Script for Defect Detection Model
MSc Thesis - Arden University Berlin

This script handles the complete training pipeline for the YOLO-based defect detection model.
"""

import argparse
import time
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.defect_detector import DefectDetectionModel
from utils.helpers import (
    setup_logging, set_seed, load_config, 
    print_system_info, format_time, check_dataset_exists
)
from evaluation.metrics import DefectDetectionEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train YOLO model for defect detection'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/dataset.yaml',
        help='Path to dataset YAML file'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name (e.g., yolov8n, yolov8s)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting defect detection model training")
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")
    
    # Print system info
    print_system_info()
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Check if dataset exists
    if not check_dataset_exists(args.data):
        logger.error("Dataset not found or improperly configured!")
        logger.error("Please prepare your dataset using the preprocessing utilities.")
        return
    
    # Initialize model
    logger.info("Initializing defect detection model")
    model = DefectDetectionModel(config_path=args.config)
    
    # Load pretrained weights or resume from checkpoint
    if args.resume:
        logger.info(f"Resuming training from: {args.resume}")
        model.load_custom(args.resume)
    else:
        model_name = args.model or config['model']['name']
        logger.info(f"Loading pretrained model: {model_name}")
        model.load_pretrained(model_name)
    
    # Display model info
    model_info = model.get_model_info()
    logger.info("Model Information:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")
    
    # Training parameters
    epochs = args.epochs or config['training']['epochs']
    batch_size = args.batch_size or config['training']['batch_size']
    
    logger.info(f"\nTraining Configuration:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")
    logger.info(f"  Device: {model.device}")
    
    # Start training
    logger.info("\n" + "="*60)
    logger.info("Starting Training")
    logger.info("="*60)
    
    start_time = time.time()
    
    try:
        results = model.train(
            data_yaml=args.data,
            epochs=epochs,
            batch_size=batch_size,
            save_dir='models/trained'
        )
        
        training_time = time.time() - start_time
        logger.info(f"\nTraining completed in {format_time(training_time)}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    
    # Validate model
    logger.info("\n" + "="*60)
    logger.info("Validating Model")
    logger.info("="*60)
    
    try:
        val_results = model.validate(data_yaml=args.data)
        
        # Evaluate and print metrics
        evaluator = DefectDetectionEvaluator(config['classes'])
        metrics = evaluator.calculate_metrics_from_yolo_results(val_results)
        
        # Print metrics
        evaluator.print_metrics(metrics)
        
        # Save metrics
        evaluator.save_metrics_to_csv(metrics, 'results/training_metrics.csv')
        
        # Generate plots
        evaluator.plot_metrics_comparison(metrics, 'results/training_metrics.png')
        if 'per_class' in metrics:
            evaluator.plot_per_class_metrics(metrics, 'results/per_class_metrics.png')
        
        logger.info("Validation and evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Validation failed with error: {str(e)}")
    
    # Export model
    logger.info("\n" + "="*60)
    logger.info("Training Summary")
    logger.info("="*60)
    logger.info(f"Total time: {format_time(time.time() - start_time)}")
    logger.info(f"Model saved in: models/trained/defect_detection")
    logger.info(f"Results saved in: results/")
    logger.info("="*60)
    
    logger.info("\nTraining pipeline completed successfully!")


if __name__ == '__main__':
    main()
