"""
Inference Script for Defect Detection
MSc Thesis - Arden University Berlin

This script handles inference/prediction on new images for defect detection.
"""

import argparse
import time
from pathlib import Path
import sys
import cv2

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models.defect_detector import DefectDetectionModel
from src.visualization.visualizer import DefectVisualizer
from src.utils.helpers import setup_logging, load_config, format_time


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run inference for defect detection'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to image, directory, or video file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model weights'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=None,
        help='Confidence threshold (overrides config)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=None,
        help='IoU threshold for NMS (overrides config)'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save predictions'
    )
    
    parser.add_argument(
        '--save-txt',
        action='store_true',
        help='Save predictions to text files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/predictions',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display results'
    )
    
    return parser.parse_args()


def main():
    """Main inference function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging('logs/inference.log')
    logger.info("Starting defect detection inference")
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Initialize model
    logger.info("Initializing defect detection model")
    model = DefectDetectionModel(config_path=args.config)
    
    # Load trained model
    logger.info(f"Loading model from: {args.model}")
    model.load_custom(args.model)
    
    # Display model info
    model_info = model.get_model_info()
    logger.info("Model Information:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")
    
    # Run inference
    logger.info(f"\nRunning inference on: {args.source}")
    start_time = time.time()
    
    try:
        results = model.predict(
            source=args.source,
            conf=args.conf,
            iou=args.iou,
            save=args.save,
            save_txt=args.save_txt,
            save_dir=args.output_dir
        )
        
        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {format_time(inference_time)}")
        
        # Print detection statistics
        total_detections = 0
        for result in results:
            total_detections += len(result.boxes)
        
        logger.info(f"\nDetection Summary:")
        logger.info(f"  Total images processed: {len(results)}")
        logger.info(f"  Total defects detected: {total_detections}")
        logger.info(f"  Average detections per image: {total_detections / len(results):.2f}")
        
        # Visualize results if requested
        if args.visualize or args.show:
            logger.info("\nGenerating visualizations...")
            visualizer = DefectVisualizer(config_path=args.config)
            
            # Visualize each result
            for idx, result in enumerate(results):
                if args.visualize:
                    save_path = Path(args.output_dir) / 'visualizations' / f'detection_{idx}.jpg'
                    visualizer.visualize_yolo_results(
                        result,
                        save_path=str(save_path),
                        show=args.show
                    )
                elif args.show:
                    visualizer.visualize_yolo_results(result, show=True)
            
            # Create summary visualization
            if args.visualize:
                visualizer.create_detection_summary(
                    results,
                    output_path=str(Path(args.output_dir) / 'detection_summary.png')
                )
        
        logger.info(f"\nResults saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Inference failed with error: {str(e)}")
        raise
    
    logger.info("\nInference completed successfully!")


if __name__ == '__main__':
    main()
