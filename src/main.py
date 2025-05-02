#!/usr/bin/env python3
"""
Main entry point for the Morphometry Tool.
This script orchestrates the entire pipeline from data loading to clinical reporting.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import time
import multiprocessing
import logging

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.image_processor import ImageProcessor
from src.feature_extraction.feature_extractor import FeatureExtractor
from src.models.model_trainer import ModelTrainer
from src.evaluation.evaluator import Evaluator
from src.visualization.visualizer import Visualizer
from src.clinical.report_generator import ReportGenerator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("morphometry.log")
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Morphometry Analysis Tool')
    parser.add_argument('--config', type=str, default='../configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to input data directory (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Path to output directory (overrides config)')
    parser.add_argument('--mode', type=str, 
                       choices=['train', 'predict', 'evaluate', 'report', 'extract_features'],
                       default='train', help='Operation mode')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to saved model for predict/evaluate mode')
    
    # Acceleration parameters
    parser.add_argument('--workers', type=int, default=None, 
                       help='Number of worker processes (default: auto-detect)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for processing (overrides config)')
    parser.add_argument('--gpu', action='store_true',
                       help='Explicitly try to use GPU acceleration')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching for image processing and feature extraction')
    parser.add_argument('--profile', action='store_true',
                       help='Enable performance profiling')
    
    # Feature extraction specific arguments
    parser.add_argument('--features', type=str, default=None,
                       help='Comma-separated list of feature groups to extract (basic,texture,shape,deep)')
    
    return parser.parse_args()


def main():
    """Main function to run the morphometry pipeline"""
    # Parse arguments and load configuration
    args = parse_arguments()
    config = load_config(args.config)
    
    # Record start time if profiling is enabled
    if args.profile:
        start_time = time.time()
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
    
    # Override config with command line arguments if provided
    if args.data_dir:
        config['data']['raw_dir'] = args.data_dir
    if args.output_dir:
        config['data']['output_dir'] = args.output_dir
    if args.batch_size:
        config['model']['batch_size'] = args.batch_size
    
    # Set worker count
    if args.workers:
        n_workers = args.workers
    else:
        n_workers = min(multiprocessing.cpu_count(), 8)
    
    # Update config with acceleration parameters
    if 'processing' not in config:
        config['processing'] = {}
    config['processing']['n_workers'] = n_workers
    config['processing']['use_gpu'] = args.gpu
    config['processing']['use_cache'] = not args.no_cache
    
    # Log execution plan
    logger.info(f"Starting Morphometry Tool in {args.mode} mode")
    logger.info(f"Using {n_workers} worker processes")
    if args.gpu:
        logger.info("GPU acceleration enabled")
    
    # Create output directories if they don't exist
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    image_processor = ImageProcessor(config)
    feature_extractor = FeatureExtractor(config)
    model_trainer = ModelTrainer(config)
    evaluator = Evaluator(config)
    visualizer = Visualizer(config)
    report_generator = ReportGenerator(config)
    
    # Execute pipeline based on mode
    if args.mode == 'train':
        # Preprocess images
        processed_images = image_processor.process_directory(config['data']['raw_dir'])
        
        # Extract features
        features = feature_extractor.extract_features(processed_images)
        
        # Train model
        model = model_trainer.train(features)
        
        # Evaluate model
        evaluation_results = evaluator.evaluate(model, features)
        
        # Generate visualizations
        visualizer.visualize(features, model, evaluation_results)
        
        # Generate report
        report_generator.generate_report(features, model, evaluation_results)
        
    elif args.mode == 'extract_features':
        # Only preprocess and extract features
        logger.info("Extracting features only")
        
        # Preprocess images
        processed_images = image_processor.process_directory(config['data']['raw_dir'])
        
        # Configure feature extraction based on arguments
        if args.features:
            selected_features = [f.strip() for f in args.features.split(',')]
            for method in config['feature_extraction']['methods']:
                method['enable'] = method['name'] in selected_features
            
            logger.info(f"Selected feature groups: {selected_features}")
        
        # Extract features
        features = feature_extractor.extract_features(processed_images)
        
        logger.info(f"Features extracted and saved to {output_dir}/features.csv")
        
    elif args.mode == 'predict':
        if not args.model_path:
            raise ValueError("Model path must be provided in predict mode")
        
        # Load model
        model = model_trainer.load_model(args.model_path)
        
        # Preprocess images
        processed_images = image_processor.process_directory(config['data']['raw_dir'])
        
        # Extract features
        features = feature_extractor.extract_features(processed_images)
        
        # Make predictions
        predictions = model.predict(features)
        
        # Generate report
        report_generator.generate_prediction_report(features, predictions)
        
    elif args.mode == 'evaluate':
        if not args.model_path:
            raise ValueError("Model path must be provided in evaluate mode")
        
        # Load model
        model = model_trainer.load_model(args.model_path)
        
        # Preprocess images
        processed_images = image_processor.process_directory(config['data']['raw_dir'])
        
        # Extract features
        features = feature_extractor.extract_features(processed_images)
        
        # Evaluate model
        evaluation_results = evaluator.evaluate(model, features)
        
        # Generate visualizations
        visualizer.visualize(features, model, evaluation_results)
        
    elif args.mode == 'report':
        if not args.model_path:
            raise ValueError("Model path must be provided in report mode")
        
        # Load model
        model = model_trainer.load_model(args.model_path)
        
        # Preprocess images
        processed_images = image_processor.process_directory(config['data']['raw_dir'])
        
        # Extract features
        features = feature_extractor.extract_features(processed_images)
        
        # Make predictions
        predictions = model.predict(features)
        
        # Generate clinical report
        report_generator.generate_clinical_report(features, predictions)
    
    # Output profiling results if enabled
    if args.profile:
        profiler.disable()
        
        # Print profiling summary to console
        import io
        import pstats
        from pstats import SortKey
        
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats(25)  # Show top 25 functions
        print("Performance profile:")
        print(s.getvalue())
        
        # Save detailed profiling to file
        profile_path = output_dir / "profile_results.txt"
        with open(profile_path, 'w') as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats(sortby)
            stats.print_stats()
            
        # Print total execution time
        elapsed_time = time.time() - start_time
        print(f"Total execution time: {elapsed_time:.2f} seconds")
    
    logger.info(f"Morphometry analysis completed in {args.mode} mode.")


if __name__ == "__main__":
    main()