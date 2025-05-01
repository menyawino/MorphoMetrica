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

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.image_processor import ImageProcessor
from src.feature_extraction.feature_extractor import FeatureExtractor
from src.models.model_trainer import ModelTrainer
from src.evaluation.evaluator import Evaluator
from src.visualization.visualizer import Visualizer
from src.clinical.report_generator import ReportGenerator


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
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'evaluate', 'report'],
                       default='train', help='Operation mode')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to saved model for predict/evaluate mode')
    return parser.parse_args()


def main():
    """Main function to run the morphometry pipeline"""
    # Parse arguments and load configuration
    args = parse_arguments()
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.data_dir:
        config['data']['raw_dir'] = args.data_dir
    if args.output_dir:
        config['data']['output_dir'] = args.output_dir
    
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
    
    print(f"Morphometry analysis completed in {args.mode} mode.")


if __name__ == "__main__":
    main()