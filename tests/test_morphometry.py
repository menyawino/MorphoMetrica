#!/usr/bin/env python3
"""
Comprehensive test suite for the Morphometry Tool.
Tests all major components including preprocessing, feature extraction, modeling, evaluation, and reporting.
"""

import os
import sys
import unittest
import tempfile
import shutil
import numpy as np
import cv2
import pandas as pd
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.preprocessing.image_processor import ImageProcessor
from src.feature_extraction.feature_extractor import FeatureExtractor
from src.models.model_trainer import ModelTrainer
from src.evaluation.evaluator import Evaluator
from src.visualization.visualizer import Visualizer
from src.clinical.report_generator import ReportGenerator


class TestMorphometryTool(unittest.TestCase):
    """Main test class for the morphometry tool"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        cls.test_dir = tempfile.mkdtemp()
        cls.config = cls._create_test_config(cls.test_dir)
        cls.test_images = cls._create_test_images(cls.test_dir)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    @staticmethod
    def _create_test_config(test_dir):
        """Create a test configuration"""
        config = {
            'data': {
                'raw_dir': os.path.join(test_dir, 'raw'),
                'processed_dir': os.path.join(test_dir, 'processed'),
                'output_dir': os.path.join(test_dir, 'output')
            },
            'preprocessing': {
                'resize': {'enable': True, 'height': 256, 'width': 256},
                'normalization': {'enable': True, 'method': 'z-score'},
                'augmentation': {'enable': False}  # Disable for testing
            },
            'feature_extraction': {
                'methods': [
                    {'name': 'basic_morphometry', 'enable': True},
                    {'name': 'texture_analysis', 'enable': True, 'glcm_distances': [1, 3], 'glcm_angles': [0, 90]},
                    {'name': 'shape_descriptors', 'enable': True},
                    {'name': 'deep_features', 'enable': False}  # Disable for testing
                ]
            },
            'model': {
                'type': 'random_forest',
                'batch_size': 16,
                'epochs': 5,
                'learning_rate': 0.001,
                'save_dir': os.path.join(test_dir, 'models')
            },
            'evaluation': {
                'cross_validation': {'enable': True, 'n_splits': 3},
                'metrics': ['accuracy', 'precision', 'recall', 'f1']
            },
            'visualization': {
                'feature_importance': True,
                'confusion_matrix': True
            },
            'clinical': {
                'include_raw_measurements': True,
                'include_visualizations': False  # Disable for testing
            }
        }
        return config
    
    @staticmethod
    def _create_test_images(test_dir):
        """Create test images for testing"""
        raw_dir = os.path.join(test_dir, 'raw')
        os.makedirs(raw_dir, exist_ok=True)
        
        test_images = {}
        for i in range(5):
            # Create synthetic test images
            if i < 3:
                # Create circular objects (class 0)
                image = np.zeros((256, 256), dtype=np.uint8)
                cv2.circle(image, (128, 128), 50 + i*10, 255, -1)
                label = 0
            else:
                # Create rectangular objects (class 1)
                image = np.zeros((256, 256), dtype=np.uint8)
                cv2.rectangle(image, (100, 100), (156, 156), 255, -1)
                label = 1
            
            # Add some noise
            noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
            image = cv2.add(image, noise)
            
            filename = f'test_image_{i}.png'
            filepath = os.path.join(raw_dir, filename)
            cv2.imwrite(filepath, image)
            
            test_images[filename] = {
                'path': filepath,
                'image': image,
                'label': label
            }
        
        return test_images


class TestImageProcessor(TestMorphometryTool):
    """Test the ImageProcessor class"""
    
    def setUp(self):
        """Set up for each test"""
        self.processor = ImageProcessor(self.config)
    
    def test_initialization(self):
        """Test ImageProcessor initialization"""
        self.assertIsInstance(self.processor, ImageProcessor)
        self.assertEqual(self.processor.config, self.config)
        self.assertTrue(os.path.exists(self.processor.processed_dir))
    
    def test_process_single_image(self):
        """Test processing a single image"""
        test_image_path = list(self.test_images.values())[0]['path']
        
        # Create a method to test single image processing
        processed = self.processor._process_image_impl(test_image_path)
        
        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(processed.shape[:2], (256, 256))  # Check resize worked
    
    def test_process_directory(self):
        """Test processing a directory of images"""
        processed_images = self.processor.process_directory(self.config['data']['raw_dir'])
        
        self.assertIsInstance(processed_images, dict)
        self.assertEqual(len(processed_images), len(self.test_images))
        
        # Check that all images were processed
        for img_id in processed_images:
            self.assertIsInstance(processed_images[img_id], np.ndarray)
    
    def test_get_image_files(self):
        """Test getting image files from directory"""
        image_files = self.processor._get_image_files(self.config['data']['raw_dir'])
        
        self.assertIsInstance(image_files, list)
        self.assertEqual(len(image_files), len(self.test_images))
    
    def test_normalization(self):
        """Test image normalization"""
        test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        normalized = self.processor._normalize_image(test_image)
        
        self.assertIsInstance(normalized, np.ndarray)
        # Check that normalization changes the image
        self.assertFalse(np.array_equal(test_image, normalized))


class TestFeatureExtractor(TestMorphometryTool):
    """Test the FeatureExtractor class"""
    
    def setUp(self):
        """Set up for each test"""
        self.extractor = FeatureExtractor(self.config)
        # Create processed images for testing
        processor = ImageProcessor(self.config)
        self.processed_images = processor.process_directory(self.config['data']['raw_dir'])
    
    def test_initialization(self):
        """Test FeatureExtractor initialization"""
        self.assertIsInstance(self.extractor, FeatureExtractor)
        self.assertEqual(self.extractor.config, self.config)
    
    def test_extract_features(self):
        """Test feature extraction from images"""
        features_df = self.extractor.extract_features(self.processed_images)
        
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertEqual(len(features_df), len(self.processed_images))
        self.assertGreater(features_df.shape[1], 0)  # Should have feature columns
    
    def test_basic_morphometry(self):
        """Test basic morphometry feature extraction"""
        test_image = list(self.processed_images.values())[0]
        features = self.extractor._extract_basic_morphometry(test_image)
        
        self.assertIsInstance(features, dict)
        self.assertIn('area', features)
        self.assertIn('perimeter', features)
        self.assertIn('aspect_ratio', features)
    
    def test_texture_analysis(self):
        """Test texture analysis feature extraction"""
        test_image = list(self.processed_images.values())[0]
        features = self.extractor._extract_texture_features(test_image)
        
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
    
    def test_shape_descriptors(self):
        """Test shape descriptor feature extraction"""
        test_image = list(self.processed_images.values())[0]
        features = self.extractor._extract_shape_descriptors(test_image)
        
        self.assertIsInstance(features, dict)
        self.assertIn('solidity', features)
        self.assertIn('extent', features)
    
    def test_is_method_enabled(self):
        """Test method enabling check"""
        self.assertTrue(self.extractor.is_method_enabled('basic_morphometry'))
        self.assertFalse(self.extractor.is_method_enabled('deep_features'))


class TestModelTrainer(TestMorphometryTool):
    """Test the ModelTrainer class"""
    
    def setUp(self):
        """Set up for each test"""
        self.trainer = ModelTrainer(self.config)
        
        # Create test features and labels
        processor = ImageProcessor(self.config)
        processed_images = processor.process_directory(self.config['data']['raw_dir'])
        
        extractor = FeatureExtractor(self.config)
        self.features_df = extractor.extract_features(processed_images)
        
        # Create labels based on our test images
        self.labels = []
        for img_name in self.features_df.index:
            img_num = int(img_name.split('_')[2].split('.')[0])
            self.labels.append(0 if img_num < 3 else 1)
        
        self.features_df['label'] = self.labels
    
    def test_initialization(self):
        """Test ModelTrainer initialization"""
        self.assertIsInstance(self.trainer, ModelTrainer)
        self.assertEqual(self.trainer.config, self.config)
    
    def test_train_model(self):
        """Test model training"""
        model = self.trainer.train(self.features_df)
        
        self.assertIsNotNone(model)
        # Check that model can make predictions
        test_features = self.features_df.drop('label', axis=1)
        predictions = model.predict(test_features)
        self.assertEqual(len(predictions), len(test_features))
    
    def test_prepare_data(self):
        """Test data preparation for training"""
        X, y = self.trainer._prepare_data(self.features_df)
        
        self.assertIsInstance(X, (np.ndarray, pd.DataFrame))
        self.assertIsInstance(y, (np.ndarray, pd.Series))
        self.assertEqual(len(X), len(y))
    
    def test_save_and_load_model(self):
        """Test model saving and loading"""
        model = self.trainer.train(self.features_df)
        
        # Save model
        model_path = self.trainer.save_model(model, 'test_model')
        self.assertTrue(os.path.exists(model_path))
        
        # Load model
        loaded_model = self.trainer.load_model(model_path)
        self.assertIsNotNone(loaded_model)


class TestEvaluator(TestMorphometryTool):
    """Test the Evaluator class"""
    
    def setUp(self):
        """Set up for each test"""
        self.evaluator = Evaluator(self.config)
        
        # Create test model and data
        processor = ImageProcessor(self.config)
        processed_images = processor.process_directory(self.config['data']['raw_dir'])
        
        extractor = FeatureExtractor(self.config)
        features_df = extractor.extract_features(processed_images)
        
        # Create labels
        labels = []
        for img_name in features_df.index:
            img_num = int(img_name.split('_')[2].split('.')[0])
            labels.append(0 if img_num < 3 else 1)
        features_df['label'] = labels
        
        trainer = ModelTrainer(self.config)
        self.model = trainer.train(features_df)
        self.features_df = features_df
    
    def test_initialization(self):
        """Test Evaluator initialization"""
        self.assertIsInstance(self.evaluator, Evaluator)
        self.assertEqual(self.evaluator.config, self.config)
    
    def test_evaluate_model(self):
        """Test model evaluation"""
        results = self.evaluator.evaluate(self.model, self.features_df)
        
        self.assertIsInstance(results, dict)
        self.assertIn('accuracy', results)
        self.assertIn('precision', results)
        self.assertIn('recall', results)
        self.assertIn('f1', results)
    
    def test_cross_validation(self):
        """Test cross-validation evaluation"""
        X = self.features_df.drop('label', axis=1)
        y = self.features_df['label']
        
        cv_scores = self.evaluator._cross_validate(self.model, X, y)
        
        self.assertIsInstance(cv_scores, dict)
        self.assertIn('accuracy', cv_scores)


class TestVisualizer(TestMorphometryTool):
    """Test the Visualizer class"""
    
    def setUp(self):
        """Set up for each test"""
        self.visualizer = Visualizer(self.config)
        
        # Create test data
        processor = ImageProcessor(self.config)
        processed_images = processor.process_directory(self.config['data']['raw_dir'])
        
        extractor = FeatureExtractor(self.config)
        features_df = extractor.extract_features(processed_images)
        
        labels = []
        for img_name in features_df.index:
            img_num = int(img_name.split('_')[2].split('.')[0])
            labels.append(0 if img_num < 3 else 1)
        features_df['label'] = labels
        
        trainer = ModelTrainer(self.config)
        model = trainer.train(features_df)
        
        evaluator = Evaluator(self.config)
        evaluation_results = evaluator.evaluate(model, features_df)
        
        self.features_df = features_df
        self.model = model
        self.evaluation_results = evaluation_results
    
    def test_initialization(self):
        """Test Visualizer initialization"""
        self.assertIsInstance(self.visualizer, Visualizer)
        self.assertEqual(self.visualizer.config, self.config)
    
    def test_visualize(self):
        """Test visualization generation"""
        # This should not raise an exception
        self.visualizer.visualize(self.features_df, self.model, self.evaluation_results)
        
        # Check that visualization files are created
        output_dir = Path(self.config['data']['output_dir'])
        viz_files = list(output_dir.glob('*.png'))
        self.assertGreater(len(viz_files), 0)
    
    def test_plot_confusion_matrix(self):
        """Test confusion matrix plotting"""
        y_true = self.features_df['label']
        X = self.features_df.drop('label', axis=1)
        y_pred = self.model.predict(X)
        
        # This should not raise an exception
        self.visualizer._plot_confusion_matrix(y_true, y_pred)


class TestReportGenerator(TestMorphometryTool):
    """Test the ReportGenerator class"""
    
    def setUp(self):
        """Set up for each test"""
        self.report_generator = ReportGenerator(self.config)
        
        # Create test data
        processor = ImageProcessor(self.config)
        processed_images = processor.process_directory(self.config['data']['raw_dir'])
        
        extractor = FeatureExtractor(self.config)
        features_df = extractor.extract_features(processed_images)
        
        labels = []
        for img_name in features_df.index:
            img_num = int(img_name.split('_')[2].split('.')[0])
            labels.append(0 if img_num < 3 else 1)
        features_df['label'] = labels
        
        trainer = ModelTrainer(self.config)
        model = trainer.train(features_df)
        
        evaluator = Evaluator(self.config)
        evaluation_results = evaluator.evaluate(model, features_df)
        
        self.features_df = features_df
        self.model = model
        self.evaluation_results = evaluation_results
    
    def test_initialization(self):
        """Test ReportGenerator initialization"""
        self.assertIsInstance(self.report_generator, ReportGenerator)
        self.assertEqual(self.report_generator.config, self.config)
    
    def test_generate_report(self):
        """Test report generation"""
        # This should not raise an exception
        self.report_generator.generate_report(
            self.features_df, 
            self.model, 
            self.evaluation_results
        )
        
        # Check that report file is created
        output_dir = Path(self.config['data']['output_dir'])
        report_files = list(output_dir.glob('*report*'))
        self.assertGreater(len(report_files), 0)
    
    def test_generate_prediction_report(self):
        """Test prediction report generation"""
        X = self.features_df.drop('label', axis=1)
        predictions = self.model.predict(X)
        
        # This should not raise an exception
        self.report_generator.generate_prediction_report(X, predictions)


class TestIntegration(TestMorphometryTool):
    """Integration tests for the complete pipeline"""
    
    def test_complete_pipeline(self):
        """Test the complete morphometry pipeline"""
        # Preprocessing
        processor = ImageProcessor(self.config)
        processed_images = processor.process_directory(self.config['data']['raw_dir'])
        
        # Feature extraction
        extractor = FeatureExtractor(self.config)
        features_df = extractor.extract_features(processed_images)
        
        # Add labels for training
        labels = []
        for img_name in features_df.index:
            img_num = int(img_name.split('_')[2].split('.')[0])
            labels.append(0 if img_num < 3 else 1)
        features_df['label'] = labels
        
        # Model training
        trainer = ModelTrainer(self.config)
        model = trainer.train(features_df)
        
        # Model evaluation
        evaluator = Evaluator(self.config)
        evaluation_results = evaluator.evaluate(model, features_df)
        
        # Visualization
        visualizer = Visualizer(self.config)
        visualizer.visualize(features_df, model, evaluation_results)
        
        # Report generation
        report_generator = ReportGenerator(self.config)
        report_generator.generate_report(features_df, model, evaluation_results)
        
        # Verify that all steps completed successfully
        self.assertIsNotNone(processed_images)
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertIsNotNone(model)
        self.assertIsInstance(evaluation_results, dict)
        
        # Check that output files exist
        output_dir = Path(self.config['data']['output_dir'])
        self.assertTrue(output_dir.exists())
        
        feature_file = output_dir / 'features.csv'
        self.assertTrue(feature_file.exists())
    
    def test_prediction_pipeline(self):
        """Test the prediction pipeline"""
        # First train a model
        processor = ImageProcessor(self.config)
        processed_images = processor.process_directory(self.config['data']['raw_dir'])
        
        extractor = FeatureExtractor(self.config)
        features_df = extractor.extract_features(processed_images)
        
        labels = []
        for img_name in features_df.index:
            img_num = int(img_name.split('_')[2].split('.')[0])
            labels.append(0 if img_num < 3 else 1)
        features_df['label'] = labels
        
        trainer = ModelTrainer(self.config)
        model = trainer.train(features_df)
        
        # Save model
        model_path = trainer.save_model(model, 'test_prediction_model')
        
        # Load model and make predictions
        loaded_model = trainer.load_model(model_path)
        X = features_df.drop('label', axis=1)
        predictions = loaded_model.predict(X)
        
        self.assertEqual(len(predictions), len(features_df))
        self.assertIsNotNone(predictions)


def run_quick_test():
    """Run a quick test to verify basic functionality"""
    print("Running quick morphometry test...")
    
    # Create a simple test
    test_suite = unittest.TestSuite()
    test_suite.addTest(TestImageProcessor('test_initialization'))
    test_suite.addTest(TestFeatureExtractor('test_initialization'))
    test_suite.addTest(TestModelTrainer('test_initialization'))
    test_suite.addTest(TestEvaluator('test_initialization'))
    test_suite.addTest(TestVisualizer('test_initialization'))
    test_suite.addTest(TestReportGenerator('test_initialization'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    if result.wasSuccessful():
        print("✓ Quick test passed!")
    else:
        print("✗ Quick test failed!")
        return False
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run morphometry tests')
    parser.add_argument('--quick', action='store_true', 
                       help='Run only quick tests for basic functionality')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_test()
        sys.exit(0 if success else 1)
    
    # Run all tests
    if args.integration:
        # Run only integration tests
        suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegration)
    else:
        # Run all tests
        suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)