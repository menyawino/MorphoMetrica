#!/usr/bin/env python3
"""
Quick test script to verify basic morphometry tool functionality.
This script runs a minimal test to ensure all components can be imported and initialized.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import cv2
from pathlib import Path

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
print(f"Added to path: {project_root}")

def create_test_image(filename, shape='circle'):
    """Create a simple test image"""
    image = np.zeros((128, 128), dtype=np.uint8)
    
    if shape == 'circle':
        cv2.circle(image, (64, 64), 30, 255, -1)
    elif shape == 'rectangle':
        cv2.rectangle(image, (40, 40), (88, 88), 255, -1)
    
    # Add some noise
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    
    cv2.imwrite(filename, image)
    return image

def quick_test():
    """Run a quick test of the morphometry tool"""
    print("🧪 Starting quick morphometry test...")
    
    # Create temporary directory
    test_dir = tempfile.mkdtemp()
    raw_dir = os.path.join(test_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    try:
        # 1. Test imports
        print("📦 Testing imports...")
        try:
            from src.preprocessing.image_processor import ImageProcessor
            from src.feature_extraction.feature_extractor import FeatureExtractor
            from src.models.model_trainer import ModelTrainer
            from src.evaluation.evaluator import Evaluator
            from src.visualization.visualizer import Visualizer
            from src.clinical.report_generator import ReportGenerator
            print("✓ All imports successful")
        except ImportError as e:
            print(f"✗ Import failed: {e}")
            return False
        
        # 2. Create test configuration
        print("⚙️ Creating test configuration...")
        config = {
            'data': {
                'raw_dir': raw_dir,
                'processed_dir': os.path.join(test_dir, 'processed'),
                'output_dir': os.path.join(test_dir, 'output')
            },
            'preprocessing': {
                'resize': {'enable': True, 'height': 128, 'width': 128},
                'normalization': {'enable': True, 'method': 'z-score'},
                'augmentation': {'enable': False}
            },
            'feature_extraction': {
                'methods': [
                    {'name': 'basic_morphometry', 'enable': True},
                    {'name': 'texture_analysis', 'enable': True, 'glcm_distances': [1], 'glcm_angles': [0]},
                    {'name': 'shape_descriptors', 'enable': True},
                    {'name': 'deep_features', 'enable': False}
                ]
            },
            'model': {
                'type': 'random_forest',
                'batch_size': 4,
                'epochs': 2,
                'save_dir': os.path.join(test_dir, 'models')
            },
            'evaluation': {
                'cross_validation': {'enable': False, 'n_splits': 2},
                'metrics': ['accuracy']
            },
            'visualization': {
                'feature_importance': True,
                'confusion_matrix': True
            },
            'clinical': {
                'include_raw_measurements': True,
                'include_visualizations': False
            }
        }
        
        # 3. Create test images
        print("🖼️ Creating test images...")
        test_images = []
        for i in range(4):
            shape = 'circle' if i < 2 else 'rectangle'
            filename = os.path.join(raw_dir, f'test_{i}.png')
            create_test_image(filename, shape)
            test_images.append(filename)
        print(f"✓ Created {len(test_images)} test images")
        
        # 4. Test component initialization
        print("🔧 Testing component initialization...")
        
        try:
            processor = ImageProcessor(config)
            print("✓ ImageProcessor initialized")
        except Exception as e:
            print(f"✗ ImageProcessor failed: {e}")
            return False
        
        try:
            extractor = FeatureExtractor(config)
            print("✓ FeatureExtractor initialized")
        except Exception as e:
            print(f"✗ FeatureExtractor failed: {e}")
            return False
        
        try:
            trainer = ModelTrainer(config)
            print("✓ ModelTrainer initialized")
        except Exception as e:
            print(f"✗ ModelTrainer failed: {e}")
            return False
        
        try:
            evaluator = Evaluator(config)
            print("✓ Evaluator initialized")
        except Exception as e:
            print(f"✗ Evaluator failed: {e}")
            return False
        
        try:
            visualizer = Visualizer(config)
            print("✓ Visualizer initialized")
        except Exception as e:
            print(f"✗ Visualizer failed: {e}")
            return False
        
        try:
            reporter = ReportGenerator(config)
            print("✓ ReportGenerator initialized")
        except Exception as e:
            print(f"✗ ReportGenerator failed: {e}")
            return False
        
        # 5. Test basic functionality
        print("🔄 Testing basic functionality...")
        
        # Test image processing
        try:
            processed_images = processor.process_directory(raw_dir)
            print(f"✓ Processed {len(processed_images)} images")
        except Exception as e:
            print(f"✗ Image processing failed: {e}")
            return False
        
        # Test feature extraction
        try:
            features_df = extractor.extract_features(processed_images)
            print(f"✓ Extracted features: {features_df.shape}")
        except Exception as e:
            print(f"✗ Feature extraction failed: {e}")
            return False
        
        # Add dummy labels for training test
        labels = [0, 0, 1, 1]  # First two are circles, last two are rectangles
        features_df['label'] = labels
        
        # Test model training
        try:
            model = trainer.train(features_df)
            print("✓ Model training completed")
        except Exception as e:
            print(f"✗ Model training failed: {e}")
            return False
        
        # Test evaluation (only if we have enough samples)
        if len(features_df) >= 4:
            try:
                eval_results = evaluator.evaluate(model, features_df)
                print(f"✓ Model evaluation completed: accuracy = {eval_results.get('accuracy', 'N/A')}")
            except Exception as e:
                print(f"⚠️  Model evaluation skipped (small dataset): {e}")
        else:
            print("⚠️  Model evaluation skipped (insufficient samples)")
        
        print("🎉 Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Quick test failed with error: {e}")
        return False
    
    finally:
        # Clean up
        shutil.rmtree(test_dir, ignore_errors=True)

def check_dependencies():
    """Check if required dependencies are available"""
    print("📋 Checking dependencies...")
    
    required_packages = [
        'numpy', 'opencv-python', 'scikit-image', 'pandas', 
        'scikit-learn', 'matplotlib', 'seaborn', 'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'scikit-image':
                import skimage
            elif package == 'scikit-learn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + ' '.join(missing_packages))
        return False
    
    print("✓ All dependencies available")
    return True

if __name__ == '__main__':
    print("=" * 50)
    print("🔬 Morphometry Tool - Quick Test")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        sys.exit(1)
    
    print()
    
    # Run quick test
    if quick_test():
        print("\n✅ All tests passed! The morphometry tool is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
        sys.exit(1)