"""
Feature extraction module for morphometry analysis.
This module contains methods to extract morphometric features from images.
"""

import numpy as np
import pandas as pd
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops, regionprops_table
from skimage import morphology
from scipy import ndimage
import logging
import tensorflow as tf
import torch
from pathlib import Path
import os
import hashlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from joblib import Memory, Parallel, delayed
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract morphometric features from preprocessed images"""
    
    def __init__(self, config):
        """Initialize the feature extractor with configuration"""
        self.config = config
        self.feature_config = config['feature_extraction']
        self.output_dir = Path(config['data']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup cache directory
        cache_dir = self.output_dir / '.feature_cache'
        self.memory = Memory(cache_dir, verbose=0)
        
        # Number of workers for parallel processing
        self.n_workers = os.cpu_count() or 1
        if self.n_workers > 2:
            self.n_workers = min(self.n_workers, 4)  # Limit workers for feature extraction
        
        # Check for GPU availability
        self.gpu_available = self._check_gpu()
        
        # Initialize deep learning model for feature extraction if needed
        if self.is_method_enabled('deep_features'):
            self._initialize_deep_model()
    
    def extract_features(self, images):
        """Extract features from a dictionary of preprocessed images"""
        start_time = time.time()
        logger.info(f"Extracting features from {len(images)} images using {self.n_workers} workers")
        
        # Process large dictionaries in chunks to avoid memory issues
        image_ids = list(images.keys())
        chunk_size = 100  # Adjust based on memory constraints
        
        all_features = {}
        for i in range(0, len(image_ids), chunk_size):
            chunk_ids = image_ids[i:i+chunk_size]
            chunk_images = {img_id: images[img_id] for img_id in chunk_ids}
            
            # Extract features in parallel
            chunk_features = self._extract_features_parallel(chunk_images)
            all_features.update(chunk_features)
            
            # Force garbage collection after each chunk
            import gc
            gc.collect()
        
        # Convert to pandas DataFrame for easier handling
        features_df = self._convert_to_dataframe(all_features)
        
        # Save features to CSV
        output_path = self.output_dir / "features.csv"
        features_df.to_csv(output_path, index=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Extracted features from {len(images)} images in {elapsed_time:.2f} seconds")
        return features_df
    
    def _extract_features_parallel(self, images):
        """Extract features from images in parallel"""
        features_dict = {}
        
        # Extract basic and shape features using processes (CPU-bound)
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {}
            for image_id, image in images.items():
                # Generate a content hash for caching
                image_hash = hashlib.md5(image.tobytes()).hexdigest()
                future = executor.submit(
                    self._extract_basic_and_shape_features, 
                    image, 
                    image_id, 
                    image_hash
                )
                futures[future] = image_id
            
            for future in as_completed(futures):
                image_id = futures[future]
                try:
                    basic_shape_features = future.result()
                    if image_id not in features_dict:
                        features_dict[image_id] = {}
                    features_dict[image_id].update(basic_shape_features)
                except Exception as e:
                    logger.error(f"Error extracting basic/shape features for {image_id}: {e}")
        
        # Extract texture features separately (CPU-bound but different parameters)
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {}
            for image_id, image in images.items():
                image_hash = hashlib.md5(image.tobytes()).hexdigest()
                future = executor.submit(
                    self._extract_texture_features_cached, 
                    image,
                    image_hash
                )
                futures[future] = image_id
            
            for future in as_completed(futures):
                image_id = futures[future]
                try:
                    texture_features = future.result()
                    if image_id not in features_dict:
                        features_dict[image_id] = {}
                    features_dict[image_id]['texture'] = texture_features
                except Exception as e:
                    logger.error(f"Error extracting texture features for {image_id}: {e}")
        
        # Extract deep learning features (can be GPU-bound)
        if self.is_method_enabled('deep_features') and hasattr(self, 'deep_model'):
            deep_features = self._extract_deep_features_batch(images)
            
            # Add deep features to the dictionary
            for image_id, feats in deep_features.items():
                if image_id in features_dict:
                    features_dict[image_id]['deep'] = feats
        
        return features_dict
    
    def _extract_basic_and_shape_features(self, image, image_id, image_hash):
        """Extract basic morphometry and shape features"""
        result = {}
        
        # Cache key based on image content and enabled methods
        cache_key = f"{image_hash}_basic_shape_{self.is_method_enabled('basic_morphometry')}_{self.is_method_enabled('shape_descriptors')}"
        
        @self.memory.cache
        def cached_extraction(key):
            features = {}
            
            # Basic morphometric features
            if self.is_method_enabled('basic_morphometry'):
                try:
                    features['basic'] = self._extract_basic_features(image)
                except Exception as e:
                    logger.error(f"Error extracting basic features: {e}")
                    features['basic'] = {'error': str(e)}
            
            # Shape descriptors
            if self.is_method_enabled('shape_descriptors'):
                try:
                    features['shape'] = self._extract_shape_features(image)
                except Exception as e:
                    logger.error(f"Error extracting shape features: {e}")
                    features['shape'] = {'error': str(e)}
                    
            return features
        
        try:
            return cached_extraction(cache_key)
        except Exception as e:
            logger.error(f"Cache error for {image_id}: {e}")
            # Fall back to direct computation if caching fails
            features = {}
            if self.is_method_enabled('basic_morphometry'):
                features['basic'] = self._extract_basic_features(image)
            if self.is_method_enabled('shape_descriptors'):
                features['shape'] = self._extract_shape_features(image)
            return features
    
    def _extract_texture_features_cached(self, image, image_hash):
        """Cached version of texture feature extraction"""
        # Cache key based on image content and GLCM parameters
        distances = self.get_method_param('texture_analysis', 'glcm_distances', [1])
        angles = self.get_method_param('texture_analysis', 'glcm_angles', [0])
        cache_key = f"{image_hash}_texture_{distances}_{angles}"
        
        @self.memory.cache
        def cached_texture_extraction(key):
            return self._extract_texture_features(image)
        
        try:
            return cached_texture_extraction(cache_key)
        except Exception as e:
            logger.error(f"Texture cache error: {e}")
            # Fall back to direct computation
            return self._extract_texture_features(image)
    
    def _check_gpu(self):
        """Check for GPU availability"""
        # Check TensorFlow GPU
        tf_gpu = False
        try:
            tf_gpu = len(tf.config.list_physical_devices('GPU')) > 0
            if tf_gpu:
                logger.info("TensorFlow GPU available")
        except:
            pass
        
        # Check PyTorch GPU
        torch_gpu = False
        try:
            torch_gpu = torch.cuda.is_available()
            if torch_gpu:
                logger.info("PyTorch GPU available")
        except:
            pass
        
        return tf_gpu or torch_gpu
    
    def _convert_to_dataframe(self, features_dict):
        """Convert dictionary of features to pandas DataFrame"""
        # Flatten nested dictionaries
        flat_dict = {}
        for image_id, features in features_dict.items():
            flat_features = {}
            for category, category_features in features.items():
                if isinstance(category_features, dict):
                    for feature_name, feature_value in category_features.items():
                        flat_name = f"{category}_{feature_name}"
                        flat_features[flat_name] = feature_value
                else:
                    flat_features[category] = category_features
            flat_dict[image_id] = flat_features
        
        df = pd.DataFrame.from_dict(flat_dict, orient='index')
        
        # Fill missing values
        df = df.fillna(0)
        
        return df
    
    def _extract_image_features(self, image):
        """Extract all configured features from a single image"""
        features = {}
        
        # Basic morphometric features
        if self.is_method_enabled('basic_morphometry'):
            features['basic'] = self._extract_basic_features(image)
        
        # Texture features
        if self.is_method_enabled('texture_analysis'):
            features['texture'] = self._extract_texture_features(image)
        
        # Shape descriptors
        if self.is_method_enabled('shape_descriptors'):
            features['shape'] = self._extract_shape_features(image)
        
        # Deep learning features
        if self.is_method_enabled('deep_features'):
            features['deep'] = self._extract_deep_features(image)
        
        return features
    
    def is_method_enabled(self, method_name):
        """Check if a feature extraction method is enabled in config"""
        for method in self.feature_config['methods']:
            if method['name'] == method_name and method.get('enable', True):
                return True
        return False
    
    def _extract_basic_features(self, image):
        """Extract basic morphometric features"""
        features = {}
        
        # Create binary mask for region properties calculation
        if image.dtype == np.float64:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        _, binary = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Statistical features - use optimized numpy operations
        features['mean'] = float(np.mean(image))
        features['std'] = float(np.std(image))
        features['min'] = float(np.min(image))
        features['max'] = float(np.max(image))
        features['median'] = float(np.median(image))
        features['sum'] = float(np.sum(image))
        
        # Histogram features
        hist = cv2.calcHist([image_uint8], [0], None, [256], [0, 256])
        features['hist_peak'] = float(np.argmax(hist))
        # Calculate entropy directly using logarithm properties
        hist_norm = hist / np.sum(hist)
        hist_norm = hist_norm[hist_norm > 0]  # Only consider non-zero probabilities
        features['hist_entropy'] = float(-np.sum(hist_norm * np.log2(hist_norm)))
        
        # Calculate gradient magnitude - use Sobel for better performance
        sobel_x = cv2.Sobel(image_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image_uint8, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        features['gradient_mean'] = float(np.mean(gradient_magnitude))
        features['gradient_std'] = float(np.std(gradient_magnitude))
        
        # Region properties - use scikit-image regionprops for efficiency
        if np.any(binary):
            labeled_image = ndimage.label(binary)[0]
            regions = regionprops(labeled_image, intensity_image=image)
            
            if regions:
                region = regions[0]  # Assuming largest region
                features['area'] = float(region.area)
                features['perimeter'] = float(region.perimeter)
                features['eccentricity'] = float(region.eccentricity)
                features['solidity'] = float(region.solidity)
                features['euler_number'] = float(region.euler_number)
                features['extent'] = float(region.extent)
                features['moments_hu'] = float(region.moments_hu[0])  # Just first Hu moment for simplicity
        
        return features
    
    def _extract_texture_features(self, image):
        """Extract texture features (GLCM, LBP)"""
        features = {}
        
        # Normalize to 8 bits if needed
        if image.dtype == np.float64:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        # Get GLCM distances and angles from config
        distances = self.get_method_param('texture_analysis', 'glcm_distances', [1])
        angles = self.get_method_param('texture_analysis', 'glcm_angles', [0])
        
        # Convert angles to radians
        angles_rad = [a * np.pi / 180 for a in angles]
        
        # Quantize image to fewer levels to save computation
        levels = 16
        image_quant = np.round(image_uint8 / 255 * (levels - 1)).astype(np.uint8)
        
        # Calculate GLCM - use optimized parameters
        try:
            glcm = graycomatrix(
                image_quant, 
                distances=distances, 
                angles=angles_rad, 
                levels=levels, 
                symmetric=True, 
                normed=True
            )
            
            # Calculate GLCM properties
            props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
            for prop in props:
                values = graycoprops(glcm, prop).flatten()
                for i, val in enumerate(values):
                    features[f'glcm_{prop}_{i}'] = float(val)
        except Exception as e:
            logger.warning(f"GLCM calculation failed: {e}")
            # Add placeholder values
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
                features[f'glcm_{prop}_0'] = 0.0
        
        # Local Binary Pattern features - optimize parameters for speed
        try:
            radius = 2  # Smaller radius is faster
            n_points = 8  # Fewer points is faster
            lbp = local_binary_pattern(
                image_uint8, 
                n_points, 
                radius, 
                method='uniform'
            )
            
            # Use numpy for faster histogram computation
            hist, _ = np.histogram(
                lbp.ravel(), 
                bins=n_points + 2, 
                range=(0, n_points + 2), 
                density=True
            )
            
            # Add LBP histogram features
            for i, val in enumerate(hist):
                features[f'lbp_hist_{i}'] = float(val)
        except Exception as e:
            logger.warning(f"LBP calculation failed: {e}")
            # Add placeholder values
            for i in range(10):  # Reasonable number of LBP features
                features[f'lbp_hist_{i}'] = 0.0
        
        return features
    
    def _extract_shape_features(self, image):
        """Extract shape descriptor features"""
        features = {}
        
        # Convert to binary for shape analysis
        if image.dtype == np.float64:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        _, binary = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # If no object detected, return empty features
        if not np.any(binary):
            return {"no_object_detected": 1.0}
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {"no_contours_found": 1.0}
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Shape features from contour
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        features['area'] = float(area)
        features['perimeter'] = float(perimeter)
        
        # Circularity
        if perimeter > 0:
            features['circularity'] = float(4 * np.pi * area / (perimeter ** 2))
        else:
            features['circularity'] = 0.0
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        features['aspect_ratio'] = float(w) / h if h > 0 else 0.0
        features['extent'] = float(area) / (w * h) if w * h > 0 else 0.0
        
        # Minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        features['equiv_diameter'] = float(np.sqrt(4 * area / np.pi))
        
        if radius > 0:
            features['roundness'] = float(area / (np.pi * radius ** 2))
        else:
            features['roundness'] = 0.0
        
        # Fit ellipse if enough points
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            features['ellipticity'] = float(minor_axis / major_axis) if major_axis > 0 else 0.0
            features['orientation'] = float(ellipse[2])
        else:
            features['ellipticity'] = 0.0
            features['orientation'] = 0.0
        
        # Convex hull features
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        features['convexity'] = float(area / hull_area) if hull_area > 0 else 0.0
        
        # Minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        features['min_enclosing_circle_radius'] = float(radius)
        
        # Skeleton features - using scikit-image for better performance
        try:
            skeleton = morphology.skeletonize(binary > 0)
            features['skeleton_length'] = float(np.sum(skeleton))
        except:
            features['skeleton_length'] = 0.0
        
        # Moments and Hu moments
        moments = cv2.moments(largest_contour)
        features['moment_m00'] = float(moments['m00']) if 'm00' in moments else 0.0
        
        try:
            hu_moments = cv2.HuMoments(moments)
            for i, moment in enumerate(hu_moments.flatten()):
                # Log transform to handle small values
                features[f'hu_moment_{i+1}'] = float(-np.sign(moment) * np.log10(abs(moment) + 1e-10))
        except:
            for i in range(7):  # 7 Hu moments
                features[f'hu_moment_{i+1}'] = 0.0
        
        return features
    
    def _initialize_deep_model(self):
        """Initialize deep learning model for feature extraction"""
        model_name = self.get_method_param('deep_features', 'model', 'resnet50')
        
        try:
            if 'resnet' in model_name:
                if self.gpu_available:
                    # Try to use mixed precision for acceleration
                    try:
                        policy = tf.keras.mixed_precision.Policy('mixed_float16')
                        tf.keras.mixed_precision.set_global_policy(policy)
                        logger.info("Using mixed precision for TensorFlow model")
                    except:
                        logger.warning("Mixed precision not available")
                
                # Initialize TensorFlow model
                base_model = tf.keras.applications.ResNet50(
                    include_top=False,
                    weights='imagenet',
                    pooling='avg'
                )
                self.deep_model = base_model
                self.deep_model_framework = 'tensorflow'
                logger.info(f"Initialized {model_name} model for deep feature extraction")
                
            elif 'vgg' in model_name:
                # Initialize PyTorch model
                if 'vgg16' in model_name:
                    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
                else:
                    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
                
                # Remove classifier to get features
                self.deep_model = torch.nn.Sequential(*list(model.children())[:-1])
                
                # Move model to GPU if available
                if self.gpu_available and torch.cuda.is_available():
                    self.deep_model = self.deep_model.cuda()
                
                self.deep_model.eval()
                self.deep_model_framework = 'pytorch'
                logger.info(f"Initialized {model_name} model for deep feature extraction")
                
            else:
                logger.warning(f"Unsupported model: {model_name}, defaulting to ResNet50")
                base_model = tf.keras.applications.ResNet50(
                    include_top=False,
                    weights='imagenet',
                    pooling='avg'
                )
                self.deep_model = base_model
                self.deep_model_framework = 'tensorflow'
                
        except Exception as e:
            logger.error(f"Failed to initialize deep model: {e}")
            self.deep_model = None
            self.deep_model_framework = None
    
    def _extract_deep_features_batch(self, images):
        """Extract deep features from multiple images in an optimized batch"""
        if not hasattr(self, 'deep_model') or self.deep_model is None:
            logger.error("Deep model not initialized")
            return {img_id: {"model_not_initialized": 1.0} for img_id in images}
        
        features_dict = {}
        batch_size = 16  # Optimal batch size for GPU processing
        
        try:
            # Prepare all images first
            prepared_images = {}
            for image_id, image in images.items():
                try:
                    # Preprocess image for deep feature extraction
                    if len(image.shape) == 2:
                        image_rgb = np.stack([image, image, image], axis=2)
                    else:
                        image_rgb = image
                    
                    # Resize to model input size
                    image_resized = cv2.resize(image_rgb, (224, 224))
                    prepared_images[image_id] = image_resized
                except Exception as e:
                    logger.error(f"Error preprocessing image for deep features: {e}")
                    features_dict[image_id] = {"preprocessing_error": 1.0}
            
            # Create batches
            image_ids = list(prepared_images.keys())
            for i in range(0, len(image_ids), batch_size):
                batch_ids = image_ids[i:i+batch_size]
                batch_images = [prepared_images[img_id] for img_id in batch_ids]
                
                if self.deep_model_framework == 'tensorflow':
                    # Convert to tensor and preprocess
                    x = np.stack(batch_images)
                    x = tf.keras.applications.resnet50.preprocess_input(x)
                    
                    # Get features
                    batch_features = self.deep_model.predict(x, verbose=0)
                    
                    # Process each result
                    for j, img_id in enumerate(batch_ids):
                        deep_features = {}
                        feats = batch_features[j]
                        
                        # Store top features
                        max_features = 100  # Limit number of features to prevent overflow
                        for k, val in enumerate(feats[:max_features]):
                            deep_features[f'deep_{k:03d}'] = float(val)
                        
                        deep_features['deep_mean'] = float(np.mean(feats))
                        deep_features['deep_std'] = float(np.std(feats))
                        features_dict[img_id] = deep_features
                
                elif self.deep_model_framework == 'pytorch':
                    # Convert to tensor and preprocess
                    x = np.stack(batch_images)
                    x = torch.tensor(np.transpose(x, (0, 3, 1, 2)).astype(np.float32) / 255.0)
                    
                    # Move to GPU if available
                    if self.gpu_available and torch.cuda.is_available():
                        x = x.cuda()
                    
                    # Get features
                    with torch.no_grad():
                        batch_features = self.deep_model(x)
                    
                    # Move back to CPU and convert to numpy
                    if self.gpu_available and torch.cuda.is_available():
                        batch_features = batch_features.cpu()
                    batch_features = batch_features.numpy()
                    
                    # Process each result
                    for j, img_id in enumerate(batch_ids):
                        deep_features = {}
                        feats = batch_features[j].flatten()
                        
                        # Store top features
                        max_features = 100  # Limit number of features
                        for k, val in enumerate(feats[:max_features]):
                            deep_features[f'deep_{k:03d}'] = float(val)
                        
                        deep_features['deep_mean'] = float(np.mean(feats))
                        deep_features['deep_std'] = float(np.std(feats))
                        features_dict[img_id] = deep_features
                
        except Exception as e:
            logger.error(f"Error extracting deep features in batch: {e}")
            for image_id in images.keys():
                if image_id not in features_dict:
                    features_dict[image_id] = {"batch_extraction_error": 1.0}
        
        return features_dict
    
    def _extract_deep_features(self, image):
        """Extract deep features using pre-trained CNN (single image)"""
        features = {}
        
        if not hasattr(self, 'deep_model') or self.deep_model is None:
            logger.error("Deep model not initialized")
            return {"model_not_initialized": 1.0}
        
        try:
            # Preprocess image for deep feature extraction
            # Convert to RGB if grayscale
            if len(image.shape) == 2:
                image_rgb = np.stack([image, image, image], axis=2)
            else:
                image_rgb = image
            
            # Resize to model input size
            image_resized = cv2.resize(image_rgb, (224, 224))
            
            if self.deep_model_framework == 'tensorflow':
                # TensorFlow preprocessing
                x = tf.keras.applications.resnet50.preprocess_input(
                    np.expand_dims(image_resized, axis=0).astype(np.float32)
                )
                
                # Get features
                deep_features = self.deep_model.predict(x, verbose=0)
                deep_features = deep_features.flatten()
                
            elif self.deep_model_framework == 'pytorch':
                # PyTorch preprocessing
                x = torch.tensor(np.transpose(image_resized, (2, 0, 1)).astype(np.float32) / 255.0)
                x = x.unsqueeze(0)
                
                # Move to GPU if available
                if self.gpu_available and torch.cuda.is_available():
                    x = x.cuda()
                
                # Get features
                with torch.no_grad():
                    deep_features = self.deep_model(x)
                
                # Move back to CPU and convert to numpy
                if self.gpu_available and torch.cuda.is_available():
                    deep_features = deep_features.cpu()
                deep_features = deep_features.numpy().flatten()
            
            # Store top features
            max_features = 100  # Limit number of features to prevent overflow
            for i, val in enumerate(deep_features[:max_features]):
                features[f'deep_{i:03d}'] = float(val)
            
            features['deep_mean'] = float(np.mean(deep_features))
            features['deep_std'] = float(np.std(deep_features))
            
        except Exception as e:
            logger.error(f"Error extracting deep features: {e}")
            features["deep_feature_error"] = 1.0
        
        return features
    
    def get_method_param(self, method_name, param_name, default=None):
        """Get parameter from config for a specific method"""
        for method in self.feature_config['methods']:
            if method['name'] == method_name and method.get('enable', False):
                return method.get(param_name, default)
        return default