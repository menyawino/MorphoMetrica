"""
Feature extraction module for morphometry analysis.
This module contains methods to extract morphometric features from images.
"""

import numpy as np
import pandas as pd
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops, regionprops_table
from scipy import ndimage
import logging
import tensorflow as tf
import torch
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract morphometric features from preprocessed images"""
    
    def __init__(self, config):
        """Initialize the feature extractor with configuration"""
        self.config = config
        self.feature_config = config['feature_extraction']
        self.output_dir = Path(config['data']['output_dir'])
        
        # Initialize deep learning model for feature extraction if needed
        if self.is_method_enabled('deep_features'):
            self._initialize_deep_model()
    
    def extract_features(self, images):
        """Extract features from a dictionary of preprocessed images"""
        logger.info("Extracting features from processed images")
        
        features_dict = {}
        for image_id, image in images.items():
            try:
                image_features = self._extract_image_features(image)
                features_dict[image_id] = image_features
            except Exception as e:
                logger.error(f"Error extracting features for {image_id}: {e}")
        
        # Convert to pandas DataFrame for easier handling
        features_df = self._convert_to_dataframe(features_dict)
        
        # Save features to CSV
        output_path = self.output_dir / "features.csv"
        features_df.to_csv(output_path, index=True)
        
        logger.info(f"Extracted features from {len(features_dict)} images")
        return features_df
    
    def _convert_to_dataframe(self, features_dict):
        """Convert dictionary of features to pandas DataFrame"""
        # Flatten nested dictionaries
        flat_dict = {}
        for image_id, features in features_dict.items():
            flat_features = {}
            for category, category_features in features.items():
                for feature_name, feature_value in category_features.items():
                    flat_name = f"{category}_{feature_name}"
                    flat_features[flat_name] = feature_value
            flat_dict[image_id] = flat_features
        
        return pd.DataFrame.from_dict(flat_dict, orient='index')
    
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
            if method['name'] == method_name and method['enable']:
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
        
        # Statistical features
        features['mean'] = np.mean(image)
        features['std'] = np.std(image)
        features['min'] = np.min(image)
        features['max'] = np.max(image)
        features['median'] = np.median(image)
        features['sum'] = np.sum(image)
        
        # Histogram features
        hist = cv2.calcHist([image_uint8], [0], None, [256], [0, 256])
        features['hist_peak'] = np.argmax(hist)
        features['hist_entropy'] = -np.sum(hist * np.log2(hist + 1e-7))
        
        # Calculate gradient magnitude
        sobel_x = cv2.Sobel(image_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image_uint8, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        features['gradient_mean'] = np.mean(gradient_magnitude)
        features['gradient_std'] = np.std(gradient_magnitude)
        
        # Region properties
        if np.any(binary):
            labeled_image = ndimage.label(binary)[0]
            regions = regionprops(labeled_image, intensity_image=image)
            
            if regions:
                region = regions[0]  # Assuming largest region
                features['area'] = region.area
                features['perimeter'] = region.perimeter
                features['eccentricity'] = region.eccentricity
                features['solidity'] = region.solidity
                features['euler_number'] = region.euler_number
                features['extent'] = region.extent
                features['moments_hu'] = region.moments_hu[0]  # Just first Hu moment for simplicity
        
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
        
        # Calculate GLCM
        glcm = graycomatrix(image_quant, distances=distances, angles=angles_rad, 
                           levels=levels, symmetric=True, normed=True)
        
        # Calculate GLCM properties
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        for prop in props:
            values = graycoprops(glcm, prop).flatten()
            for i, val in enumerate(values):
                features[f'glcm_{prop}_{i}'] = val
        
        # Local Binary Pattern features
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(image_uint8, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2), density=True)
        
        # Add LBP histogram features
        for i, val in enumerate(hist):
            features[f'lbp_hist_{i}'] = val
        
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
            return {"no_object_detected": 1}
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {"no_contours_found": 1}
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Shape features from contour
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        features['area'] = area
        features['perimeter'] = perimeter
        
        # Circularity
        if perimeter > 0:
            features['circularity'] = 4 * np.pi * area / (perimeter ** 2)
        else:
            features['circularity'] = 0
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        features['aspect_ratio'] = float(w) / h if h > 0 else 0
        features['extent'] = float(area) / (w * h) if w * h > 0 else 0
        
        # Minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        features['equiv_diameter'] = np.sqrt(4 * area / np.pi)
        
        if radius > 0:
            features['roundness'] = area / (np.pi * radius ** 2)
        else:
            features['roundness'] = 0
        
        # Fit ellipse if enough points
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            features['ellipticity'] = minor_axis / major_axis if major_axis > 0 else 0
            features['orientation'] = ellipse[2]
        
        # Convex hull features
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        features['convexity'] = area / hull_area if hull_area > 0 else 0
        
        # Minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        features['min_enclosing_circle_radius'] = radius
        
        # Skeleton features
        skeleton = morphology.skeletonize(binary > 0)
        features['skeleton_length'] = np.sum(skeleton)
        
        # Moments and Hu moments
        moments = cv2.moments(largest_contour)
        features['moment_m00'] = moments['m00']
        
        hu_moments = cv2.HuMoments(moments)
        for i, moment in enumerate(hu_moments.flatten()):
            # Log transform to handle small values
            features[f'hu_moment_{i+1}'] = -np.sign(moment) * np.log10(abs(moment) + 1e-10)
        
        return features
    
    def _initialize_deep_model(self):
        """Initialize deep learning model for feature extraction"""
        model_name = self.get_method_param('deep_features', 'model', 'resnet50')
        
        try:
            if 'resnet' in model_name:
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
    
    def _extract_deep_features(self, image):
        """Extract deep features using pre-trained CNN"""
        features = {}
        
        if not hasattr(self, 'deep_model') or self.deep_model is None:
            logger.error("Deep model not initialized")
            return {"model_not_initialized": 1}
        
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
                deep_features = self.deep_model.predict(x)
                deep_features = deep_features.flatten()
                
            elif self.deep_model_framework == 'pytorch':
                # PyTorch preprocessing
                x = torch.tensor(np.transpose(image_resized, (2, 0, 1)).astype(np.float32) / 255.0)
                x = x.unsqueeze(0)
                
                # Get features
                with torch.no_grad():
                    deep_features = self.deep_model(x)
                deep_features = deep_features.numpy().flatten()
            
            # Store top features
            max_features = 100  # Limit number of features to prevent overflow
            for i, val in enumerate(deep_features[:max_features]):
                features[f'deep_{i:03d}'] = float(val)
            
            features['deep_mean'] = float(np.mean(deep_features))
            features['deep_std'] = float(np.std(deep_features))
            
        except Exception as e:
            logger.error(f"Error extracting deep features: {e}")
            features["deep_feature_error"] = 1
        
        return features
    
    def get_method_param(self, method_name, param_name, default=None):
        """Get parameter from config for a specific method"""
        for method in self.feature_config['methods']:
            if method['name'] == method_name and method.get('enable', False):
                return method.get(param_name, default)
        return default