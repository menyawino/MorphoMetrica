"""
Image preprocessing module for the morphometry tool.
Handles various preprocessing operations like resizing, normalization, filtering, etc.
"""

import os
import numpy as np
import cv2
import SimpleITK as sitk
import pydicom
from pathlib import Path
from skimage import exposure, filters, morphology, segmentation
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
from joblib import Memory
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageProcessor:
    """Class for preprocessing images for morphometric analysis"""
    
    def __init__(self, config):
        """Initialize the image processor with configuration"""
        self.config = config
        self.preprocessing_config = config['preprocessing']
        self.processed_dir = Path(config['data']['processed_dir'])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup cache for processed images
        cache_dir = self.processed_dir / '.cache'
        self.memory = Memory(cache_dir, verbose=0)
        self.cached_process_image = self.memory.cache(self._process_image_impl)
        
        # Number of workers for parallel processing
        self.n_workers = os.cpu_count() or 1
        # Use fewer workers if there's not much RAM available
        if self.n_workers > 4:
            self.n_workers = min(self.n_workers, 8)  # Limit workers to avoid memory issues
    
    def process_directory(self, input_dir):
        """Process all images in a directory using parallel execution"""
        input_dir = Path(input_dir)
        logger.info(f"Processing images from {input_dir} with {self.n_workers} workers")
        
        image_files = self._get_image_files(input_dir)
        total_files = len(image_files)
        logger.info(f"Found {total_files} images to process")
        
        processed_images = {}
        
        # Use batched processing to avoid memory issues with large datasets
        batch_size = 50  # Adjust based on available memory
        image_batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
        
        for batch_idx, batch in enumerate(image_batches):
            logger.info(f"Processing batch {batch_idx+1}/{len(image_batches)} ({len(batch)} images)")
            batch_results = {}
            
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                # Submit all jobs
                future_to_path = {
                    executor.submit(self._process_single_image, file_path): file_path 
                    for file_path in batch
                }
                
                # Process results as they complete
                for future in as_completed(future_to_path):
                    file_path = future_to_path[future]
                    try:
                        image_id, processed_image = future.result()
                        batch_results[image_id] = processed_image
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
            
            processed_images.update(batch_results)
            
            # Explicit garbage collection after each batch
            import gc
            gc.collect()
        
        logger.info(f"Processed {len(processed_images)} images")
        return processed_images
    
    def _process_single_image(self, file_path):
        """Process a single image file (designed for parallel execution)"""
        try:
            image_id = file_path.stem
            image = self.load_image(file_path)
            
            # Generate cache key based on image content and config
            config_str = str(self.preprocessing_config)
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            cache_key = f"{image_hash}_{config_str}"
            
            processed_image = self.cached_process_image(image, image_id, cache_key)
            
            # Save processed image
            output_path = self.processed_dir / f"{image_id}.npy"
            np.save(output_path, processed_image)
            
            return image_id, processed_image
            
        except Exception as e:
            logger.error(f"Error in _process_single_image for {file_path}: {str(e)}")
            raise
    
    def _get_image_files(self, directory):
        """Get all image files in directory and subdirectories"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dcm', '.nii', '.nii.gz']
        
        image_files = []
        for ext in valid_extensions:
            image_files.extend(Path(directory).glob(f"**/*{ext}"))
        
        return image_files
    
    def load_image(self, file_path):
        """Load an image based on its format"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension in ['.dcm']:
            # Handle DICOM files
            ds = pydicom.dcmread(str(file_path))
            image = ds.pixel_array
            
            # Apply appropriate windowing if available
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                window_center = ds.WindowCenter
                window_width = ds.WindowWidth
                if isinstance(window_center, pydicom.multival.MultiValue):
                    window_center = window_center[0]
                if isinstance(window_width, pydicom.multival.MultiValue):
                    window_width = window_width[0]
                
                image = self._apply_windowing(image, window_center, window_width)
            
        elif extension in ['.nii', '.nii.gz']:
            # Handle NIfTI files using SimpleITK
            img = sitk.ReadImage(str(file_path))
            image = sitk.GetArrayFromImage(img)
            
            # If 3D volume, take middle slice by default
            if len(image.shape) == 3:
                image = image[image.shape[0] // 2, :, :]
                
        else:
            # Handle standard image formats - use IMREAD_UNCHANGED for best compatibility
            image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Failed to load image: {file_path}")
            
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _apply_windowing(self, image, window_center, window_width):
        """Apply windowing to a DICOM image"""
        lower = window_center - window_width / 2
        upper = window_center + window_width / 2
        image = np.clip(image, lower, upper)
        image = (image - lower) / (upper - lower)
        image = (image * 255).astype(np.uint8)
        return image
    
    def process_image(self, image, image_id=None):
        """Apply preprocessing steps to an image with caching"""
        # Generate cache key based on image content and config
        config_str = str(self.preprocessing_config)
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        cache_key = f"{image_hash}_{config_str}"
        
        return self.cached_process_image(image, image_id, cache_key)
    
    def _process_image_impl(self, image, image_id=None, cache_key=None):
        """Implementation of image processing (cached version)"""
        # Make a copy to avoid modifying the original
        processed = image.copy()
        
        # Resize if enabled
        if self.preprocessing_config['resize']['enable']:
            height = self.preprocessing_config['resize']['height']
            width = self.preprocessing_config['resize']['width']
            # Use INTER_AREA for downsampling and INTER_CUBIC for upsampling
            if processed.shape[0] > height or processed.shape[1] > width:
                interpolation = cv2.INTER_AREA
            else:
                interpolation = cv2.INTER_CUBIC
                
            processed = cv2.resize(processed, (width, height), interpolation=interpolation)
        
        # Convert to grayscale if it's a color image
        if len(processed.shape) == 3 and processed.shape[2] == 3:
            processed_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        else:
            processed_gray = processed
        
        # Apply segmentation to isolate the object of interest
        try:
            segmented = self._segment_image(processed_gray)
            processed_gray = segmented
        except Exception as e:
            logger.warning(f"Segmentation failed for {image_id}: {e}")
        
        # Apply normalization if enabled
        if self.preprocessing_config['normalization']['enable']:
            method = self.preprocessing_config['normalization']['method']
            processed_gray = self._normalize_image(processed_gray, method)
        
        # Return the image in the appropriate format
        # For feature extraction, often grayscale is sufficient
        return processed_gray
    
    def _segment_image(self, image):
        """
        Segment the image to isolate the object of interest
        Using Otsu thresholding as a simple default method
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours found, return original image
        if not contours:
            return image
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a mask of the largest contour
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        # Apply the mask to the original image
        segmented = cv2.bitwise_and(image, mask)
        
        return segmented
    
    def _normalize_image(self, image, method='z-score'):
        """Normalize the image using the specified method"""
        if method == 'z-score':
            # Z-score normalization
            mean = np.mean(image)
            std = np.std(image)
            if std > 0:
                normalized = (image - mean) / std
            else:
                normalized = image - mean
            
        elif method == 'min-max':
            # Min-max normalization
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                normalized = (image - min_val) / (max_val - min_val)
            else:
                normalized = image
            
        elif method == 'histogram_equalization':
            # Histogram equalization
            normalized = exposure.equalize_hist(image)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized