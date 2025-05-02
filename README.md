# MorphoMetrica: Advanced Morphometric Analysis Platform

MorphoMetrica is a comprehensive Python-based platform that extracts, analyzes, and classifies morphological features from images, leveraging NumPy, OpenCV, scikit-image, TensorFlow and PyTorch to bridge research and clinical applications through advanced image processing, machine learning, and automated reporting capabilities.

## Overview

MorphoMetrica provides a full pipeline for morphometric analysis of biological and medical images, from preprocessing and feature extraction through model training, evaluation, and clinical reporting. The platform is designed to be both powerful and flexible, supporting various image formats, feature extraction techniques, and machine learning approaches.

## Features

- **Versatile Image Preprocessing**
  - Support for multiple medical and scientific image formats (DICOM, NIfTI, standard formats)
  - Automated segmentation and normalization
  - Parallel processing for large datasets
  - Image processing caching for performance

- **Comprehensive Feature Extraction**
  - Basic statistical metrics
  - Advanced texture analysis (GLCM, LBP)
  - Detailed shape descriptors
  - Deep learning-based features using pre-trained models
  - GPU-accelerated deep feature extraction

- **Flexible Machine Learning Pipeline**
  - Multiple classification algorithms
  - Ensemble methods
  - Hyperparameter optimization
  - GPU acceleration for neural networks
  - Feature selection for dimensionality reduction

- **Robust Evaluation Framework**
  - Cross-validation
  - Comprehensive performance metrics
  - Parallel metrics calculation
  - Confusion matrices and ROC curves

- **Clinical Reporting System**
  - Detailed reports for research applications
  - Clinical-oriented reporting
  - Prediction reports for new data
  - Interactive HTML reports with visualizations

## Installation

### Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for deep learning features)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/morphometrica.git
   cd morphometrica
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Quickstart

### Basic Usage

```bash
# Run the full pipeline (preprocessing, feature extraction, model training, evaluation, visualization, and reporting)
python -m src.main --config configs/config.yaml --data_dir path/to/images --output_dir path/to/results

# Just extract features from images
python -m src.main --mode extract_features --data_dir path/to/images --output_dir path/to/results

# Train a model using existing features
python -m src.main --mode train --data_dir path/to/features.csv --output_dir path/to/results

# Make predictions with an existing model
python -m src.main --mode predict --model_path models/your_model.pkl --data_dir path/to/images

# Generate a clinical report
python -m src.main --mode report --model_path models/your_model.pkl --data_dir path/to/images
```

### Performance Optimization

MorphoMetrica includes various acceleration tweaks for improved performance:

```bash
# Use specific number of worker processes
python -m src.main --workers 8 --data_dir path/to/images

# Enable GPU acceleration explicitly
python -m src.main --gpu --data_dir path/to/images

# Disable caching for fresh results
python -m src.main --no-cache --data_dir path/to/images

# Enable performance profiling
python -m src.main --profile --data_dir path/to/images

# Extract only specific feature groups
python -m src.main --mode extract_features --features basic,shape --data_dir path/to/images
```

## Project Structure

```
morphometry/
├── configs/           # Configuration files
│   └── config.yaml    # Main configuration
├── data/              # Data directory
│   ├── processed/     # Processed images
│   └── raw/           # Raw input images
├── results/           # Results directory
│   ├── figures/       # Generated visualizations
│   └── reports/       # Generated reports
├── src/               # Source code
│   ├── clinical/      # Clinical reporting modules
│   ├── evaluation/    # Evaluation modules
│   ├── feature_extraction/ # Feature extraction modules
│   ├── models/        # Machine learning models
│   ├── preprocessing/ # Image preprocessing modules
│   ├── visualization/ # Data visualization modules
│   └── main.py        # Main entry point
└── tests/             # Unit tests
```

## Configuration

MorphoMetrica is configured through a YAML file. The default configuration is in `configs/config.yaml`. You can customize:

- Preprocessing parameters
- Feature extraction methods
- Model type and hyperparameters
- Evaluation metrics
- Visualization options
- Clinical reporting settings

Example configuration:

```yaml
# Preprocessing parameters
preprocessing:
  resize:
    enable: true
    height: 512
    width: 512
  normalization:
    enable: true
    method: "z-score"  # Options: z-score, min-max, histogram_equalization

# Feature extraction parameters
feature_extraction:
  methods:
    - name: "basic_morphometry"
      enable: true
    - name: "texture_analysis"
      enable: true
    - name: "shape_descriptors"
      enable: true
    - name: "deep_features"
      enable: true
      model: "resnet50"

# Model parameters
model:
  type: "ensemble"  # Options: cnn, random_forest, gradient_boosting, svm, ensemble
  batch_size: 32
```

## Use Cases

### Research Applications

- Cell morphology analysis
- Histopathology feature extraction
- Tissue classification
- Morphological phenotyping

### Clinical Applications

- Support for diagnostic workflows
- Standardized morphometric assessment
- Quantitative feature extraction
- Automated preliminary classification
- Clinical report generation

## Performance Optimizations

MorphoMetrica includes multiple optimizations for improved performance:

- **Parallel Processing**: Utilize multiple CPU cores for image processing and feature extraction
- **GPU Acceleration**: Leverage GPU for deep feature extraction and neural network training
- **Caching**: Store intermediate results to avoid redundant computation
- **Batched Processing**: Process images in batches to manage memory efficiently
- **Mixed Precision Training**: Use 16-bit floating point where supported for faster neural network training
- **Memory Management**: Automatic cleanup to prevent memory leaks with large datasets
