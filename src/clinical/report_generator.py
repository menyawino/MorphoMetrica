"""
Clinical reporting module for morphometry analysis.
Generates clinical reports from morphometric features and model predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import json
from datetime import datetime
import logging
from jinja2 import Template
import markdown
import base64
from io import BytesIO
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate clinical reports from morphometry analysis results"""
    
    def __init__(self, config):
        """Initialize the report generator with configuration"""
        self.config = config
        self.clinical_config = config.get('clinical', {})
        self.output_dir = Path(config['data']['output_dir'])
        self.reports_dir = self.output_dir / 'reports'
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / 'figures'
        
        # Load reference ranges if available
        self.reference_ranges = self._load_reference_ranges()
    
    def generate_report(self, features, model, evaluation_results):
        """Generate a comprehensive report for research purposes"""
        logger.info("Generating comprehensive analysis report")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"morphometry_report_{timestamp}.html"
        
        # Prepare report content
        report_data = {
            "title": "Morphometric Analysis Report",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "features_summary": self._get_features_summary(features),
            "model_summary": self._get_model_summary(model),
            "evaluation_summary": self._get_evaluation_summary(evaluation_results),
            "figures": self._get_figures()
        }
        
        # Generate HTML report
        html_content = self._render_template(report_data, template_type='research')
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Research report saved to {report_path}")
        return report_path
    
    def generate_prediction_report(self, features, predictions):
        """Generate a report for prediction results"""
        logger.info("Generating prediction report")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"prediction_report_{timestamp}.html"
        
        # Prepare report content
        report_data = {
            "title": "Morphometry Prediction Report",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "features_summary": self._get_features_summary(features),
            "predictions": self._format_predictions(predictions),
        }
        
        # Generate HTML report
        html_content = self._render_template(report_data, template_type='prediction')
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Prediction report saved to {report_path}")
        return report_path
    
    def generate_clinical_report(self, features, predictions, metadata=None):
        """Generate a clinical report suitable for healthcare providers"""
        logger.info("Generating clinical report")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"clinical_report_{timestamp}.html"
        
        # Extract key morphological features
        morphology_summary = self._extract_clinical_features(features)
        
        # Interpret morphological findings
        clinical_interpretation = self._interpret_morphology(features, predictions)
        
        # Prepare report content
        report_data = {
            "title": "Clinical Morphometry Report",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "patient_info": metadata if metadata else {"id": "Unknown", "age": "N/A", "sex": "N/A"},
            "morphology_summary": morphology_summary,
            "clinical_interpretation": clinical_interpretation,
            "predictions": self._format_predictions(predictions, clinical=True),
            "reference_ranges": self.reference_ranges,
            "recommendations": self._generate_recommendations(features, predictions),
            "figures": self._get_clinical_figures(features)
        }
        
        # Generate HTML report
        html_content = self._render_template(report_data, template_type='clinical')
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Clinical report saved to {report_path}")
        return report_path
    
    def _load_reference_ranges(self):
        """Load reference ranges for morphometric features"""
        reference_ranges = {}
        
        # Check if reference ranges should be included
        if self.clinical_config.get('reference_ranges', {}).get('include', False):
            source_path = self.clinical_config['reference_ranges'].get('source', '')
            source_path = Path(source_path)
            
            if source_path.exists():
                try:
                    if source_path.suffix == '.csv':
                        df = pd.read_csv(source_path)
                        reference_ranges = df.set_index('feature').to_dict(orient='index')
                    elif source_path.suffix == '.json':
                        with open(source_path, 'r') as f:
                            reference_ranges = json.load(f)
                    
                    logger.info(f"Loaded reference ranges from {source_path}")
                except Exception as e:
                    logger.error(f"Failed to load reference ranges: {e}")
        
        return reference_ranges
    
    def _get_features_summary(self, features):
        """Summarize extracted features"""
        summary = {}
        
        # Convert features to DataFrame if not already
        if isinstance(features, pd.DataFrame):
            df = features
        else:
            df = pd.DataFrame(features)
        
        # Basic statistics
        summary['count'] = len(df)
        summary['feature_count'] = df.shape[1]
        
        # Group features by type
        feature_groups = {}
        for col in df.columns:
            if '_' in col:
                prefix = col.split('_')[0]
                if prefix not in feature_groups:
                    feature_groups[prefix] = []
                feature_groups[prefix].append(col)
        
        summary['feature_groups'] = feature_groups
        summary['feature_group_counts'] = {k: len(v) for k, v in feature_groups.items()}
        
        return summary
    
    def _get_model_summary(self, model):
        """Summarize model information"""
        summary = {}
        
        # Model type
        if hasattr(model, '__class__'):
            summary['type'] = model.__class__.__name__
        else:
            summary['type'] = 'Unknown'
        
        # For sklearn pipeline, get classifier type
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            summary['classifier'] = model.named_steps['classifier'].__class__.__name__
        
        # For custom ensemble model
        if hasattr(model, 'base_models'):
            summary['ensemble'] = True
            summary['base_models'] = list(model.base_models.keys())
        else:
            summary['ensemble'] = False
        
        return summary
    
    def _get_evaluation_summary(self, evaluation_results):
        """Summarize model evaluation results"""
        if not evaluation_results:
            return {}
        
        summary = {}
        
        # Test set results
        if 'test_set' in evaluation_results:
            test_results = evaluation_results['test_set']
            metrics = {}
            
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'weighted_auc']:
                if metric in test_results:
                    metrics[metric] = float(test_results[metric])
            
            summary['metrics'] = metrics
            
            # Classification report summary
            if 'classification_report' in test_results:
                report = test_results['classification_report']
                class_metrics = {}
                
                for class_name, class_metrics_dict in report.items():
                    if isinstance(class_metrics_dict, dict) and 'precision' in class_metrics_dict:
                        class_metrics[class_name] = class_metrics_dict
                
                summary['class_metrics'] = class_metrics
        
        # Cross-validation results
        if 'cross_validation' in evaluation_results:
            cv_results = evaluation_results['cross_validation']
            cv_summary = {}
            
            for metric, values in cv_results.items():
                if 'mean' in values:
                    cv_summary[metric] = {
                        'mean': float(values['mean']),
                        'std': float(values['std'])
                    }
            
            summary['cross_validation'] = cv_summary
        
        return summary
    
    def _get_figures(self):
        """Get list of generated figures"""
        figures = []
        
        if self.figures_dir.exists():
            for image_path in self.figures_dir.glob('*.png'):
                figures.append({
                    'name': image_path.stem,
                    'path': str(image_path),
                    'title': image_path.stem.replace('_', ' ').title()
                })
        
        return figures
    
    def _format_predictions(self, predictions, clinical=False):
        """Format prediction results"""
        if isinstance(predictions, np.ndarray):
            # Simple array of class labels
            unique_labels = np.unique(predictions)
            counts = {str(label): int(np.sum(predictions == label)) for label in unique_labels}
            
            # Format for report
            formatted = []
            for label, count in counts.items():
                item = {
                    'label': label,
                    'count': count,
                    'percentage': round(count / len(predictions) * 100, 1)
                }
                if clinical:
                    item['interpretation'] = self._get_clinical_interpretation(label)
                formatted.append(item)
            
            return formatted
        else:
            # Dictionary or other format
            return predictions
    
    def _extract_clinical_features(self, features):
        """Extract key morphological features relevant for clinical interpretation"""
        clinical_features = {}
        
        # Convert features to DataFrame if not already
        if isinstance(features, pd.DataFrame):
            df = features
        else:
            df = pd.DataFrame(features)
        
        # Key morphological features and their clinical relevance
        morphology_mapping = {
            'shape_circularity': 'Circularity',
            'shape_ellipticity': 'Ellipticity',
            'shape_convexity': 'Convexity',
            'shape_solidity': 'Solidity',
            'shape_eccentricity': 'Eccentricity',
            'shape_roundness': 'Roundness',
            'shape_area': 'Area',
            'shape_perimeter': 'Perimeter',
            'shape_equivalent_diameter': 'Equivalent Diameter',
            'basic_mean': 'Mean Intensity',
            'basic_std': 'Intensity Variation',
            'texture_glcm_contrast': 'Texture Contrast',
            'texture_glcm_homogeneity': 'Texture Homogeneity'
        }
        
        # Extract available clinical features
        for feature_key, display_name in morphology_mapping.items():
            matching_columns = [col for col in df.columns if feature_key in col.lower()]
            
            if matching_columns:
                # Use the first matching column
                col = matching_columns[0]
                
                # Calculate average if multiple samples
                mean_value = df[col].mean()
                std_value = df[col].std() if len(df) > 1 else None
                
                clinical_features[display_name] = {
                    'value': round(mean_value, 3),
                    'std': round(std_value, 3) if std_value is not None else None,
                    'reference_range': self._get_reference_range(col),
                    'interpretation': self._interpret_feature_value(col, mean_value)
                }
        
        return clinical_features
    
    def _get_reference_range(self, feature):
        """Get reference range for a feature"""
        if not self.reference_ranges or feature not in self.reference_ranges:
            return {"lower": None, "upper": None}
        
        return {
            "lower": self.reference_ranges[feature].get('lower', None),
            "upper": self.reference_ranges[feature].get('upper', None)
        }
    
    def _interpret_feature_value(self, feature, value):
        """Interpret the clinical significance of a feature value"""
        if not self.reference_ranges or feature not in self.reference_ranges:
            return "No reference range available"
        
        ref_range = self.reference_ranges[feature]
        
        if 'lower' in ref_range and 'upper' in ref_range:
            lower = ref_range['lower']
            upper = ref_range['upper']
            
            if value < lower:
                return f"Below normal range ({lower}-{upper})"
            elif value > upper:
                return f"Above normal range ({lower}-{upper})"
            else:
                return f"Within normal range ({lower}-{upper})"
        
        return "No reference range available"
    
    def _interpret_morphology(self, features, predictions):
        """Generate clinical interpretation of morphological findings"""
        # This would be customized based on the specific application
        # Here's a placeholder implementation
        
        # Convert features to DataFrame if not already
        if isinstance(features, pd.DataFrame):
            df = features
        else:
            df = pd.DataFrame(features)
        
        interpretation = {
            "summary": "This is an automated interpretation of the morphometric analysis.",
            "findings": []
        }
        
        # Example morphological interpretation based on shape features
        shape_cols = [col for col in df.columns if 'shape_' in col]
        if shape_cols:
            if 'shape_circularity' in df.columns:
                mean_circ = df['shape_circularity'].mean()
                if mean_circ > 0.85:
                    interpretation["findings"].append(
                        "High circularity observed, suggesting regular, circular structures."
                    )
                elif mean_circ < 0.6:
                    interpretation["findings"].append(
                        "Low circularity observed, suggesting irregular or elongated structures."
                    )
            
            if 'shape_convexity' in df.columns:
                mean_conv = df['shape_convexity'].mean()
                if mean_conv < 0.8:
                    interpretation["findings"].append(
                        "Low convexity observed, indicating complex morphology with potential invaginations."
                    )
        
        # Example interpretation based on texture features
        texture_cols = [col for col in df.columns if 'texture_' in col]
        if texture_cols:
            if any('glcm_contrast' in col for col in texture_cols):
                contrast_cols = [col for col in texture_cols if 'glcm_contrast' in col]
                mean_contrast = df[contrast_cols].values.mean()
                
                if mean_contrast > 0.8:  # Arbitrary threshold
                    interpretation["findings"].append(
                        "High contrast texture observed, suggesting heterogeneous composition."
                    )
                elif mean_contrast < 0.2:  # Arbitrary threshold
                    interpretation["findings"].append(
                        "Low contrast texture observed, suggesting homogeneous composition."
                    )
        
        # Add more specific interpretations based on your domain knowledge
        
        # Add disclaimer
        interpretation["disclaimer"] = (
            "This automated analysis is for informational purposes only. "
            "Results should be validated by a qualified professional."
        )
        
        return interpretation
    
    def _get_clinical_interpretation(self, label):
        """Get clinical interpretation for a predicted class label"""
        # This would be customized based on your specific application and classes
        # Here's a placeholder implementation
        interpretations = {
            "0": "Normal morphology detected",
            "1": "Abnormal morphology suggestive of pathology",
            "normal": "Normal morphological pattern",
            "abnormal": "Abnormal morphological pattern",
            "benign": "Morphological features consistent with benign pattern",
            "malignant": "Morphological features suggesting malignancy"
        }
        
        return interpretations.get(str(label), "No specific interpretation available")
    
    def _generate_recommendations(self, features, predictions):
        """Generate clinical recommendations based on morphology and predictions"""
        # This would be customized based on your specific application
        # Here's a placeholder implementation
        
        if isinstance(predictions, np.ndarray):
            unique_labels = np.unique(predictions)
            
            if len(unique_labels) == 1:
                label = str(unique_labels[0])
                
                if label in ["0", "normal", "benign"]:
                    return [
                        "Based on morphometric analysis, no further action is recommended.",
                        "Follow standard monitoring protocols."
                    ]
                elif label in ["1", "abnormal", "malignant"]:
                    return [
                        "Further clinical correlation is strongly recommended.",
                        "Consider additional diagnostic testing to confirm findings.",
                        "Clinical follow-up with specialist consultation is advised."
                    ]
            else:
                return [
                    "Mixed morphometric patterns detected.",
                    "Clinical correlation and further evaluation is recommended.",
                    "Consider specialized review of the samples."
                ]
        
        return ["No specific recommendations available based on the current analysis."]
    
    def _get_clinical_figures(self, features):
        """Generate clinical-relevant figures for the report"""
        figures = []
        
        # Use existing figures if available
        if self.figures_dir.exists():
            # Priority figures for clinical reporting
            priority_figures = [
                'feature_importance',
                'tsne_visualization',
                'umap_visualization',
                'feature_correlation',
                'feature_pairplot'
            ]
            
            # Check for each priority figure
            for fig_name in priority_figures:
                for ext in ['.png', '.jpg', '.jpeg']:
                    fig_path = self.figures_dir / f"{fig_name}{ext}"
                    if fig_path.exists():
                        figures.append({
                            'name': fig_path.stem,
                            'path': str(fig_path),
                            'title': fig_path.stem.replace('_', ' ').title()
                        })
        
        return figures
    
    def _render_template(self, data, template_type='clinical'):
        """Render HTML template with the provided data"""
        if template_type == 'clinical':
            template_str = self._get_clinical_template()
        elif template_type == 'prediction':
            template_str = self._get_prediction_template()
        else:  # Research/default
            template_str = self._get_research_template()
        
        # Create Jinja2 template
        template = Template(template_str)
        
        # Add image encoding function to data
        data['encode_image'] = self._encode_image
        
        # Render template
        return template.render(**data)
    
    def _encode_image(self, image_path):
        """Encode image as base64 for embedding in HTML"""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/png;base64,{encoded_string}"
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return ""
    
    def _get_clinical_template(self):
        """Get HTML template for clinical reports"""
        return '''<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 10px;
            border-bottom: 2px solid #4682B4;
        }
        .section {
            margin-bottom: 30px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        h1 {
            color: #4682B4;
        }
        h2 {
            color: #4682B4;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4682B4;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .interpretation {
            font-style: italic;
            color: #666;
        }
        .abnormal {
            color: #d9534f;
            font-weight: bold;
        }
        .normal {
            color: #5cb85c;
        }
        .warning {
            background-color: #fcf8e3;
            padding: 15px;
            border-left: 5px solid #f0ad4e;
            margin-bottom: 20px;
        }
        .figure-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
        }
        .figure {
            margin: 10px;
            max-width: 45%;
            text-align: center;
        }
        .figure img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .figure-caption {
            margin-top: 5px;
            color: #666;
            font-size: 0.9em;
        }
        .disclaimer {
            margin-top: 30px;
            padding: 10px;
            background-color: #f7f7f7;
            border-left: 5px solid #777;
            font-size: 0.9em;
            color: #555;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Generated on: {{ timestamp }}</p>
    </div>

    <div class="section">
        <h2>Patient Information</h2>
        <table>
            <tr>
                <td><strong>Patient ID:</strong></td>
                <td>{{ patient_info.id }}</td>
                <td><strong>Age:</strong></td>
                <td>{{ patient_info.age }}</td>
                <td><strong>Sex:</strong></td>
                <td>{{ patient_info.sex }}</td>
            </tr>
        </table>
    </div>

    <div class="section">
        <h2>Morphology Summary</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>Value</th>
                <th>Reference Range</th>
                <th>Interpretation</th>
            </tr>
            {% for feature, data in morphology_summary.items() %}
            <tr>
                <td>{{ feature }}</td>
                <td>{{ data.value }}{% if data.std %} ± {{ data.std }}{% endif %}</td>
                <td>
                    {% if data.reference_range.lower is not none and data.reference_range.upper is not none %}
                        {{ data.reference_range.lower }} - {{ data.reference_range.upper }}
                    {% else %}
                        Not available
                    {% endif %}
                </td>
                <td class="interpretation {% if 'normal' in data.interpretation.lower() %}normal{% elif 'below' in data.interpretation.lower() or 'above' in data.interpretation.lower() %}abnormal{% endif %}">
                    {{ data.interpretation }}
                </td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="section">
        <h2>Clinical Interpretation</h2>
        <p>{{ clinical_interpretation.summary }}</p>
        
        <h3>Key Findings:</h3>
        <ul>
            {% for finding in clinical_interpretation.findings %}
            <li>{{ finding }}</li>
            {% endfor %}
        </ul>
        
        <div class="warning">
            <p>{{ clinical_interpretation.disclaimer }}</p>
        </div>
    </div>

    <div class="section">
        <h2>Analysis Results</h2>
        <table>
            <tr>
                <th>Classification</th>
                <th>Count</th>
                <th>Percentage</th>
                <th>Interpretation</th>
            </tr>
            {% for pred in predictions %}
            <tr>
                <td>{{ pred.label }}</td>
                <td>{{ pred.count }}</td>
                <td>{{ pred.percentage }}%</td>
                <td>{{ pred.interpretation }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        <ul>
            {% for rec in recommendations %}
            <li>{{ rec }}</li>
            {% endfor %}
        </ul>
    </div>

    {% if figures %}
    <div class="section">
        <h2>Visualizations</h2>
        <div class="figure-container">
            {% for figure in figures %}
            <div class="figure">
                <img src="{{ encode_image(figure.path) }}" alt="{{ figure.title }}">
                <div class="figure-caption">{{ figure.title }}</div>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}

    <div class="disclaimer">
        <p><strong>Disclaimer:</strong> This report is generated automatically from morphometric analysis and should be interpreted by qualified healthcare professionals. The results should be correlated with clinical findings and other diagnostic tests.</p>
    </div>
</body>
</html>'''
    
    def _get_prediction_template(self):
        """Get HTML template for prediction reports"""
        return '''<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 10px;
            border-bottom: 2px solid #4682B4;
        }
        .section {
            margin-bottom: 30px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        h1 {
            color: #4682B4;
        }
        h2 {
            color: #4682B4;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4682B4;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Generated on: {{ timestamp }}</p>
    </div>

    <div class="section">
        <h2>Features Summary</h2>
        <table>
            <tr>
                <td><strong>Number of samples:</strong></td>
                <td>{{ features_summary.count }}</td>
            </tr>
            <tr>
                <td><strong>Number of features:</strong></td>
                <td>{{ features_summary.feature_count }}</td>
            </tr>
        </table>
        
        <h3>Feature Groups:</h3>
        <table>
            <tr>
                <th>Group</th>
                <th>Count</th>
            </tr>
            {% for group, count in features_summary.feature_group_counts.items() %}
            <tr>
                <td>{{ group }}</td>
                <td>{{ count }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="section">
        <h2>Prediction Results</h2>
        <table>
            <tr>
                <th>Class</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
            {% for pred in predictions %}
            <tr>
                <td>{{ pred.label }}</td>
                <td>{{ pred.count }}</td>
                <td>{{ pred.percentage }}%</td>
            </tr>
            {% endfor %}
        </table>
    </div>
</body>
</html>'''
    
    def _get_research_template(self):
        """Get HTML template for research reports"""
        return '''<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 10px;
            border-bottom: 2px solid #4682B4;
        }
        .section {
            margin-bottom: 30px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        h1 {
            color: #4682B4;
        }
        h2 {
            color: #4682B4;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4682B4;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .figure-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
        }
        .figure {
            margin: 10px;
            max-width: 45%;
            text-align: center;
        }
        .figure img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .figure-caption {
            margin-top: 5px;
            color: #666;
            font-size: 0.9em;
        }
        .metrics-table {
            width: auto;
            min-width: 300px;
        }
        .metrics-table th:first-child {
            width: 150px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Generated on: {{ timestamp }}</p>
    </div>

    <div class="section">
        <h2>Features Summary</h2>
        <table>
            <tr>
                <td><strong>Number of samples:</strong></td>
                <td>{{ features_summary.count }}</td>
            </tr>
            <tr>
                <td><strong>Number of features:</strong></td>
                <td>{{ features_summary.feature_count }}</td>
            </tr>
        </table>
        
        <h3>Feature Groups:</h3>
        <table>
            <tr>
                <th>Group</th>
                <th>Count</th>
            </tr>
            {% for group, count in features_summary.feature_group_counts.items() %}
            <tr>
                <td>{{ group }}</td>
                <td>{{ count }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="section">
        <h2>Model Information</h2>
        <table>
            <tr>
                <td><strong>Model Type:</strong></td>
                <td>{{ model_summary.type }}</td>
            </tr>
            {% if model_summary.classifier %}
            <tr>
                <td><strong>Classifier:</strong></td>
                <td>{{ model_summary.classifier }}</td>
            </tr>
            {% endif %}
            <tr>
                <td><strong>Ensemble Model:</strong></td>
                <td>{{ model_summary.ensemble }}</td>
            </tr>
            {% if model_summary.ensemble and model_summary.base_models %}
            <tr>
                <td><strong>Base Models:</strong></td>
                <td>{{ model_summary.base_models|join(', ') }}</td>
            </tr>
            {% endif %}
        </table>
    </div>

    <div class="section">
        <h2>Evaluation Results</h2>
        
        {% if evaluation_summary.metrics %}
        <h3>Test Set Metrics:</h3>
        <table class="metrics-table">
            {% for metric, value in evaluation_summary.metrics.items() %}
            <tr>
                <th>{{ metric|capitalize }}</th>
                <td>{{ "%.4f"|format(value) }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
        
        {% if evaluation_summary.cross_validation %}
        <h3>Cross-Validation Metrics:</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Mean</th>
                <th>Std</th>
            </tr>
            {% for metric, values in evaluation_summary.cross_validation.items() %}
            <tr>
                <td>{{ metric|capitalize }}</td>
                <td>{{ "%.4f"|format(values.mean) }}</td>
                <td>{{ "%.4f"|format(values.std) }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
        
        {% if evaluation_summary.class_metrics %}
        <h3>Class-level Performance:</h3>
        <table>
            <tr>
                <th>Class</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-score</th>
                <th>Support</th>
            </tr>
            {% for class, metrics in evaluation_summary.class_metrics.items() %}
            <tr>
                <td>{{ class }}</td>
                <td>{{ "%.3f"|format(metrics.precision) }}</td>
                <td>{{ "%.3f"|format(metrics.recall) }}</td>
                <td>{{ "%.3f"|format(metrics.f1) }}</td>
                <td>{{ metrics.support }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
    </div>

    {% if figures %}
    <div class="section">
        <h2>Visualizations</h2>
        <div class="figure-container">
            {% for figure in figures %}
            <div class="figure">
                <img src="{{ encode_image(figure.path) }}" alt="{{ figure.title }}">
                <div class="figure-caption">{{ figure.title }}</div>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}
</body>
</html>'''