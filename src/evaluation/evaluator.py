"""
Evaluation module for morphometry analysis.
This module contains methods to evaluate model performance and validate results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pickle
import joblib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_curve, auc, precision_recall_curve, confusion_matrix, 
    classification_report, roc_auc_score
)
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate morphometric models and validate results"""
    
    def __init__(self, config):
        """Initialize the evaluator with configuration"""
        self.config = config
        self.eval_config = config['evaluation']
        self.output_dir = Path(config['data']['output_dir'])
        self.results_dir = self.output_dir / 'evaluation'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set number of parallel jobs
        self.n_jobs = min(multiprocessing.cpu_count(), 4)
    
    def evaluate(self, model, features_df, labels=None, test_size=0.2):
        """
        Evaluate a trained model on data
        
        Parameters:
        -----------
        model : trained model object
            The model to evaluate
        features_df : pandas DataFrame
            DataFrame containing extracted features
        labels : pandas Series or numpy array, optional
            Target labels for evaluation. If not provided, 
            will look for a 'label' column in features_df
        test_size : float, optional
            Proportion of data to use for testing. Only used if not
            using cross-validation.
            
        Returns:
        --------
        results : dict
            Dictionary containing evaluation results
        """
        logger.info("Evaluating model performance")
        start_time = time.time()
        
        # Prepare data
        X, y = self._prepare_data(features_df, labels)
        
        results = {}
        
        # Perform cross-validation if enabled
        if self.eval_config.get('cross_validation', {}).get('enable', False):
            cv_results = self._perform_cross_validation(model, X, y)
            results['cross_validation'] = cv_results
            
        # Evaluate on test set
        test_results = self._evaluate_on_test_set(model, X, y, test_size)
        results['test_set'] = test_results
        
        # Save results
        self._save_results(results)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Model evaluation completed in {elapsed_time:.2f} seconds")
        
        return results
    
    def _prepare_data(self, features_df, labels=None):
        """Prepare data for evaluation"""
        # If labels not provided, check if they're in the DataFrame
        if labels is None:
            if 'label' in features_df.columns:
                X = features_df.drop('label', axis=1)
                y = features_df['label']
            else:
                raise ValueError("No labels provided and no 'label' column in features DataFrame")
        else:
            X = features_df
            y = labels
        
        # Drop any non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Fill missing values - use median for robustness
        X = X.fillna(X.median())
        
        logger.info(f"Prepared data with {X.shape[0]} samples and {X.shape[1]} features")
        return X, y
    
    def _perform_cross_validation(self, model, X, y):
        """Perform cross-validation evaluation with parallel processing"""
        n_splits = self.eval_config['cross_validation'].get('n_splits', 5)
        cv_metrics = self.eval_config.get('metrics', ['accuracy'])
        
        logger.info(f"Performing {n_splits}-fold cross-validation with {self.n_jobs} parallel jobs")
        
        results = {}
        
        # Define stratified k-fold cross-validation
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Calculate metrics for each fold, using joblib for parallel processing
        for metric in cv_metrics:
            try:
                if metric == 'accuracy':
                    scoring = 'accuracy'
                elif metric == 'precision':
                    scoring = 'precision_weighted'
                elif metric == 'recall':
                    scoring = 'recall_weighted'
                elif metric == 'f1':
                    scoring = 'f1_weighted'
                elif metric == 'roc_auc':
                    # ROC AUC needs special handling for multi-class
                    scoring = 'roc_auc_ovr_weighted' if len(np.unique(y)) > 2 else 'roc_auc'
                else:
                    logger.warning(f"Unknown metric: {metric}, skipping")
                    continue
                
                scores = cross_val_score(
                    model, X, y, cv=cv, scoring=scoring, n_jobs=self.n_jobs
                )
                
                # Store results
                results[metric] = {
                    'scores': scores.tolist(),
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores))
                }
                
                logger.info(f"Cross-validation {metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
                
            except Exception as e:
                logger.error(f"Error calculating {metric} in cross-validation: {e}")
                results[metric] = {
                    'scores': [],
                    'mean': 0.0,
                    'std': 0.0,
                    'error': str(e)
                }
        
        return results
    
    def _evaluate_on_test_set(self, model, X, y, test_size=0.2):
        """Evaluate the model on a test set with efficient computation"""
        from sklearn.model_selection import train_test_split
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model on training set if not already fitted
        # This handles the case where a new model instance is passed
        try:
            # Quick check if model is already trained
            y_pred_proba = model.predict_proba(X_test[:1])
            already_trained = True
        except:
            already_trained = False
            
        if not already_trained:
            logger.info("Model not trained, fitting on training data")
            model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics in parallel where possible
        metrics = self.eval_config.get('metrics', ['accuracy'])
        results = {}
        
        def calculate_metric(metric_name):
            """Helper function to calculate a single metric"""
            if metric_name == 'accuracy':
                return float(accuracy_score(y_test, y_pred))
            
            elif metric_name == 'precision':
                return float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            
            elif metric_name == 'recall':
                return float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            
            elif metric_name == 'f1':
                return float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            
            elif metric_name == 'roc_auc' and hasattr(model, 'predict_proba'):
                # For binary classification
                if len(np.unique(y)) == 2:
                    try:
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        return float(roc_auc_score(y_test, y_pred_proba))
                    except:
                        return None
                # For multi-class
                else:
                    try:
                        y_pred_proba = model.predict_proba(X_test)
                        return float(roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted'))
                    except:
                        return None
            
            return None
        
        # Use ThreadPool for metrics calculation
        with joblib.parallel_backend('threading', n_jobs=self.n_jobs):
            results_list = joblib.Parallel()(
                joblib.delayed(calculate_metric)(metric) for metric in metrics
            )
        
        # Assign results to dictionary
        for i, metric in enumerate(metrics):
            if results_list[i] is not None:
                results[metric] = results_list[i]
        
        # Confusion matrix - not parallelized since it's simple and fast
        if self.eval_config.get('confusion_matrix', False):
            cm = confusion_matrix(y_test, y_pred)
            results['confusion_matrix'] = cm.tolist()
            
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        results['classification_report'] = report
        
        # ROC curve and AUC for each class (if model supports predict_proba)
        if 'roc_auc' in metrics and hasattr(model, 'predict_proba'):
            try:
                # Get class probabilities
                y_pred_proba = model.predict_proba(X_test)
                
                # For multi-class, calculate ROC AUC for each class
                classes = np.unique(y)
                roc_results = {}
                
                for i, cls in enumerate(classes):
                    # Convert to binary classification problem (one-vs-rest)
                    y_true_binary = (y_test == cls).astype(int)
                    
                    # For multi-class, get the probability of the current class
                    if y_pred_proba.shape[1] > 2:
                        y_score = y_pred_proba[:, i]
                    else:
                        # For binary classification
                        if i == 0:
                            y_score = 1 - y_pred_proba[:, 1]
                        else:
                            y_score = y_pred_proba[:, 1]
                    
                    # Calculate ROC curve - efficient implementation
                    # Use fewer thresholds to speed up calculation
                    fpr, tpr, thresholds = roc_curve(y_true_binary, y_score, drop_intermediate=True)
                    roc_auc = auc(fpr, tpr)
                    
                    roc_results[str(cls)] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'thresholds': thresholds.tolist(),
                        'auc': float(roc_auc)
                    }
                
                results['roc_auc'] = roc_results
                
                # Also calculate weighted average AUC
                try:
                    avg_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                    results['weighted_auc'] = float(avg_auc)
                except:
                    logger.warning("Weighted AUC calculation failed")
            
            except Exception as e:
                logger.error(f"Error calculating ROC AUC: {e}")
        
        # Log results
        logger.info(f"Test set evaluation:")
        for metric, value in results.items():
            if isinstance(value, (float, int)):
                logger.info(f"{metric}: {value:.4f}")
        
        return results
    
    def _save_results(self, results):
        """Save evaluation results to disk using optimized JSON serialization"""
        import json
        import gzip
        
        # Save regular results
        results_path = self.results_dir / "evaluation_results.json"
        
        # Convert numpy arrays and other non-JSON-serializable objects
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
        
        # Recursively convert all values
        def make_serializable(d):
            result = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    result[key] = make_serializable(value)
                elif isinstance(value, list):
                    result[key] = [convert_to_serializable(item) for item in value]
                else:
                    result[key] = convert_to_serializable(value)
            return result
        
        serializable_results = make_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Additionally, save a compressed version for large result sets
        gzip_path = self.results_dir / "evaluation_results.json.gz"
        try:
            with gzip.open(gzip_path, 'wt') as f:
                json.dump(serializable_results, f)
            logger.info(f"Compressed evaluation results saved to {gzip_path}")
        except Exception as e:
            logger.warning(f"Failed to save compressed results: {e}")
        
        logger.info(f"Evaluation results saved to {results_path}")
        
        return results_path