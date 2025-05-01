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
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_curve, auc, precision_recall_curve, confusion_matrix, 
    classification_report
)
import logging

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
        
        # Fill missing values
        X = X.fillna(0)
        
        logger.info(f"Prepared data with {X.shape[0]} samples and {X.shape[1]} features")
        return X, y
    
    def _perform_cross_validation(self, model, X, y):
        """Perform cross-validation evaluation"""
        n_splits = self.eval_config['cross_validation'].get('n_splits', 5)
        cv_metrics = self.eval_config.get('metrics', ['accuracy'])
        
        logger.info(f"Performing {n_splits}-fold cross-validation")
        
        results = {}
        
        # Define stratified k-fold cross-validation
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Calculate metrics for each fold
        for metric in cv_metrics:
            if metric == 'accuracy':
                scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            elif metric == 'precision':
                scores = cross_val_score(model, X, y, cv=cv, scoring='precision_weighted')
            elif metric == 'recall':
                scores = cross_val_score(model, X, y, cv=cv, scoring='recall_weighted')
            elif metric == 'f1':
                scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
            elif metric == 'roc_auc':
                try:
                    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc_ovr_weighted')
                except:
                    logger.warning("ROC AUC score calculation failed, skipping")
                    scores = np.array([])
            else:
                logger.warning(f"Unknown metric: {metric}, skipping")
                scores = np.array([])
                
            # Store results
            results[metric] = {
                'scores': scores.tolist(),
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores))
            }
            
            logger.info(f"Cross-validation {metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        return results
    
    def _evaluate_on_test_set(self, model, X, y, test_size=0.2):
        """Evaluate the model on a test set"""
        from sklearn.model_selection import train_test_split
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model on training set if not already fitted
        # This handles the case where a new model instance is passed
        try:
            y_pred_proba = model.predict_proba(X_test)
            already_trained = True
        except:
            already_trained = False
            
        if not already_trained:
            logger.info("Model not trained, fitting on training data")
            model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self.eval_config.get('metrics', ['accuracy'])
        results = {}
        
        # Basic metrics
        if 'accuracy' in metrics:
            results['accuracy'] = float(accuracy_score(y_test, y_pred))
        
        if 'precision' in metrics:
            results['precision'] = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        
        if 'recall' in metrics:
            results['recall'] = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        
        if 'f1' in metrics:
            results['f1'] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        
        # Confusion matrix
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
                        y_score = y_pred_proba[:, 1]
                    
                    # Calculate ROC curve
                    fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
                    roc_auc = auc(fpr, tpr)
                    
                    roc_results[str(cls)] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'thresholds': thresholds.tolist(),
                        'auc': float(roc_auc)
                    }
                
                results['roc_auc'] = roc_results
                
                # Also calculate weighted average AUC
                from sklearn.metrics import roc_auc_score
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
        """Save evaluation results to disk"""
        import json
        
        # Save as JSON 
        results_path = self.results_dir / "evaluation_results.json"
        
        # Convert numpy arrays and other non-JSON-serializable objects
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            if isinstance(obj, np.int32) or isinstance(obj, np.int64):
                return int(obj)
            return obj
        
        # Recursively convert all values
        def make_serializable(d):
            result = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    result[key] = make_serializable(value)
                else:
                    result[key] = convert_to_serializable(value)
            return result
        
        serializable_results = make_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_path}")
        
        return results_path