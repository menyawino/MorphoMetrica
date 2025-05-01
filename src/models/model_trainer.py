"""
Model training module for morphometry analysis.
This module contains methods to train machine learning models on extracted features.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train machine learning models on extracted morphometric features"""
    
    def __init__(self, config):
        """Initialize the model trainer with configuration"""
        self.config = config
        self.model_config = config['model']
        self.save_dir = Path(self.model_config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self, features_df, labels=None):
        """
        Train a model on extracted features
        
        Parameters:
        -----------
        features_df : pandas DataFrame
            DataFrame containing extracted features
        labels : pandas Series or numpy array, optional
            Target labels for supervised learning. If not provided, 
            will look for a 'label' column in features_df
            
        Returns:
        --------
        model : trained model object
            The trained model
        """
        logger.info("Training model on extracted features")
        
        # Prepare data
        X, y = self._prepare_data(features_df, labels)
        
        # Create and train model based on config
        model_type = self.model_config.get('type', 'random_forest')
        
        if model_type == 'ensemble':
            model = self._train_ensemble(X, y)
        elif model_type == 'cnn':
            model = self._train_cnn(X, y)
        elif model_type == 'random_forest':
            model = self._train_random_forest(X, y)
        elif model_type == 'gradient_boosting':
            model = self._train_gradient_boosting(X, y)
        elif model_type == 'svm':
            model = self._train_svm(X, y)
        else:
            logger.warning(f"Unknown model type: {model_type}, defaulting to random_forest")
            model = self._train_random_forest(X, y)
        
        # Save the trained model
        self._save_model(model)
        
        return model
    
    def _prepare_data(self, features_df, labels=None):
        """Prepare data for model training"""
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
    
    def _train_random_forest(self, X, y):
        """Train a Random Forest classifier"""
        logger.info("Training Random Forest classifier")
        
        # Create a pipeline with scaling and PCA
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100, 
                max_depth=None,
                min_samples_split=2,
                random_state=42
            ))
        ])
        
        # Train the model
        pipeline.fit(X, y)
        
        # Evaluate on training data
        train_pred = pipeline.predict(X)
        train_acc = accuracy_score(y, train_pred)
        logger.info(f"Random Forest training accuracy: {train_acc:.4f}")
        
        return pipeline
    
    def _train_gradient_boosting(self, X, y):
        """Train a Gradient Boosting classifier"""
        logger.info("Training Gradient Boosting classifier")
        
        # Create a pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            ))
        ])
        
        # Train the model
        pipeline.fit(X, y)
        
        # Evaluate on training data
        train_pred = pipeline.predict(X)
        train_acc = accuracy_score(y, train_pred)
        logger.info(f"Gradient Boosting training accuracy: {train_acc:.4f}")
        
        return pipeline
    
    def _train_svm(self, X, y):
        """Train an SVM classifier"""
        logger.info("Training SVM classifier")
        
        # Create a pipeline with scaling and PCA for dimensionality reduction
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=min(50, X.shape[1]), random_state=42)),
            ('classifier', SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42
            ))
        ])
        
        # Train the model
        pipeline.fit(X, y)
        
        # Evaluate on training data
        train_pred = pipeline.predict(X)
        train_acc = accuracy_score(y, train_pred)
        logger.info(f"SVM training accuracy: {train_acc:.4f}")
        
        return pipeline
    
    def _train_ensemble(self, X, y):
        """Train an ensemble of multiple models"""
        logger.info("Training ensemble of classifiers")
        
        # Split data for blending
        X_train, X_blend, y_train, y_blend = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Define base models
        models = {
            'rf': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
            ]),
            'gb': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
            ]),
            'svm': Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=min(30, X.shape[1]))),
                ('clf', SVC(probability=True, random_state=42))
            ]),
            'mlp': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
            ])
        }
        
        # Train base models and get predictions for blending
        blend_preds = np.zeros((X_blend.shape[0], len(models), len(np.unique(y))))
        
        for i, (name, model) in enumerate(models.items()):
            logger.info(f"Training base model: {name}")
            model.fit(X_train, y_train)
            blend_preds[:, i, :] = model.predict_proba(X_blend)
        
        # Flatten predictions for meta-model
        blend_X = blend_preds.reshape(X_blend.shape[0], -1)
        
        # Train meta-model
        meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
        meta_model.fit(blend_X, y_blend)
        
        # Create and return ensemble model wrapper
        ensemble = EnsembleModel(base_models=models, meta_model=meta_model)
        
        return ensemble
    
    def _train_cnn(self, X, y):
        """Train a simple CNN for feature-based classification"""
        logger.info("Training CNN classifier")
        
        # Preprocess data for CNN
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Reshape data for CNN (add channel dimension)
        n_features = X.shape[1]
        X_reshaped = X_scaled.reshape(-1, n_features, 1)
        
        # Convert labels to one-hot encoding
        from tensorflow.keras.utils import to_categorical
        unique_labels = np.unique(y)
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        y_idx = np.array([label_to_idx[label] for label in y])
        y_onehot = to_categorical(y_idx)
        
        # Define CNN model
        model = models.Sequential([
            layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_features, 1)),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(unique_labels), activation='softmax')
        ])
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=self.model_config.get('learning_rate', 0.001))
        model.compile(
            optimizer=optimizer,
            loss=self.model_config.get('loss', 'categorical_crossentropy'),
            metrics=self.model_config.get('metrics', ['accuracy'])
        )
        
        # Early stopping callback
        callbacks = []
        if self.model_config.get('early_stopping', {}).get('enable', False):
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=self.model_config['early_stopping'].get('monitor', 'val_loss'),
                patience=self.model_config['early_stopping'].get('patience', 10),
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_reshaped, y_onehot, test_size=0.1, random_state=42
        )
        
        # Train model
        batch_size = self.model_config.get('batch_size', 32)
        epochs = self.model_config.get('epochs', 100)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Create a wrapper for the CNN model
        cnn_model = CNNModel(model, scaler, unique_labels)
        
        return cnn_model
    
    def _save_model(self, model):
        """Save the trained model to disk"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.save_dir / f"morphometry_model_{timestamp}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a trained model from disk"""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        logger.info(f"Model loaded from {model_path}")
        return model


class EnsembleModel:
    """Wrapper for ensemble model with base models and meta-model"""
    
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        self.num_models = len(base_models)
        self.model_names = list(base_models.keys())
        
    def predict(self, X):
        """Make predictions using the ensemble"""
        # Get base model predictions
        preds = np.zeros((X.shape[0], self.num_models, self.meta_model.n_classes_))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            preds[:, i, :] = model.predict_proba(X)
        
        # Reshape for meta-model
        meta_features = preds.reshape(X.shape[0], -1)
        
        # Make final predictions
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X):
        """Return probability estimates"""
        # Get base model predictions
        preds = np.zeros((X.shape[0], self.num_models, self.meta_model.n_classes_))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            preds[:, i, :] = model.predict_proba(X)
        
        # Reshape for meta-model
        meta_features = preds.reshape(X.shape[0], -1)
        
        # Return probability estimates
        return self.meta_model.predict_proba(meta_features)
    
    def get_feature_importance(self):
        """Get feature importance from base models (if supported)"""
        importance = {}
        
        for name, model in self.base_models.items():
            if hasattr(model, 'named_steps') and 'clf' in model.named_steps:
                clf = model.named_steps['clf']
                if hasattr(clf, 'feature_importances_'):
                    importance[name] = clf.feature_importances_
        
        return importance


class CNNModel:
    """Wrapper for CNN model to provide scikit-learn-like interface"""
    
    def __init__(self, model, scaler, label_mapping):
        self.model = model
        self.scaler = scaler
        self.label_mapping = label_mapping
        
    def predict(self, X):
        """Make predictions using the CNN"""
        X_scaled = self.scaler.transform(X)
        X_reshaped = X_scaled.reshape(-1, X.shape[1], 1)
        
        # Get predicted probabilities
        y_probs = self.model.predict(X_reshaped)
        
        # Return class with highest probability
        y_pred_idx = np.argmax(y_probs, axis=1)
        return np.array([self.label_mapping[idx] for idx in y_pred_idx])
    
    def predict_proba(self, X):
        """Return probability estimates"""
        X_scaled = self.scaler.transform(X)
        X_reshaped = X_scaled.reshape(-1, X.shape[1], 1)
        
        return self.model.predict(X_reshaped)