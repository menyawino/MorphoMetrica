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
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import time
import multiprocessing

# Configure logging
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
        
        # Set number of threads for parallel model training
        self.n_jobs = min(multiprocessing.cpu_count(), 4)
        
        # Check for GPU availability
        self.gpu_available = self._check_gpu()
        logger.info(f"GPU available: {self.gpu_available}")
        
        # Set batch size based on available hardware
        self.batch_size = self.model_config.get('batch_size', 32)
        if self.gpu_available:
            # Use larger batch sizes on GPU for better throughput
            self.batch_size = min(self.batch_size * 2, 128)
    
    def _check_gpu(self):
        """Check for GPU availability"""
        # Check TensorFlow GPU
        tf_gpu = False
        try:
            tf_gpu = len(tf.config.list_physical_devices('GPU')) > 0
        except:
            pass
        
        # Check PyTorch GPU
        torch_gpu = False
        try:
            torch_gpu = torch.cuda.is_available()
        except:
            pass
        
        return tf_gpu or torch_gpu
    
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
        start_time = time.time()
        logger.info("Training model on extracted features")
        
        # Prepare data
        X, y = self._prepare_data(features_df, labels)
        
        # Apply automated feature selection if there are too many features
        if X.shape[1] > 100:
            X = self._perform_feature_selection(X, y)
        
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
        
        elapsed_time = time.time() - start_time
        logger.info(f"Model training completed in {elapsed_time:.2f} seconds")
        
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
        
        # Fill missing values (use median instead of mean for more robustness)
        X = X.fillna(X.median())
        
        # Remove near-zero variance features for efficiency
        near_zero_vars = []
        for col in X.columns:
            if X[col].var() < 1e-8:
                near_zero_vars.append(col)
        
        if near_zero_vars:
            logger.info(f"Removing {len(near_zero_vars)} near-zero variance features")
            X = X.drop(columns=near_zero_vars)
        
        # Handle infinity and large values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        logger.info(f"Prepared data with {X.shape[0]} samples and {X.shape[1]} features")
        return X, y
    
    def _perform_feature_selection(self, X, y):
        """Perform feature selection to reduce dimensionality"""
        logger.info(f"Performing feature selection from {X.shape[1]} features")
        
        # Use a quick filter method (ANOVA F-value)
        try:
            # Determine number of features to keep - use heuristic based on sample count
            k_features = min(100, max(20, X.shape[0] // 5))
            selector = SelectKBest(f_classif, k=k_features)
            X_new = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_indices = selector.get_support(indices=True)
            selected_features = X.columns[selected_indices]
            
            logger.info(f"Selected {len(selected_features)} features using ANOVA F-values")
            
            # Create new dataframe with selected features
            X_selected = pd.DataFrame(X_new, index=X.index)
            X_selected.columns = selected_features
            
            return X_selected
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return X
    
    def _train_random_forest(self, X, y):
        """Train a Random Forest classifier"""
        logger.info("Training Random Forest classifier")
        
        # Use better hyperparameters for efficiency
        n_estimators = self.model_config.get('n_estimators', 100)
        
        # Calculate optimal parameters based on data size
        if X.shape[0] < 100:  # Small dataset
            n_estimators = min(n_estimators, 50)
            max_depth = 10
        elif X.shape[0] > 1000:  # Large dataset
            # For large datasets, we can use more trees for better accuracy
            n_estimators = max(n_estimators, 150)
            max_depth = 20
        else:
            max_depth = 15
            
        # Create a pipeline with scaling and Random Forest
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth,
                min_samples_split=2,
                bootstrap=True,
                n_jobs=self.n_jobs,  # Use parallelism
                random_state=42,
                # Use class weight balancing if needed
                class_weight='balanced' if len(np.unique(y)) <= 10 else None,
                # Use float32 for efficiency
                verbose=0
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
        """Train a Gradient Boosting classifier with optimized parameters"""
        logger.info("Training Gradient Boosting classifier")
        
        # Adjust hyperparameters based on dataset size
        if X.shape[0] < 100:  # Small dataset
            n_estimators = 50
            max_depth = 3
            learning_rate = 0.1
        elif X.shape[0] > 1000:  # Large dataset
            n_estimators = 200
            max_depth = 5
            learning_rate = 0.05
        else:
            n_estimators = 100
            max_depth = 4
            learning_rate = 0.1
            
        # Use histogram-based gradient boosting if available for better speed
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            
            # Create a pipeline with scaling and Histogram Gradient Boosting
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', HistGradientBoostingClassifier(
                    max_iter=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=42,
                    verbose=0
                ))
            ])
            logger.info("Using Histogram-based Gradient Boosting for faster training")
            
        except ImportError:
            # Fall back to regular gradient boosting
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
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
        """Train an SVM classifier with optimized parameters"""
        logger.info("Training SVM classifier")
        
        # Adjust parameters based on dataset size
        if X.shape[0] > 1000 or X.shape[1] > 50:
            # For larger datasets, use linear kernel which is much faster
            kernel = 'linear'
            C = 1.0
            reduce_dim = min(50, X.shape[1])
        else:
            # For smaller datasets, RBF kernel often performs better
            kernel = 'rbf'
            C = 10.0
            reduce_dim = None
            
        # Create a pipeline with scaling and SVM
        pipeline_steps = [
            ('scaler', RobustScaler()),  # RobustScaler is less sensitive to outliers
        ]
        
        # Add PCA for dimensionality reduction if needed
        if reduce_dim is not None and X.shape[1] > reduce_dim:
            pipeline_steps.append(('pca', PCA(n_components=reduce_dim, random_state=42)))
            
        # Add the classifier
        pipeline_steps.append(('classifier', SVC(
            C=C,
            kernel=kernel,
            probability=True,
            random_state=42,
            # Use class weight balancing for better performance with imbalanced data
            class_weight='balanced',
            cache_size=1000  # Larger cache helps with larger datasets
        )))
        
        pipeline = Pipeline(pipeline_steps)
        
        # Train the model
        pipeline.fit(X, y)
        
        # Evaluate on training data
        train_pred = pipeline.predict(X)
        train_acc = accuracy_score(y, train_pred)
        logger.info(f"SVM training accuracy: {train_acc:.4f}")
        
        return pipeline
    
    def _train_ensemble(self, X, y):
        """Train an ensemble of multiple models with efficient hyperparameters"""
        logger.info("Training ensemble of classifiers")
        
        # Split data for blending
        X_train, X_blend, y_train, y_blend = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Define simplified base models that are faster to train
        models = {}
        
        # Random Forest
        models['rf'] = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=100, 
                max_depth=15, 
                n_jobs=self.n_jobs // 2,  # Split cores between models
                random_state=42
            ))
        ])
        
        # Try using Histogram Gradient Boosting if available (much faster)
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            models['hgb'] = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', HistGradientBoostingClassifier(
                    max_iter=100, 
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=43
                ))
            ])
        except ImportError:
            # Fall back to regular gradient boosting with fewer estimators
            models['gb'] = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=3,
                    random_state=43
                ))
            ])
        
        # SVM (only for smaller datasets)
        if X.shape[0] < 1000:
            models['svm'] = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=min(30, X.shape[1]))),
                ('clf', SVC(probability=True, kernel='rbf', random_state=44))
            ])
        
        # MLP (neural network)
        hidden_layer_size = min(100, X.shape[1] * 2)
        models['mlp'] = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(hidden_layer_size, hidden_layer_size // 2), 
                max_iter=300, 
                early_stopping=True,
                random_state=45
            ))
        ])
        
        # Train base models in parallel if possible
        if self.n_jobs > 1:
            from joblib import Parallel, delayed
            
            def train_model(name, model, X_train, y_train):
                logger.info(f"Training base model: {name}")
                model.fit(X_train, y_train)
                return name, model
                
            results = Parallel(n_jobs=min(len(models), self.n_jobs))(
                delayed(train_model)(name, model, X_train, y_train)
                for name, model in models.items()
            )
            
            # Update models dictionary with trained models
            models = {name: model for name, model in results}
            
        else:
            # Train sequentially
            for name, model in models.items():
                logger.info(f"Training base model: {name}")
                model.fit(X_train, y_train)
        
        # Get predictions for blending
        blend_preds = np.zeros((X_blend.shape[0], len(models), len(np.unique(y))))
        
        for i, (name, model) in enumerate(models.items()):
            blend_preds[:, i, :] = model.predict_proba(X_blend)
        
        # Flatten predictions for meta-model
        blend_X = blend_preds.reshape(X_blend.shape[0], -1)
        
        # Train meta-model (use a simpler model for efficiency)
        meta_model = RandomForestClassifier(n_estimators=50, random_state=46)
        meta_model.fit(blend_X, y_blend)
        
        # Create and return ensemble model wrapper
        ensemble = EnsembleModel(base_models=models, meta_model=meta_model)
        
        return ensemble
    
    def _train_cnn(self, X, y):
        """Train a CNN for feature-based classification with mixed precision and GPU support"""
        logger.info("Training CNN classifier")
        
        # Check if we should use TensorFlow or PyTorch based on available GPU
        if hasattr(tf, 'executing_eagerly') and tf.executing_eagerly():
            # Use TensorFlow
            return self._train_tf_cnn(X, y)
        else:
            # Use PyTorch
            return self._train_torch_cnn(X, y)
    
    def _train_tf_cnn(self, X, y):
        """Train a TensorFlow CNN with GPU acceleration and mixed precision"""
        # Preprocess data
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
        
        # Set up mixed precision for faster training on compatible GPUs
        if self.gpu_available:
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Using mixed precision training")
            except:
                logger.warning("Mixed precision not available")
        
        # Define optimized CNN model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', 
                                   input_shape=(n_features, 1)),
            tf.keras.layers.BatchNormalization(),  # Add batch normalization for stability
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),  # Increased dropout for better generalization
            tf.keras.layers.Dense(len(unique_labels), activation='softmax')
        ])
        
        # Compile model with optimized settings
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.model_config.get('learning_rate', 0.001)
        )
        
        model.compile(
            optimizer=optimizer,
            loss=self.model_config.get('loss', 'categorical_crossentropy'),
            metrics=['accuracy']
        )
        
        # Callbacks for improved training
        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_reshaped, y_onehot, test_size=0.2, random_state=42
        )
        
        # Train model with efficient batch size
        batch_size = self.batch_size
        epochs = self.model_config.get('epochs', 100)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Create a wrapper for the CNN model
        cnn_model = CNNModel(model, scaler, unique_labels)
        
        return cnn_model
    
    def _train_torch_cnn(self, X, y):
        """Train a PyTorch CNN with GPU acceleration"""
        # Preprocess data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled.reshape(-1, 1, X.shape[1]))
        
        # Convert labels
        unique_labels = np.unique(y)
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        y_idx = np.array([label_to_idx[label] for label in y])
        y_tensor = torch.LongTensor(y_idx)
        
        # Create dataset and dataloader for efficient batching
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Define the CNN model
        class CNN1D(nn.Module):
            def __init__(self, input_size, num_classes):
                super(CNN1D, self).__init__()
                self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm1d(32)
                self.pool = nn.MaxPool1d(2)
                self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm1d(64)
                
                # Calculate size after pooling
                self.feature_size = input_size // 2
                
                self.fc1 = nn.Linear(64 * (self.feature_size // 2), 64)
                self.dropout = nn.Dropout(0.3)
                self.fc2 = nn.Linear(64, num_classes)
                
            def forward(self, x):
                # Conv layers
                x = torch.relu(self.bn1(self.conv1(x)))
                x = self.pool(x)
                x = torch.relu(self.bn2(self.conv2(x)))
                x = self.pool(x)
                
                # Flatten
                x = x.view(x.size(0), -1)
                
                # Fully connected layers
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        # Initialize model
        model = CNN1D(X.shape[1], len(unique_labels))
        
        # Move to GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Set up optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=self.model_config.get('learning_rate', 0.001))
        criterion = nn.CrossEntropyLoss()
        
        # Train the model
        epochs = self.model_config.get('epochs', 100)
        best_val_loss = float('inf')
        best_model_state = None
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            val_acc = correct / total
            val_loss /= len(val_loader)
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Create wrapper model that implements scikit-learn interface
        torch_model = TorchModel(model, scaler, unique_labels, device)
        
        return torch_model
    
    def _save_model(self, model):
        """Save the trained model to disk"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.save_dir / f"optimal_model_{timestamp}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        logger.info(f"Model saved to {model_path}")
        return model_path
    
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
        # Check if X is a DataFrame
        if hasattr(X, 'values'):
            X = X.values
            
        # Get base model predictions
        preds = np.zeros((X.shape[0], self.num_models, self.meta_model.n_classes_))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            try:
                preds[:, i, :] = model.predict_proba(X)
            except:
                # If model fails for some reason, use uniform distribution
                preds[:, i, :] = 1.0 / self.meta_model.n_classes_
                logger.warning(f"Model {name} failed to predict, using uniform distribution")
        
        # Reshape for meta-model
        meta_features = preds.reshape(X.shape[0], -1)
        
        # Make final predictions
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X):
        """Return probability estimates"""
        # Check if X is a DataFrame
        if hasattr(X, 'values'):
            X = X.values
            
        # Get base model predictions
        preds = np.zeros((X.shape[0], self.num_models, self.meta_model.n_classes_))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            try:
                preds[:, i, :] = model.predict_proba(X)
            except:
                # If model fails, use uniform distribution
                preds[:, i, :] = 1.0 / self.meta_model.n_classes_
        
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
    """Wrapper for TensorFlow CNN model to provide scikit-learn-like interface"""
    
    def __init__(self, model, scaler, label_mapping):
        self.model = model
        self.scaler = scaler
        self.label_mapping = label_mapping
        
    def predict(self, X):
        """Make predictions using the CNN"""
        # Check if X is a DataFrame
        if hasattr(X, 'values'):
            X = X.values
            
        X_scaled = self.scaler.transform(X)
        X_reshaped = X_scaled.reshape(-1, X.shape[1], 1)
        
        # Get predicted probabilities
        y_probs = self.model.predict(X_reshaped, verbose=0)
        
        # Return class with highest probability
        y_pred_idx = np.argmax(y_probs, axis=1)
        return np.array([self.label_mapping[idx] for idx in y_pred_idx])
    
    def predict_proba(self, X):
        """Return probability estimates"""
        # Check if X is a DataFrame
        if hasattr(X, 'values'):
            X = X.values
            
        X_scaled = self.scaler.transform(X)
        X_reshaped = X_scaled.reshape(-1, X.shape[1], 1)
        
        return self.model.predict(X_reshaped, verbose=0)


class TorchModel:
    """Wrapper for PyTorch CNN model to provide scikit-learn-like interface"""
    
    def __init__(self, model, scaler, label_mapping, device):
        self.model = model
        self.scaler = scaler
        self.label_mapping = label_mapping
        self.device = device
        self.model.to(device)
        self.model.eval()  # Set model to evaluation mode
        
    def predict(self, X):
        """Make predictions using the CNN"""
        # Check if X is a DataFrame
        if hasattr(X, 'values'):
            X = X.values
            
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled.reshape(-1, 1, X.shape[1])).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = outputs.max(1)
        
        # Convert back to original labels
        predicted = predicted.cpu().numpy()
        return np.array([self.label_mapping[idx] for idx in predicted])
    
    def predict_proba(self, X):
        """Return probability estimates"""
        # Check if X is a DataFrame
        if hasattr(X, 'values'):
            X = X.values
            
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled.reshape(-1, 1, X.shape[1])).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()