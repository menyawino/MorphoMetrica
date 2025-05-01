"""
Visualization module for morphometry analysis.
This module contains methods to create visualizations for morphometric features and results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')


class Visualizer:
    """Generate visualizations for morphometric analysis"""
    
    def __init__(self, config):
        """Initialize visualizer with configuration"""
        self.config = config
        self.viz_config = config['visualization']
        self.output_dir = Path(config['data']['output_dir'])
        self.figures_dir = self.output_dir / 'figures'
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize(self, features_df, model=None, evaluation_results=None):
        """
        Generate visualizations for features and model results
        
        Parameters:
        -----------
        features_df : pandas DataFrame
            DataFrame containing extracted morphometric features
        model : trained model object, optional
            The trained model for feature importance visualizations
        evaluation_results : dict, optional
            Dictionary containing evaluation results
        """
        logger.info("Generating visualizations")
        
        # Prepare data
        X, y = self._prepare_data(features_df)
        
        # Generate visualizations
        if self.viz_config.get('tsne', False):
            self._visualize_tsne(X, y)
            
        if self.viz_config.get('umap', False):
            self._visualize_umap(X, y)
            
        if model is not None:
            if self.viz_config.get('feature_importance', False):
                self._visualize_feature_importance(X, y, model, features_df.columns)
                
            if evaluation_results is not None and self.viz_config.get('confusion_matrix', False):
                self._visualize_confusion_matrix(evaluation_results)
                
            if evaluation_results is not None and self.viz_config.get('roc_curve', False):
                self._visualize_roc_curve(evaluation_results)
        
        # Feature correlation heatmap
        self._visualize_feature_correlation(X, features_df.columns)
        
        # Distribution of top features
        self._visualize_feature_distributions(X, y, features_df.columns)
        
        logger.info(f"Visualizations saved to {self.figures_dir}")
    
    def _prepare_data(self, features_df):
        """Prepare data for visualization"""
        # Check if labels are in DataFrame
        if 'label' in features_df.columns:
            X = features_df.drop('label', axis=1)
            y = features_df['label']
        else:
            X = features_df
            y = None
        
        # Drop any non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Fill missing values
        X = X.fillna(0)
        
        return X, y
    
    def _visualize_tsne(self, X, y=None, n_components=2, perplexity=30):
        """Generate t-SNE visualization of features"""
        logger.info("Generating t-SNE visualization")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply t-SNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        tsne_features = tsne.fit_transform(X_scaled)
        
        # Create DataFrame for plotting
        df_tsne = pd.DataFrame(tsne_features, columns=[f'TSNE{i+1}' for i in range(n_components)])
        
        # Add label if available
        if y is not None:
            df_tsne['label'] = y
        
        # Create plot
        plt.figure(figsize=(10, 8))
        if y is not None:
            scatter = sns.scatterplot(
                data=df_tsne,
                x='TSNE1',
                y='TSNE2',
                hue='label',
                palette='viridis',
                alpha=0.8
            )
            plt.legend(title='Class')
        else:
            scatter = sns.scatterplot(
                data=df_tsne,
                x='TSNE1',
                y='TSNE2',
                alpha=0.8
            )
        
        plt.title('t-SNE Visualization of Morphometric Features')
        plt.tight_layout()
        
        # Save figure
        fig_path = self.figures_dir / 'tsne_visualization.png'
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        logger.info(f"t-SNE visualization saved to {fig_path}")
    
    def _visualize_umap(self, X, y=None, n_components=2, n_neighbors=15):
        """Generate UMAP visualization of features"""
        logger.info("Generating UMAP visualization")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply UMAP
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
        umap_features = reducer.fit_transform(X_scaled)
        
        # Create DataFrame for plotting
        df_umap = pd.DataFrame(umap_features, columns=[f'UMAP{i+1}' for i in range(n_components)])
        
        # Add label if available
        if y is not None:
            df_umap['label'] = y
        
        # Create plot
        plt.figure(figsize=(10, 8))
        if y is not None:
            scatter = sns.scatterplot(
                data=df_umap,
                x='UMAP1',
                y='UMAP2',
                hue='label',
                palette='viridis',
                alpha=0.8
            )
            plt.legend(title='Class')
        else:
            scatter = sns.scatterplot(
                data=df_umap,
                x='UMAP1',
                y='UMAP2',
                alpha=0.8
            )
        
        plt.title('UMAP Visualization of Morphometric Features')
        plt.tight_layout()
        
        # Save figure
        fig_path = self.figures_dir / 'umap_visualization.png'
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        logger.info(f"UMAP visualization saved to {fig_path}")
    
    def _visualize_feature_importance(self, X, y, model, feature_names):
        """Visualize feature importance from the model"""
        logger.info("Generating feature importance visualization")
        
        # Check if model supports feature importance
        feature_importance = None
        
        # For ensemble models with custom wrapper
        if hasattr(model, 'get_feature_importance'):
            importance_dict = model.get_feature_importance()
            
            # Create a figure for each base model
            for model_name, importance in importance_dict.items():
                # Get feature names
                if hasattr(model, 'base_models') and model_name in model.base_models:
                    base_model = model.base_models[model_name]
                    if hasattr(base_model, 'named_steps') and 'pca' in base_model.named_steps:
                        # For PCA-based models, we can't directly map to original features
                        logger.info(f"Skipping feature importance for {model_name} (uses PCA)")
                        continue
                
                # Sort features by importance
                if len(importance) == len(feature_names):
                    sorted_idx = np.argsort(importance)[::-1]
                    sorted_importance = importance[sorted_idx]
                    sorted_features = np.array(feature_names)[sorted_idx]
                    
                    # Plot top 20 features
                    plt.figure(figsize=(12, 10))
                    plt.barh(range(min(20, len(sorted_features))), 
                            sorted_importance[:20], 
                            tick_label=sorted_features[:20])
                    plt.xlabel('Importance')
                    plt.title(f'Feature Importance ({model_name})')
                    plt.tight_layout()
                    
                    # Save figure
                    fig_path = self.figures_dir / f'feature_importance_{model_name}.png'
                    plt.savefig(fig_path, dpi=300)
                    plt.close()
                    
                    logger.info(f"Feature importance for {model_name} saved to {fig_path}")
        
        # For sklearn models with feature_importances_
        elif hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        
        # For sklearn pipeline with classifier having feature_importances_
        elif hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            clf = model.named_steps['classifier']
            if hasattr(clf, 'feature_importances_'):
                feature_importance = clf.feature_importances_
        
        # If feature importance is available, plot it
        if feature_importance is not None:
            # Sort features by importance
            sorted_idx = np.argsort(feature_importance)[::-1]
            sorted_importance = feature_importance[sorted_idx]
            sorted_features = np.array(feature_names)[sorted_idx]
            
            # Plot top 20 features
            plt.figure(figsize=(12, 10))
            plt.barh(range(min(20, len(sorted_features))), 
                    sorted_importance[:20], 
                    tick_label=sorted_features[:20])
            plt.xlabel('Importance')
            plt.title('Feature Importance')
            plt.tight_layout()
            
            # Save figure
            fig_path = self.figures_dir / 'feature_importance.png'
            plt.savefig(fig_path, dpi=300)
            plt.close()
            
            logger.info(f"Feature importance visualization saved to {fig_path}")
            
        # If model doesn't support feature importance, use SelectKBest
        else:
            logger.info("Model doesn't support feature importance, using F-statistic")
            
            # Use F-statistic for feature selection
            if y is not None:
                selector = SelectKBest(score_func=f_classif, k=min(20, X.shape[1]))
                selector.fit(X, y)
                
                # Get scores and features
                scores = selector.scores_
                sorted_idx = np.argsort(scores)[::-1]
                sorted_scores = scores[sorted_idx]
                sorted_features = np.array(feature_names)[sorted_idx]
                
                # Plot top 20 features
                plt.figure(figsize=(12, 10))
                plt.barh(range(min(20, len(sorted_features))), 
                        sorted_scores[:20], 
                        tick_label=sorted_features[:20])
                plt.xlabel('F-score')
                plt.title('Feature Importance (F-test)')
                plt.tight_layout()
                
                # Save figure
                fig_path = self.figures_dir / 'feature_importance_f_test.png'
                plt.savefig(fig_path, dpi=300)
                plt.close()
                
                logger.info(f"F-test feature importance visualization saved to {fig_path}")
            else:
                logger.warning("Cannot calculate feature importance without labels")
    
    def _visualize_confusion_matrix(self, evaluation_results):
        """Visualize confusion matrix from evaluation results"""
        logger.info("Generating confusion matrix visualization")
        
        # Check if confusion matrix is in evaluation results
        if 'test_set' in evaluation_results and 'confusion_matrix' in evaluation_results['test_set']:
            cm = np.array(evaluation_results['test_set']['confusion_matrix'])
            
            # Create plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            
            # Save figure
            fig_path = self.figures_dir / 'confusion_matrix.png'
            plt.savefig(fig_path, dpi=300)
            plt.close()
            
            logger.info(f"Confusion matrix visualization saved to {fig_path}")
        else:
            logger.warning("Confusion matrix not found in evaluation results")
    
    def _visualize_roc_curve(self, evaluation_results):
        """Visualize ROC curve from evaluation results"""
        logger.info("Generating ROC curve visualization")
        
        # Check if ROC curve data is in evaluation results
        if ('test_set' in evaluation_results and 
            'roc_auc' in evaluation_results['test_set']):
            
            roc_data = evaluation_results['test_set']['roc_auc']
            
            # Create plot
            plt.figure(figsize=(10, 8))
            
            for class_label, data in roc_data.items():
                fpr = np.array(data['fpr'])
                tpr = np.array(data['tpr'])
                auc_value = data['auc']
                
                plt.plot(fpr, tpr, label=f'Class {class_label} (AUC = {auc_value:.2f})')
            
            # Add diagonal line (random classifier)
            plt.plot([0, 1], [0, 1], 'k--')
            
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.legend()
            plt.tight_layout()
            
            # Save figure
            fig_path = self.figures_dir / 'roc_curves.png'
            plt.savefig(fig_path, dpi=300)
            plt.close()
            
            logger.info(f"ROC curve visualization saved to {fig_path}")
        else:
            logger.warning("ROC curve data not found in evaluation results")
    
    def _visualize_feature_correlation(self, X, feature_names, max_features=30):
        """Visualize correlation between features"""
        logger.info("Generating feature correlation heatmap")
        
        # Create DataFrame with feature names
        df = pd.DataFrame(X, columns=feature_names)
        
        # Limit to a reasonable number of features for visualization
        if df.shape[1] > max_features:
            # Select features with highest variance
            variances = df.var().sort_values(ascending=False)
            top_features = variances.index[:max_features]
            df = df[top_features]
        
        # Calculate correlation matrix
        corr = df.corr()
        
        # Create plot
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='RdBu_r', vmax=1, vmin=-1, 
                   center=0, square=True, linewidths=.5, annot=False)
        plt.title('Feature Correlation Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        fig_path = self.figures_dir / 'feature_correlation.png'
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        logger.info(f"Feature correlation heatmap saved to {fig_path}")
    
    def _visualize_feature_distributions(self, X, y, feature_names, top_n=5):
        """Visualize distributions of top features"""
        logger.info("Generating feature distribution plots")
        
        if y is None:
            logger.warning("Cannot visualize feature distributions without labels")
            return
        
        # Create DataFrame with feature names
        df = pd.DataFrame(X, columns=feature_names)
        df['label'] = y
        
        # Find most discriminative features using ANOVA F-value
        selector = SelectKBest(score_func=f_classif, k=top_n)
        selector.fit(X, y)
        
        # Get top features
        top_features = feature_names[selector.get_support()]
        
        # Create plot for each top feature
        for feature in top_features:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=feature, hue='label', kde=True, element='step')
            plt.title(f'Distribution of {feature} by Class')
            plt.tight_layout()
            
            # Save figure
            fig_path = self.figures_dir / f'distribution_{feature}.png'
            plt.savefig(fig_path, dpi=300)
            plt.close()
        
        # Create pair plot for top features
        plt.figure(figsize=(15, 15))
        sns.pairplot(df[list(top_features) + ['label']], hue='label', diag_kind='kde')
        plt.tight_layout()
        
        # Save figure
        fig_path = self.figures_dir / 'feature_pairplot.png'
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        logger.info(f"Feature distribution plots saved to {self.figures_dir}")