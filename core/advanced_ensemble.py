#!/usr/bin/env python3
"""
Advanced Ensemble Model with Hyperparameter Optimization
Implements sophisticated ensemble methods and GPU-accelerated training
"""

import numpy as np
import pandas as pd
import pickle
import time
import warnings
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
from scipy.stats import randint, uniform
import logging

warnings.filterwarnings('ignore')

class AdvancedEnsembleModel:
    """Advanced ensemble model with comprehensive optimization"""
    
    def __init__(self, gpu_available=False, device=None):
        self.gpu_available = gpu_available
        self.device = device
        self.models = {}
        self.optimized_models = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.ensemble_model = None
        self.best_params = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def create_optimized_models(self):
        """Create models with optimized hyperparameters"""
        
        # Random Forest with extensive hyperparameter space
        rf_params = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
            'bootstrap': [True, False],
            'class_weight': ['balanced', None],
            'criterion': ['gini', 'entropy']
        }
        
        # Extra Trees with extensive hyperparameter space
        et_params = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
            'bootstrap': [True, False],
            'class_weight': ['balanced', None],
            'criterion': ['gini', 'entropy']
        }
        
        # XGBoost with comprehensive hyperparameter space
        xgb_params = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bylevel': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [1, 2, 5, 10],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.5],
            'scale_pos_weight': [1, 2, 3]
        }
        
        # MLP with extensive hyperparameter space
        mlp_params = {
            'hidden_layer_sizes': [
                (50,), (100,), (200,), (300,),
                (50, 50), (100, 50), (200, 100), (300, 150),
                (100, 100), (200, 200),
                (50, 50, 50), (100, 100, 100)
            ],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'lbfgs'],
            'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'max_iter': [200, 300, 500],
            'early_stopping': [True, False],
            'validation_fraction': [0.1, 0.2, 0.3],
            'beta_1': [0.9, 0.95, 0.99],
            'beta_2': [0.999, 0.9999]
        }
        
        return {
            'rf': (RandomForestClassifier(random_state=42, n_jobs=-1), rf_params),
            'et': (ExtraTreesClassifier(random_state=42, n_jobs=-1), et_params),
            'xgb': (xgb.XGBClassifier(
                random_state=42,
                tree_method='gpu_hist' if self.gpu_available else 'hist',
                gpu_id=0 if self.gpu_available else None,
                objective='binary:logistic',
                eval_metric=['logloss', 'auc', 'error']
            ), xgb_params),
            'mlp': (MLPClassifier(random_state=42), mlp_params)
        }
    
    def optimize_hyperparameters(self, X, y, cv_folds=10):
        """Optimize hyperparameters using advanced search strategies"""
        self.logger.info("Starting comprehensive hyperparameter optimization...")
        
        # Create optimized models
        model_configs = self.create_optimized_models()
        
        for name, (base_model, param_grid) in model_configs.items():
            self.logger.info(f"Optimizing {name.upper()}...")
            
            try:
                # Use RandomizedSearchCV for efficiency with large parameter spaces
                search = RandomizedSearchCV(
                    base_model,
                    param_distributions=param_grid,
                    n_iter=100,  # More iterations for better optimization
                    cv=cv_folds,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    random_state=42,
                    verbose=0
                )
                
                # Fit the search
                start_time = time.time()
                search.fit(X, y)
                optimization_time = time.time() - start_time
                
                # Store results
                self.optimized_models[name] = search.best_estimator_
                self.best_params[name] = search.best_params_
                
                self.logger.info(f"{name.upper()} optimization completed in {optimization_time:.2f}s")
                self.logger.info(f"Best {name.upper()} score: {search.best_score_:.4f}")
                
                # Evaluate with cross-validation
                cv_scores = cross_val_score(
                    search.best_estimator_, X, y, 
                    cv=cv_folds, scoring='accuracy'
                )
                
                self.logger.info(f"{name.upper()} CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                self.logger.error(f"Error optimizing {name}: {e}")
                # Use default model if optimization fails
                self.optimized_models[name] = base_model
                self.best_params[name] = {}
        
        return self.optimized_models
    
    def create_advanced_ensemble(self, X, y):
        """Create advanced ensemble with multiple voting strategies"""
        self.logger.info("Creating advanced ensemble model...")
        
        if not self.optimized_models:
            self.logger.error("No optimized models available for ensemble")
            return None
        
        # Create voting ensemble
        estimators = [(name, model) for name, model in self.optimized_models.items()]
        
        # Hard voting ensemble
        hard_ensemble = VotingClassifier(
            estimators=estimators,
            voting='hard',
            n_jobs=-1
        )
        
        # Soft voting ensemble (if all models support predict_proba)
        soft_ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        # Evaluate both ensembles
        hard_scores = cross_val_score(hard_ensemble, X, y, cv=5, scoring='accuracy')
        soft_scores = cross_val_score(soft_ensemble, X, y, cv=5, scoring='accuracy')
        
        self.logger.info(f"Hard Voting Ensemble CV Accuracy: {hard_scores.mean():.4f} ± {hard_scores.std():.4f}")
        self.logger.info(f"Soft Voting Ensemble CV Accuracy: {soft_scores.mean():.4f} ± {soft_scores.std():.4f}")
        
        # Choose the better ensemble
        if soft_scores.mean() > hard_scores.mean():
            self.ensemble_model = soft_ensemble
            self.logger.info("Selected Soft Voting Ensemble")
        else:
            self.ensemble_model = hard_ensemble
            self.logger.info("Selected Hard Voting Ensemble")
        
        # Fit the ensemble
        self.ensemble_model.fit(X, y)
        
        return self.ensemble_model
    
    def create_stacking_ensemble(self, X, y):
        """Create stacking ensemble with meta-learner"""
        try:
            from sklearn.ensemble import StackingClassifier
            
            self.logger.info("Creating stacking ensemble...")
            
            if not self.optimized_models:
                self.logger.error("No optimized models available for stacking")
                return None
            
            # Create base estimators
            estimators = [(name, model) for name, model in self.optimized_models.items()]
            
            # Try different meta-learners
            meta_learners = {
                'rf_meta': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
                'xgb_meta': xgb.XGBClassifier(
                    n_estimators=50,
                    random_state=42,
                    tree_method='gpu_hist' if self.gpu_available else 'hist'
                ),
                'mlp_meta': MLPClassifier(
                    hidden_layer_sizes=(100,),
                    random_state=42,
                    max_iter=200
                )
            }
            
            best_stacking = None
            best_score = 0
            
            for meta_name, meta_learner in meta_learners.items():
                try:
                    stacking = StackingClassifier(
                        estimators=estimators,
                        final_estimator=meta_learner,
                        cv=5,
                        stack_method='auto',
                        n_jobs=-1
                    )
                    
                    scores = cross_val_score(stacking, X, y, cv=5, scoring='accuracy')
                    avg_score = scores.mean()
                    
                    self.logger.info(f"Stacking with {meta_name}: {avg_score:.4f} ± {scores.std():.4f}")
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_stacking = stacking
                        
                except Exception as e:
                    self.logger.warning(f"Failed to create stacking with {meta_name}: {e}")
            
            if best_stacking is not None:
                best_stacking.fit(X, y)
                self.logger.info(f"Best stacking ensemble score: {best_score:.4f}")
                return best_stacking
            else:
                self.logger.warning("No stacking ensemble could be created")
                return None
                
        except ImportError:
            self.logger.warning("StackingClassifier not available, using voting ensemble")
            return None
    
    def apply_advanced_feature_selection(self, X, y, k='auto'):
        """Apply advanced feature selection techniques"""
        self.logger.info("Applying advanced feature selection...")
        
        n_features = X.shape[1]
        
        if k == 'auto':
            # Automatically determine optimal number of features
            # Try different percentages and use cross-validation to find best
            percentages = [0.1, 0.2, 0.3, 0.5, 0.7]
            best_score = 0
            best_k = min(100, n_features)
            
            for pct in percentages:
                k_test = int(n_features * pct)
                if k_test < 10:
                    continue
                
                # Test with mutual information
                selector = SelectKBest(score_func=mutual_info_classif, k=k_test)
                X_selected = selector.fit_transform(X, y)
                
                # Quick evaluation with a simple model
                rf_test = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                scores = cross_val_score(rf_test, X_selected, y, cv=3, scoring='accuracy')
                avg_score = scores.mean()
                
                self.logger.info(f"Feature selection with {k_test} features: {avg_score:.4f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_k = k_test
            
            k = best_k
            self.logger.info(f"Selected optimal number of features: {k}")
        
        # Apply feature selection with the best k
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        self.logger.info(f"Feature selection: {X.shape[1]} -> {X_selected.shape[1]} features")
        
        return X_selected
    
    def train_comprehensive_model(self, X, y, feature_selection=True, ensemble_type='voting'):
        """Train comprehensive model with all optimizations"""
        self.logger.info("Starting comprehensive model training...")
        
        # Scale features
        self.logger.info("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply feature selection if requested
        if feature_selection:
            X_processed = self.apply_advanced_feature_selection(X_scaled, y)
        else:
            X_processed = X_scaled
        
        # Optimize individual models
        self.optimize_hyperparameters(X_processed, y)
        
        # Create ensemble
        if ensemble_type == 'voting':
            final_model = self.create_advanced_ensemble(X_processed, y)
        elif ensemble_type == 'stacking':
            final_model = self.create_stacking_ensemble(X_processed, y)
            if final_model is None:
                self.logger.info("Falling back to voting ensemble")
                final_model = self.create_advanced_ensemble(X_processed, y)
        else:
            self.logger.error(f"Unknown ensemble type: {ensemble_type}")
            final_model = self.create_advanced_ensemble(X_processed, y)
        
        # Final evaluation
        if final_model is not None:
            final_scores = cross_val_score(final_model, X_processed, y, cv=10, scoring='accuracy')
            self.logger.info(f"Final model CV Accuracy: {final_scores.mean():.4f} ± {final_scores.std():.4f}")
            
            # Additional metrics
            f1_scores = cross_val_score(final_model, X_processed, y, cv=10, scoring='f1_weighted')
            precision_scores = cross_val_score(final_model, X_processed, y, cv=10, scoring='precision_weighted')
            recall_scores = cross_val_score(final_model, X_processed, y, cv=10, scoring='recall_weighted')
            
            self.logger.info(f"Final model CV F1: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
            self.logger.info(f"Final model CV Precision: {precision_scores.mean():.4f} ± {precision_scores.std():.4f}")
            self.logger.info(f"Final model CV Recall: {recall_scores.mean():.4f} ± {recall_scores.std():.4f}")
            
            return final_model
        else:
            self.logger.error("Failed to create final model")
            return None
    
    def save_model(self, model, filepath_prefix):
        """Save the trained model and preprocessing components"""
        try:
            # Save the main model
            with open(f"{filepath_prefix}_ensemble_model.pkl", 'wb') as f:
                pickle.dump(model, f)
            
            # Save individual optimized models
            with open(f"{filepath_prefix}_individual_models.pkl", 'wb') as f:
                pickle.dump(self.optimized_models, f)
            
            # Save scaler
            with open(f"{filepath_prefix}_scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save feature selector
            if self.feature_selector is not None:
                with open(f"{filepath_prefix}_feature_selector.pkl", 'wb') as f:
                    pickle.dump(self.feature_selector, f)
            
            # Save best parameters
            with open(f"{filepath_prefix}_best_params.pkl", 'wb') as f:
                pickle.dump(self.best_params, f)
            
            self.logger.info(f"Model saved successfully with prefix: {filepath_prefix}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath_prefix):
        """Load the trained model and preprocessing components"""
        try:
            # Load the main model
            with open(f"{filepath_prefix}_ensemble_model.pkl", 'rb') as f:
                model = pickle.load(f)
            
            # Load individual optimized models
            try:
                with open(f"{filepath_prefix}_individual_models.pkl", 'rb') as f:
                    self.optimized_models = pickle.load(f)
            except FileNotFoundError:
                self.logger.warning("Individual models file not found")
            
            # Load scaler
            with open(f"{filepath_prefix}_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load feature selector
            try:
                with open(f"{filepath_prefix}_feature_selector.pkl", 'rb') as f:
                    self.feature_selector = pickle.load(f)
            except FileNotFoundError:
                self.logger.warning("Feature selector file not found")
                self.feature_selector = None
            
            # Load best parameters
            try:
                with open(f"{filepath_prefix}_best_params.pkl", 'rb') as f:
                    self.best_params = pickle.load(f)
            except FileNotFoundError:
                self.logger.warning("Best parameters file not found")
                self.best_params = {}
            
            self.logger.info(f"Model loaded successfully from prefix: {filepath_prefix}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None
    
    def predict_with_ensemble(self, model, X):
        """Make predictions with proper preprocessing"""
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Apply feature selection if available
            if self.feature_selector is not None:
                X_processed = self.feature_selector.transform(X_scaled)
            else:
                X_processed = X_scaled
            
            # Make predictions
            predictions = model.predict(X_processed)
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_processed)
                return predictions, probabilities
            else:
                return predictions, None
                
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            return None, None
