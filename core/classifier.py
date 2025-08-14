#!/usr/bin/env python3
"""
Ultra-Enhanced XGBoost Classifier with Advanced Ensemble Methods
Implementing XGBoost and ensemble techniques
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, roc_curve, precision_recall_curve
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier
import xgboost as xgb

warnings.filterwarnings('ignore')

# Optional imports for ensemble methods
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

class UltraEnhancedXGBoostClassifier:
    """
    Ultra-enhanced XGBoost classifier with advanced hyperparameter tuning,
    ensemble methods, and comprehensive evaluation metrics
    """
    def __init__(self, n_estimators=5000, max_depth=20, learning_rate=0.01, 
                 subsample=0.9, colsample_bytree=0.9, random_state=42,
                 objective='binary:logistic', eval_metric='logloss',
                 min_child_weight=5, gamma=0.05, reg_alpha=0.01, 
                 reg_lambda=0.05, n_jobs=-1, verbosity=0, **kwargs):
        
        # Enhanced parameters for maximum performance
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'colsample_bylevel': 0.9,
            'colsample_bynode': 0.9,
            'random_state': random_state,
            'objective': objective,
            'eval_metric': eval_metric,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'n_jobs': n_jobs,
            'verbosity': verbosity,
            'tree_method': 'hist',
            'enable_categorical': False,
            'max_delta_step': 2,
            'min_split_loss': 0.05,
            'scale_pos_weight': 1
        }
        
        # Remove None values and add kwargs
        params = {k: v for k, v in params.items() if v is not None}
        params.update(kwargs)
        
        try:
            self.model = xgb.XGBClassifier(**params)
            print(f"✅ Enhanced XGBoost classifier initialized with {n_estimators} estimators")
        except Exception as e:
            print(f"❌ Error initializing XGBoost: {e}")
            # Fallback with basic parameters
            self.model = xgb.XGBClassifier(
                n_estimators=1000,
                max_depth=10,
                learning_rate=0.1,
                random_state=random_state
            )
        
        self.feature_importance_ = None
        self.training_history_ = None
        self.params = params
        self.ensemble_models = {}
        self.voting_classifier = None

    def fit(self, features, labels, eval_set=None, verbose=True):
        """
        Train the enhanced XGBoost model with optional validation set
        """
        print("Training ultra-enhanced XGBoost model...")
        
        if eval_set is not None:
            self.model.fit(
                features, labels,
                eval_set=eval_set,
                early_stopping_rounds=100,
                verbose=False
            )
            if hasattr(self.model, 'evals_result_'):
                self.training_history_ = self.model.evals_result_
        else:
            self.model.fit(features, labels)
        
        # Store feature importance
        self.feature_importance_ = self.model.feature_importances_
        
        if verbose:
            print(f"✅ Enhanced XGBoost training completed")
            print(f"   - Estimators used: {self.model.n_estimators}")
            print(f"   - Feature importance shape: {self.feature_importance_.shape}")

    def create_ensemble_models(self, features, labels):
        """Create ensemble with multiple gradient boosting algorithms"""
        print("Creating ultra-enhanced ensemble models...")
        
        ensemble_models = []
        
        # XGBoost (already trained)
        ensemble_models.append(('xgboost', self.model))
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            try:
                lgb_model = lgb.LGBMClassifier(
                    objective='binary',
                    metric='binary_logloss',
                    num_leaves=300,
                    learning_rate=0.01,
                    feature_fraction=0.9,
                    bagging_fraction=0.9,
                    bagging_freq=5,
                    min_child_samples=20,
                    max_depth=15,
                    reg_alpha=0.01,
                    reg_lambda=0.05,
                    n_estimators=3000,
                    random_state=42,
                    verbosity=-1,
                    force_col_wise=True
                )
                lgb_model.fit(features, labels)
                ensemble_models.append(('lightgbm', lgb_model))
                self.ensemble_models['lightgbm'] = lgb_model
                print("✅ LightGBM model added to ensemble")
            except Exception as e:
                print(f"⚠️ LightGBM failed: {e}")
        
        # CatBoost
        if CATBOOST_AVAILABLE:
            try:
                cat_model = cb.CatBoostClassifier(
                    objective='Logloss',
                    eval_metric='AUC',
                    iterations=2000,
                    learning_rate=0.01,
                    depth=12,
                    l2_leaf_reg=3,
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False
                )
                cat_model.fit(features, labels)
                ensemble_models.append(('catboost', cat_model))
                self.ensemble_models['catboost'] = cat_model
                print("✅ CatBoost model added to ensemble")
            except Exception as e:
                print(f"⚠️ CatBoost failed: {e}")
        
        # Create voting classifier if we have multiple models
        if len(ensemble_models) > 1:
            self.voting_classifier = VotingClassifier(
                estimators=ensemble_models,
                voting='soft',
                n_jobs=-1
            )
            print(f"✅ Voting ensemble created with {len(ensemble_models)} models")
        else:
            print("⚠️ Only XGBoost available, ensemble not created")

    def predict(self, features):
        """Predict class labels using ensemble if available"""
        if self.voting_classifier:
            return self.voting_classifier.predict(features)
        else:
            return self.model.predict(features)

    def predict_proba(self, features):
        """Predict class probabilities using ensemble if available"""
        if self.voting_classifier:
            return self.voting_classifier.predict_proba(features)
        else:
            return self.model.predict_proba(features)

    def set_params(self, **params):
        """Set parameters for the XGBoost model"""
        self.model.set_params(**params)
        return self
    
    def get_params(self, deep=True):
        """Get parameters for the XGBoost model"""
        return self.model.get_params(deep)
    
    def get_cv_model(self):
        """Get a model suitable for cross-validation"""
        cv_params = self.params.copy()
        # Remove early stopping parameters for cross-validation
        cv_params.pop('early_stopping_rounds', None)
        return xgb.XGBClassifier(**cv_params)

    def comprehensive_evaluate(self, features, labels, class_names=['Authentic', 'Forged']):
        """
        Ultra-comprehensive evaluation of the model with detailed metrics
        """
        print("Performing comprehensive model evaluation...")
        
        # Get predictions from ensemble if available, otherwise from XGBoost
        if self.voting_classifier:
            predictions = self.voting_classifier.predict(features)
            probabilities = self.voting_classifier.predict_proba(features)
            model_name = "Ensemble"
        else:
            predictions = self.model.predict(features)
            probabilities = self.model.predict_proba(features)
            model_name = "XGBoost"
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        auc_score = roc_auc_score(labels, probabilities[:, 1])
        
        print(f"\\n{model_name} Model Performance:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy:.1%})")
        print(f"AUC Score: {auc_score:.4f}")
        
        # Detailed classification report
        print("\\nDetailed Classification Report:")
        print(classification_report(labels, predictions, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Image Forgery Detection')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(labels, probabilities[:, 1])
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, linewidth=2, label=f'{model_name}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Image Forgery Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Individual model performance if ensemble is used
        if self.voting_classifier and len(self.ensemble_models) > 0:
            print("\\nIndividual Model Performance:")
            
            # XGBoost
            xgb_pred = self.model.predict(features)
            xgb_prob = self.model.predict_proba(features)[:, 1]
            xgb_acc = accuracy_score(labels, xgb_pred)
            xgb_auc = roc_auc_score(labels, xgb_prob)
            print(f"  XGBoost: Accuracy = {xgb_acc:.4f}, AUC = {xgb_auc:.4f}")
            
            # LightGBM
            if 'lightgbm' in self.ensemble_models:
                lgb_pred = self.ensemble_models['lightgbm'].predict(features)
                lgb_prob = self.ensemble_models['lightgbm'].predict_proba(features)[:, 1]
                lgb_acc = accuracy_score(labels, lgb_pred)
                lgb_auc = roc_auc_score(labels, lgb_prob)
                print(f"  LightGBM: Accuracy = {lgb_acc:.4f}, AUC = {lgb_auc:.4f}")
            
            # CatBoost
            if 'catboost' in self.ensemble_models:
                cat_pred = self.ensemble_models['catboost'].predict(features)
                cat_prob = self.ensemble_models['catboost'].predict_proba(features)[:, 1]
                cat_acc = accuracy_score(labels, cat_pred)
                cat_auc = roc_auc_score(labels, cat_prob)
                print(f"  CatBoost: Accuracy = {cat_acc:.4f}, AUC = {cat_auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'predictions': predictions,
            'probabilities': probabilities,
            'confusion_matrix': cm,
            'model_used': model_name
        }

    def advanced_cross_validate(self, features, labels, cv=10):
        """
        Perform advanced cross-validation with multiple models
        """
        print(f"Performing {cv}-fold cross-validation...")
        
        results = {}
        
        # XGBoost cross-validation
        cv_model = self.get_cv_model()
        xgb_scores = cross_val_score(cv_model, features, labels, cv=cv, scoring='roc_auc', n_jobs=-1)
        results['xgboost'] = {
            'scores': xgb_scores,
            'mean': xgb_scores.mean(),
            'std': xgb_scores.std()
        }
        print(f"XGBoost CV AUC: {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")
        
        # LightGBM cross-validation with conservative parameters
        if LIGHTGBM_AVAILABLE:
            try:
                lgb_model = lgb.LGBMClassifier(
                    objective='binary',
                    metric='binary_logloss',
                    num_leaves=15,  # Much smaller
                    learning_rate=0.1,  # Higher but with fewer estimators
                    n_estimators=200,  # Fewer estimators
                    min_child_samples=50,  # Strong regularization
                    min_split_gain=0.1,
                    reg_alpha=0.3,
                    reg_lambda=0.3,
                    feature_fraction=0.6,
                    bagging_fraction=0.6,
                    random_state=42,
                    verbosity=-1
                )
                lgb_scores = cross_val_score(lgb_model, features, labels, cv=cv, scoring='roc_auc', n_jobs=-1)
                results['lightgbm'] = {
                    'scores': lgb_scores,
                    'mean': lgb_scores.mean(),
                    'std': lgb_scores.std()
                }
                print(f"LightGBM CV AUC: {lgb_scores.mean():.4f} ± {lgb_scores.std():.4f}")
            except Exception as e:
                print(f"⚠️ LightGBM CV failed: {e}")
        
        # CatBoost cross-validation
        if CATBOOST_AVAILABLE:
            try:
                cat_model = cb.CatBoostClassifier(
                    iterations=1000,
                    learning_rate=0.05,
                    depth=6,
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False
                )
                cat_scores = cross_val_score(cat_model, features, labels, cv=cv, scoring='roc_auc', n_jobs=-1)
                results['catboost'] = {
                    'scores': cat_scores,
                    'mean': cat_scores.mean(),
                    'std': cat_scores.std()
                }
                print(f"CatBoost CV AUC: {cat_scores.mean():.4f} ± {cat_scores.std():.4f}")
            except Exception as e:
                print(f"⚠️ CatBoost CV failed: {e}")
        
        return results

    def advanced_hyperparameter_tuning(self, features, labels, param_distributions=None, cv=5, n_iter=50):
        """
        Perform advanced hyperparameter tuning using RandomizedSearchCV
        """
        if param_distributions is None:
            param_distributions = {
                'n_estimators': [1000, 2000, 3000, 5000],
                'max_depth': [10, 15, 20, 25],
                'learning_rate': [0.005, 0.01, 0.02, 0.05],
                'subsample': [0.8, 0.85, 0.9, 0.95],
                'colsample_bytree': [0.8, 0.85, 0.9, 0.95],
                'min_child_weight': [1, 3, 5, 7],
                'gamma': [0, 0.01, 0.05, 0.1],
                'reg_alpha': [0, 0.01, 0.05, 0.1],
                'reg_lambda': [0.01, 0.05, 0.1, 0.2]
            }
        
        print(f"Performing advanced hyperparameter tuning with {n_iter} iterations...")
        
        # Create base model for tuning - Remove early stopping for CV compatibility
        base_model = xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            # Remove any early stopping parameters that require validation set
            early_stopping_rounds=None
        )
        
        # Randomized search
        random_search = RandomizedSearchCV(
            base_model, 
            param_distributions, 
            n_iter=n_iter,
            cv=cv, 
            scoring='roc_auc',
            n_jobs=-1, 
            verbose=1,
            random_state=42
        )
        
        random_search.fit(features, labels)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV AUC score: {random_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.model = random_search.best_estimator_
        
        return random_search

    def plot_enhanced_feature_importance(self, feature_names=None, top_n=30):
        """
        Plot enhanced feature importance analysis
        """
        if self.feature_importance_ is None:
            print("⚠️ Model not trained yet. Train the model first.")
            return
        
        # Get top N important features
        indices = np.argsort(self.feature_importance_)[::-1][:top_n]
        
        plt.figure(figsize=(12, 10))
        if feature_names is not None:
            names = [feature_names[i] for i in indices]
        else:
            names = [f'Feature_{i}' for i in indices]
        
        # Create horizontal bar plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
        bars = plt.barh(range(len(indices)), self.feature_importance_[indices], color=colors)
        
        plt.yticks(range(len(indices)), names)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances - Ultra-Enhanced XGBoost')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.show()

    def plot_enhanced_training_history(self):
        """
        Plot enhanced training history with multiple metrics
        """
        if self.training_history_ is None:
            print("⚠️ No training history available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training and validation loss
        if 'validation_0' in self.training_history_:
            train_loss = self.training_history_['train']['logloss']
            val_loss = self.training_history_['validation_0']['logloss']
            
            axes[0, 0].plot(train_loss, label='Training Loss', linewidth=2)
            axes[0, 0].plot(val_loss, label='Validation Loss', linewidth=2)
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Iterations')
            axes[0, 0].set_ylabel('Log Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot AUC if available
            if 'auc' in self.training_history_['validation_0']:
                val_auc = self.training_history_['validation_0']['auc']
                axes[0, 1].plot(val_auc, label='Validation AUC', linewidth=2, color='green')
                axes[0, 1].set_title('Validation AUC Score')
                axes[0, 1].set_xlabel('Iterations')
                axes[0, 1].set_ylabel('AUC Score')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Plot error if available
            if 'error' in self.training_history_['validation_0']:
                val_error = self.training_history_['validation_0']['error']
                axes[1, 0].plot(val_error, label='Validation Error', linewidth=2, color='red')
                axes[1, 0].set_title('Validation Error Rate')
                axes[1, 0].set_xlabel('Iterations')
                axes[1, 0].set_ylabel('Error Rate')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Learning curve analysis
            if len(train_loss) > 50:
                window = 50
                train_smooth = np.convolve(train_loss, np.ones(window)/window, mode='valid')
                val_smooth = np.convolve(val_loss, np.ones(window)/window, mode='valid')
                
                axes[1, 1].plot(train_smooth, label='Training (Smoothed)', linewidth=2)
                axes[1, 1].plot(val_smooth, label='Validation (Smoothed)', linewidth=2)
                axes[1, 1].set_title('Smoothed Learning Curves')
                axes[1, 1].set_xlabel('Iterations')
                axes[1, 1].set_ylabel('Smoothed Log Loss')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def save_enhanced_model(self, path="ultra_enhanced_xgb_model.pkl"):
        """Save the enhanced model with all components"""
        model_data = {
            'xgboost_model': self.model,
            'ensemble_models': self.ensemble_models,
            'voting_classifier': self.voting_classifier,
            'feature_importance': self.feature_importance_,
            'training_history': self.training_history_,
            'params': self.params
        }
        joblib.dump(model_data, path)
        print(f"✅ Enhanced model saved to {path}")

    def load_enhanced_model(self, path="ultra_enhanced_xgb_model.pkl"):
        """Load the enhanced model with all components"""
        try:
            model_data = joblib.load(path)
            self.model = model_data['xgboost_model']
            self.ensemble_models = model_data.get('ensemble_models', {})
            self.voting_classifier = model_data.get('voting_classifier', None)
            self.feature_importance_ = model_data.get('feature_importance', None)
            self.training_history_ = model_data.get('training_history', None)
            self.params = model_data.get('params', {})
            print(f"✅ Enhanced model loaded from {path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

    def get_enhanced_model_info(self):
        """Get comprehensive model information"""
        info = {
            'model_type': 'Ultra-Enhanced XGBoost with Ensemble',
            'xgboost_params': self.params,
            'ensemble_available': self.voting_classifier is not None,
            'ensemble_models': list(self.ensemble_models.keys()),
            'feature_importance_available': self.feature_importance_ is not None,
            'training_history_available': self.training_history_ is not None
        }
        
        if self.feature_importance_ is not None:
            info['n_features'] = len(self.feature_importance_)
            info['top_feature_importance'] = float(np.max(self.feature_importance_))
        
        return info

# Alias for backward compatibility
XGBoostClassifier = UltraEnhancedXGBoostClassifier

print("Ultra-Enhanced XGBoost Classifier loaded successfully!")
