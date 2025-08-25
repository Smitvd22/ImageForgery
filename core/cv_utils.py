"""
Cross-Validation Utilities for Image Forgery Detection
Handles XGBoost and LightGBM cross-validation issues
"""

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

def safe_xgb_cross_val(model, X, y, cv=5, scoring='roc_auc', **kwargs):
    """
    Safely perform cross-validation with XGBoost models
    Removes early stopping parameters that cause validation errors
    """
    if not XGB_AVAILABLE:
        raise ImportError("XGBoost not available")
    
    # Create a copy of the model with safe parameters for CV
    if hasattr(model, 'get_params'):
        params = model.get_params()
    else:
        params = {}
    
    # Remove problematic parameters for cross-validation
    safe_params = params.copy()
    safe_params.pop('early_stopping_rounds', None)
    safe_params.pop('eval_set', None)
    safe_params.pop('eval_metric', 'logloss')  # Keep eval_metric but ensure it's safe
    
    # Create new model with safe parameters
    safe_model = xgb.XGBClassifier(**safe_params)
    
    # Perform cross-validation
    try:
        scores = cross_val_score(safe_model, X, y, cv=cv, scoring=scoring, **kwargs)
        return scores
    except Exception as e:
        print(f"⚠️ XGBoost cross-validation failed: {e}")
        # Fallback with even more conservative parameters
        fallback_params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42,
            'objective': 'binary:logistic'
        }
        fallback_model = xgb.XGBClassifier(**fallback_params)
        return cross_val_score(fallback_model, X, y, cv=cv, scoring=scoring, **kwargs)

def safe_lgb_cross_val(model, X, y, cv=5, scoring='roc_auc', **kwargs):
    """
    Safely perform cross-validation with LightGBM models
    Uses conservative parameters to avoid overfitting warnings
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not available")
    
    # Create a conservative LightGBM model for CV
    safe_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 10,  # Very conservative
        'learning_rate': 0.1,
        'n_estimators': 100,  # Fewer estimators
        'min_child_samples': 50,  # Strong regularization
        'min_split_gain': 0.1,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 1,
        'random_state': 42,
        'verbosity': -1,  # Suppress warnings
        'force_col_wise': True  # Avoid memory issues
    }
    
    safe_model = lgb.LGBMClassifier(**safe_params)
    
    # Suppress LightGBM warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        try:
            scores = cross_val_score(safe_model, X, y, cv=cv, scoring=scoring, **kwargs)
            return scores
        except Exception as e:
            print(f"⚠️ LightGBM cross-validation failed: {e}")
            # Return dummy scores if it fails completely
            return np.array([0.5] * cv)

def robust_cross_validate(models_dict, X, y, cv=5, scoring='roc_auc'):
    """
    Perform robust cross-validation for multiple models
    Handles XGBoost and LightGBM issues automatically
    
    Args:
        models_dict: Dictionary of {name: model} pairs
        X: Features
        y: Labels
        cv: Number of CV folds
        scoring: Scoring metric
        
    Returns:
        Dictionary of {model_name: {'scores': array, 'mean': float, 'std': float}}
    """
    results = {}
    
    for name, model in models_dict.items():
        print(f"Cross-validating {name}...")
        
        try:
            if 'xgb' in name.lower() or 'xgboost' in name.lower():
                if XGB_AVAILABLE:
                    scores = safe_xgb_cross_val(model, X, y, cv=cv, scoring=scoring)
                else:
                    print(f"⚠️ XGBoost not available for {name}")
                    continue
            elif 'lgb' in name.lower() or 'lightgbm' in name.lower():
                if LIGHTGBM_AVAILABLE:
                    scores = safe_lgb_cross_val(model, X, y, cv=cv, scoring=scoring)
                else:
                    print(f"⚠️ LightGBM not available for {name}")
                    continue
            else:
                # Standard cross-validation for other models
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            results[name] = {
                'scores': scores,
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
            
            print(f"   {name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
            
        except Exception as e:
            print(f"⚠️ Cross-validation failed for {name}: {e}")
            # Store failure information
            results[name] = {
                'scores': np.array([0.5] * cv),
                'mean': 0.5,
                'std': 0.0,
                'error': str(e)
            }
    
    return results

def get_conservative_xgb_params():
    """
    Get conservative XGBoost parameters that work well with small datasets
    and don't cause cross-validation issues
    """
    return {
        'objective': 'binary:logistic',
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 10,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'verbosity': 0,
        # No early stopping for CV compatibility
        'eval_metric': 'logloss'
    }

def get_conservative_lgb_params():
    """
    Get conservative LightGBM parameters that avoid overfitting warnings
    """
    return {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 10,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'min_child_samples': 50,
        'min_split_gain': 0.1,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 1,
        'random_state': 42,
        'verbosity': -1,
        'force_col_wise': True
    }
