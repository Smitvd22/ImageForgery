
import os
import time
import json
import pickle
import logging
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from scipy.stats import skew, kurtosis

import torch
import torchvision.transforms as T

# Scikit-learn
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Import other libraries if available
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

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

# Local imports
from core.config import (
    RANDOM_SEED, CV_FOLDS, RESULTS_DIR, BEST_MODEL_PATH,
    ALL_MODELS_PATH, SCALER_PATH, FEATURE_SELECTOR_PATH, TRAIN_CSV, CV_CONFIG
)
from core.preprocessing import comprehensive_noise_suppression
from core.cv_utils import get_conservative_xgb_params, get_conservative_lgb_params
from core.hyperparameter_tuning import optimize_xgboost, optimize_lightgbm, optimize_random_forest

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedForgeryTrainer:
    """
    Enhanced trainer for image forgery detection with advanced features,
    regularization, and robust evaluation.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_available = torch.cuda.is_available()
        self.gpu_name = torch.cuda.get_device_name(0) if self.gpu_available else "N/A"
        self.training_start_time = time.time()
        self.cnn_models = {}

        logger.info(f"Enhanced Trainer initialized on device: {self.device}")
        if self.gpu_available:
            logger.info(f"GPU: {self.gpu_name}")

    def enhanced_feature_extraction(self, image):
        """
        Extracts a rich set of features designed to capture forgery artifacts,
        with added robustness against common image variations.
        Includes patch-based, frequency domain, and color inconsistency features.
        """
        try:
            # Ensure image is in RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')

            img_array = np.array(image) / 255.0  # Normalize to [0, 1]

            # Apply comprehensive noise suppression
            noise_suppressed = comprehensive_noise_suppression(img_array)

            features = []

            # 1. Color and statistical features from noise-suppressed image
            for ch_idx in range(3):
                ch = noise_suppressed[:, :, ch_idx]
                features.extend([
                    np.mean(ch), np.std(ch), np.var(ch),
                    np.percentile(ch, 25), np.percentile(ch, 75),
                    np.min(ch), np.max(ch),
                    skew(ch.flatten()),
                    kurtosis(ch.flatten())
                ])

            # 2. Texture features using Local Binary Patterns
            gray = cv2.cvtColor((noise_suppressed * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

            # Gradient-based features
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)

            features.extend([
                np.mean(grad_mag), np.std(grad_mag),
                np.percentile(grad_mag, 90),
                np.sum(grad_mag > np.percentile(grad_mag, 95)) / grad_mag.size
            ])

            # 3. Edge density features
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            features.append(edge_density)

            # 4. Frequency domain features
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)

            features.extend([
                np.mean(magnitude_spectrum),
                np.std(magnitude_spectrum),
                np.percentile(magnitude_spectrum, 90)
            ])

            # 5. Compression artifacts detection
            block_size = 8
            h, w = gray.shape
            block_variance = []
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    block_variance.append(np.var(block))
            if block_variance:
                features.extend([
                    np.mean(block_variance),
                    np.std(block_variance),
                    np.max(block_variance)
                ])
            else:
                features.extend([0, 0, 0])

            # 6. Patch-based features (mean, std, color inconsistency)
            patch_size = 32
            for i in range(0, h - patch_size, patch_size):
                for j in range(0, w - patch_size, patch_size):
                    patch = noise_suppressed[i:i+patch_size, j:j+patch_size, :]
                    for ch in range(3):
                        features.append(np.mean(patch[:,:,ch]))
                        features.append(np.std(patch[:,:,ch]))
                    # Color inconsistency (mean diff between channels)
                    features.append(np.mean(np.abs(patch[:,:,0] - patch[:,:,1])))
                    features.append(np.mean(np.abs(patch[:,:,1] - patch[:,:,2])))
                    features.append(np.mean(np.abs(patch[:,:,0] - patch[:,:,2])))

            # 7. Multi-scale frequency domain features
            for scale in [1, 2, 4]:
                scaled_gray = cv2.resize(gray, (gray.shape[1]//scale, gray.shape[0]//scale))
                f_transform = np.fft.fft2(scaled_gray)
                f_shift = np.fft.fftshift(f_transform)
                magnitude_spectrum = np.log(np.abs(f_shift) + 1)
                features.extend([
                    np.mean(magnitude_spectrum),
                    np.std(magnitude_spectrum)
                ])

                # 8. Higher-order statistics (entropy, energy, contrast)
                entropy = -np.sum(gray/255.0 * np.log2(gray/255.0 + 1e-12))
                energy = np.sum(gray**2)
                contrast = np.std(gray)
                features.extend([entropy, energy, contrast])

                # 9. GLCM texture features
                try:
                    import skimage.feature
                    glcm = skimage.feature.greycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
                    contrast_glcm = skimage.feature.greycoprops(glcm, 'contrast')[0, 0]
                    dissimilarity_glcm = skimage.feature.greycoprops(glcm, 'dissimilarity')[0, 0]
                    homogeneity_glcm = skimage.feature.greycoprops(glcm, 'homogeneity')[0, 0]
                    ASM_glcm = skimage.feature.greycoprops(glcm, 'ASM')[0, 0]
                    energy_glcm = skimage.feature.greycoprops(glcm, 'energy')[0, 0]
                    features.extend([contrast_glcm, dissimilarity_glcm, homogeneity_glcm, ASM_glcm, energy_glcm])
                except Exception as e:
                    features.extend([0, 0, 0, 0, 0])

                # 10. Advanced frequency features (high/low band ratios)
                f_transform = np.fft.fft2(gray)
                f_shift = np.fft.fftshift(f_transform)
                magnitude_spectrum = np.abs(f_shift)
                center = magnitude_spectrum[magnitude_spectrum.shape[0]//4:magnitude_spectrum.shape[0]*3//4,
                                            magnitude_spectrum.shape[1]//4:magnitude_spectrum.shape[1]*3//4]
                high_band = magnitude_spectrum - center
                features.extend([
                    np.mean(center), np.std(center), np.mean(high_band), np.std(high_band),
                    np.mean(center)/np.mean(high_band + 1e-6)
                ])

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.warning(f"Feature extraction error: {e}")
            return np.zeros(100, dtype=np.float32)  # Fallback features

    def train_ensemble_with_regularization(self, features, labels, feature_names):
        """
        Train ensemble models with enhanced regularization and cross-validation
        """
        logger.info("🎯 Training Enhanced Ensemble with Regularization...")

        # Feature selection to reduce overfitting
        logger.info("🔎 Performing feature selection...")

        # 1. Statistical feature selection
        selector = SelectKBest(score_func=f_classif, k=min(200, features.shape[1]))
        features_selected = selector.fit_transform(features, labels)
        
        logger.info(f"   Selected {features_selected.shape[1]} features from {features.shape[1]}")

        # 2. Recursive feature elimination for further refinement
        if features_selected.shape[1] > 100:
            rf_selector = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
            rfe = RFE(estimator=rf_selector, n_features_to_select=100, step=10)
            features_final = rfe.fit_transform(features_selected, labels)
            logger.info(f"   Refined to {features_final.shape[1]} features using RFE")
        else:
            features_final = features_selected
            rfe = None

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_final)

        # Handle class imbalance with class weights
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}

        # Enhanced models with regularization
        models = {}

        # Hyperparameter tuning
        logger.info("Tuning hyperparameters for top models...")
        best_xgb_params = optimize_xgboost(features_scaled, labels)
        best_lgb_params = optimize_lightgbm(features_scaled, labels)
        best_rf_params = optimize_random_forest(features_scaled, labels)

        models['xgboost'] = xgb.XGBClassifier(**best_xgb_params, scale_pos_weight=class_weights[1]/class_weights[0])
        models['lightgbm'] = lgb.LGBMClassifier(**best_lgb_params, class_weight=class_weight_dict)
        models['random_forest'] = RandomForestClassifier(**best_rf_params, class_weight=class_weight_dict)

        # 4. Extra Trees with regularization
        et_params = {
            'n_estimators': 500, 'max_depth': 12, 'min_samples_split': 10,
            'min_samples_leaf': 5, 'max_features': 'sqrt', 'bootstrap': False,
            'random_state': RANDOM_SEED, 'n_jobs': -1
        }
        models['extra_trees'] = ExtraTreesClassifier(**et_params)

        # 5. MLP with dropout and regularization
        mlp_params = {
            'hidden_layer_sizes': (128, 64, 32), 'alpha': 0.01,
            'learning_rate': 'adaptive', 'learning_rate_init': 0.001,
            'max_iter': 1000, 'early_stopping': True, 'validation_fraction': 0.1,
            'n_iter_no_change': 20, 'random_state': RANDOM_SEED
        }
        models['mlp'] = MLPClassifier(**mlp_params)

        # Add CNN/Transformer models as scikit-learn wrappers
        if TIMM_AVAILABLE and hasattr(self, 'cnn_models') and self.cnn_models:
            from sklearn.base import BaseEstimator, ClassifierMixin
            class TimmWrapper(BaseEstimator, ClassifierMixin):
                def __init__(self, timm_model, feature_dim):
                    self.timm_model = timm_model
                    self.feature_dim = feature_dim
                def fit(self, X, y):
                    return self  # Pretrained, no fitting
                def predict(self, X):
                    import torch
                    X_tensor = torch.tensor(X.transpose(0,3,1,2), dtype=torch.float32).to('cuda')
                    with torch.no_grad():
                        logits = self.timm_model(X_tensor)
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                    return preds
            for model_name, timm_model in self.cnn_models.items():
                models[model_name] = TimmWrapper(timm_model, feature_dim=None)

        # Train models with cross-validation
        trained_models = {}
        cv_scores = {}
        skf = StratifiedKFold(n_splits=CV_CONFIG['cv_folds'], shuffle=True, random_state=RANDOM_SEED)
        
        for name, model in models.items():
            logger.info(f"   Training {name}...")
            try:
                cv_score = cross_val_score(model, features_scaled, labels, cv=skf, scoring='accuracy', n_jobs=-1)
                cv_scores[name] = {'mean': cv_score.mean(), 'std': cv_score.std(), 'scores': cv_score.tolist()}
                model.fit(features_scaled, labels)
                trained_models[name] = model
                logger.info(f"     CV Accuracy: {cv_score.mean():.4f} ± {cv_score.std():.4f}")
            except Exception as e:
                logger.warning(f"     Failed to train {name}: {e}")

            # Model calibration (Platt scaling)
            from sklearn.calibration import CalibratedClassifierCV
            calibrated_models = {}
            for name, model in trained_models.items():
                try:
                    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=skf)
                    calibrated.fit(features_scaled, labels)
                    calibrated_models[name] = calibrated
                    logger.info(f"   Calibrated {name} with Platt scaling.")
                except Exception as e:
                    logger.warning(f"   Calibration failed for {name}: {e}")
            trained_models.update(calibrated_models)

        # Create and train advanced stacking/blending meta-ensembles
        if len(trained_models) > 1:
            ensemble_models = list(trained_models.items())

            # 1. Logistic Regression meta-learner
            stacking_lr = StackingClassifier(
                estimators=ensemble_models,
                final_estimator=LogisticRegression(),
                cv=skf
            )
            stacking_lr.fit(features_scaled, labels)
            trained_models['stacking_lr'] = stacking_lr
            stacking_lr_cv = cross_val_score(stacking_lr, features_scaled, labels, cv=skf, scoring='accuracy', n_jobs=-1)
            cv_scores['stacking_lr'] = {
                'mean': stacking_lr_cv.mean(),
                'std': stacking_lr_cv.std(),
                'scores': stacking_lr_cv.tolist()
            }
            logger.info(f"   Stacking LR Ensemble CV Accuracy: {stacking_lr_cv.mean():.4f} ± {stacking_lr_cv.std():.4f}")

            # 2. XGBoost meta-learner
            if XGB_AVAILABLE:
                stacking_xgb = StackingClassifier(
                    estimators=ensemble_models,
                    final_estimator=xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                    cv=skf
                )
                stacking_xgb.fit(features_scaled, labels)
                trained_models['stacking_xgb'] = stacking_xgb
                stacking_xgb_cv = cross_val_score(stacking_xgb, features_scaled, labels, cv=skf, scoring='accuracy', n_jobs=-1)
                cv_scores['stacking_xgb'] = {
                    'mean': stacking_xgb_cv.mean(),
                    'std': stacking_xgb_cv.std(),
                    'scores': stacking_xgb_cv.tolist()
                }
                logger.info(f"   Stacking XGB Ensemble CV Accuracy: {stacking_xgb_cv.mean():.4f} ± {stacking_xgb_cv.std():.4f}")

            # 3. LightGBM meta-learner
            if LIGHTGBM_AVAILABLE:
                stacking_lgb = StackingClassifier(
                    estimators=ensemble_models,
                    final_estimator=lgb.LGBMClassifier(),
                    cv=skf
                )
                stacking_lgb.fit(features_scaled, labels)
                trained_models['stacking_lgb'] = stacking_lgb
                stacking_lgb_cv = cross_val_score(stacking_lgb, features_scaled, labels, cv=skf, scoring='accuracy', n_jobs=-1)
                cv_scores['stacking_lgb'] = {
                    'mean': stacking_lgb_cv.mean(),
                    'std': stacking_lgb_cv.std(),
                    'scores': stacking_lgb_cv.tolist()
                }
                logger.info(f"   Stacking LGB Ensemble CV Accuracy: {stacking_lgb_cv.mean():.4f} ± {stacking_lgb_cv.std():.4f}")

        # Find best model
        best_model_name = max(cv_scores, key=lambda k: cv_scores[k]['mean'])
        best_model = trained_models[best_model_name]
        logger.info(f"✅ Best Model: {best_model_name}")
        logger.info(f"   CV Accuracy: {cv_scores[best_model_name]['mean']:.4f} ± {cv_scores[best_model_name]['std']:.4f}")

        return {
            'best_model': best_model, 'best_model_name': best_model_name,
            'all_models': trained_models, 'cv_scores': cv_scores,
            'scaler': scaler, 'feature_selector': selector,
            'rfe_selector': rfe
        }

    def load_cnn_models(self):
        """Load pre-trained CNN models for feature extraction."""
        if not TIMM_AVAILABLE:
            logger.warning("⚠️ TIMM not available, using basic feature extraction only")
            return False
        
        self.cnn_models = {}
        model_configs = [
            ('efficientnet_v2_s.in1k', 1280),
            ('efficientnet_v2_l.in1k', 1280),
            ('resnet50.a1_in1k', 2048),
            ('convnext_tiny.fb_in22k_ft_in1k', 768),
            ('convnext_large.fb_in22k_ft_in1k', 1536),
            ('swin_large_patch4_window12_384.ms_in22k_ft_in1k', 1536),
            ('vit_base_patch16_384', 768)
        ]

        for model_name, feature_dim in model_configs:
            try:
                logger.info(f"Loading {model_name}...")
                model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='avg')
                model = model.to(self.device)
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                self.cnn_models[model_name] = model
                logger.info(f"✅ Loaded {model_name} (features: {feature_dim})")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {model_name}: {e}")
        
        logger.info(f"✅ Successfully loaded {len(self.cnn_models)} CNN models")
        return len(self.cnn_models) > 0

    def extract_features_from_dataset(self, csv_path, dataset_name="Dataset"):
        """Extracts features from the dataset using enhanced and CNN-based methods."""
        logger.info(f"🔎 Extracting enhanced features from {dataset_name}...")
        df = pd.read_csv(csv_path)
        image_paths, labels = df['filepath'].values, df['label'].values
        logger.info(f"📊 {dataset_name} size: {len(image_paths)} images")

        transform = T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        all_features, valid_labels, processing_times = [], [], []
        with torch.no_grad():
            for i, img_path in enumerate(tqdm(image_paths, desc=f"Processing {dataset_name}")):
                start_time = time.time()
                try:
                    image = Image.open(img_path).convert('RGB')
                    enhanced_features = self.enhanced_feature_extraction(image)
                    
                    cnn_features = []
                    if self.cnn_models:
                        image_tensor = transform(image).unsqueeze(0).to(self.device)
                        for model in self.cnn_models.values():
                            features = model(image_tensor).view(1, -1)
                            cnn_features.append(features.cpu().numpy().flatten())
                    
                    final_features = np.concatenate([enhanced_features] + cnn_features) if cnn_features else enhanced_features
                    all_features.append(final_features)
                    valid_labels.append(labels[i])
                    processing_times.append(time.time() - start_time)
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")

        features_array, labels_array = np.array(all_features), np.array(valid_labels)
        avg_time = np.mean(processing_times) if processing_times else 0
        logger.info(f"✅ Feature extraction completed: Processed {len(all_features)} images, Shape: {features_array.shape}, Avg time: {avg_time:.3f}s")
        
        # Create feature names
        feature_names = [f'enhanced_feat_{i}' for i in range(enhanced_features.shape[0])]
        for model_name in self.cnn_models.keys():
            # This part is tricky as feature dimensions vary. A placeholder approach:
            feature_names.extend([f'{model_name}_feat_{j}' for j in range(self.cnn_models[model_name].num_features)])

        return features_array, labels_array, feature_names

    def save_results(self, training_results, train_features):
        """Saves all models, scaler, selectors, and comprehensive results."""
        logger.info("💾 Saving models and results...")
        
        with open(BEST_MODEL_PATH, 'wb') as f:
            pickle.dump(training_results['best_model'], f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(training_results['scaler'], f)
        with open(FEATURE_SELECTOR_PATH, 'wb') as f:
            pickle.dump(training_results['feature_selector'], f)
        if training_results.get('rfe_selector'):
            rfe_path = FEATURE_SELECTOR_PATH.replace('_feature_selector.pkl', '_rfe_selector.pkl')
            with open(rfe_path, 'wb') as f:
                pickle.dump(training_results['rfe_selector'], f)
        with open(ALL_MODELS_PATH, 'wb') as f:
            pickle.dump(training_results['all_models'], f)

        final_results = {
            'best_model': training_results['best_model_name'],
            'cv_results': training_results['cv_scores'],
            'training_time': time.time() - self.training_start_time,
            'feature_count': train_features.shape[1],
            'sample_count': train_features.shape[0],
            'gpu_used': self.gpu_available,
            'gpu_name': self.gpu_name,
            'timestamp': datetime.now().isoformat()
        }
        with open(os.path.join(RESULTS_DIR, 'enhanced_training_results.json'), 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
            
        logger.info(f"   - Best model: {BEST_MODEL_PATH}")
        logger.info(f"   - All models: {ALL_MODELS_PATH}")
        logger.info(f"   - Scaler: {SCALER_PATH}")
        logger.info(f"   - Results: {os.path.join(RESULTS_DIR, 'enhanced_training_results.json')}")
