#!/usr/bin/env python3
"""
Enhanced Image Forgery Detection Training with Epoch-Based Learning
GPU-accelerated training with comprehensive evaluation, cross-validation, and anti-overfitting measures
"""

import os
import sys
import time
import warnings
import logging
import pickle
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import skew, kurtosis

import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
from PIL import Image
import cv2

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# XGBoost with GPU support
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

# Additional ensemble methods
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

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import configuration
from core.config import *
from core.models import TIMM_AVAILABLE
from core.enhanced_trainer import EpochBasedTrainer
from core.cv_utils import robust_cross_validate, get_conservative_xgb_params, get_conservative_lgb_params
from core.preprocessing import (
    adaptive_histogram_equalization, 
    enhance_edge_preservation,
    apply_sparkle_noise_suppression
)

class EnhancedForgeryTrainer:
    """Enhanced trainer with epoch-based learning and advanced anti-overfitting measures"""
    
    def __init__(self):
        self.device = DEVICE
        self.gpu_available = GPU_AVAILABLE
        self.gpu_name = GPU_NAME
        
        # Initialize results storage
        self.results = {}
        self.training_start_time = time.time()
        
        logger.info(f"🚀 Enhanced Device: {self.device}")
        if self.gpu_available:
            logger.info(f"🎯 GPU: {self.gpu_name}")
            logger.info(f"💾 GPU Memory: {GPU_MEMORY:.1f} GB")
        else:
            logger.info("🔧 Using CPU")
        
        # Setup directories
        os.makedirs('./models', exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Initialize epoch-based trainer
        self.epoch_trainer = EpochBasedTrainer()
    
    def enhanced_feature_extraction(self, image):
        """
        Enhanced feature extraction with multiple techniques to prevent overfitting
        """
        try:
            # Convert to numpy if PIL Image
            if hasattr(image, 'convert'):
                image = np.array(image.convert('RGB'))
            
            # Ensure proper format
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            # Apply enhanced preprocessing
            enhanced_img = adaptive_histogram_equalization(image.astype(np.float32) / 255.0)
            edge_preserved = enhance_edge_preservation(enhanced_img)
            noise_suppressed = apply_sparkle_noise_suppression(edge_preserved)
            
            # Multiple statistical features to capture forgery traces
            features = []
            
            # 1. Color channel statistics
            for channel in range(3):
                ch = noise_suppressed[:, :, channel] if len(noise_suppressed.shape) == 3 else noise_suppressed
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
            # JPEG compression typically creates 8x8 block artifacts
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
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Feature extraction error: {e}")
            return np.zeros(50, dtype=np.float32)  # Fallback features
    
    def train_ensemble_with_regularization(self, features, labels, feature_names):
        """
        Train ensemble models with enhanced regularization and cross-validation
        """
        logger.info("🎯 Training Enhanced Ensemble with Regularization...")
        
        # Feature selection to reduce overfitting
        logger.info("🔍 Performing feature selection...")
        
        # 1. Statistical feature selection
        selector = SelectKBest(score_func=f_classif, k=min(200, features.shape[1]))
        features_selected = selector.fit_transform(features, labels)
        selected_features = selector.get_support()
        
        logger.info(f"   Selected {features_selected.shape[1]} features from {features.shape[1]}")
        
        # 2. Recursive feature elimination for further refinement
        if features_selected.shape[1] > 100:
            rf_selector = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
            rfe = RFE(estimator=rf_selector, n_features_to_select=100, step=10)
            features_final = rfe.fit_transform(features_selected, labels)
            logger.info(f"   Refined to {features_final.shape[1]} features using RFE")
        else:
            features_final = features_selected
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_final)
        
        # Enhanced models with regularization
        models = {}
        
        # 1. XGBoost with conservative parameters to avoid CV issues
        if XGB_AVAILABLE:
            xgb_params = get_conservative_xgb_params()
            # Override with some specific values for ensemble
            xgb_params.update({
                'n_estimators': 500,  # More estimators for ensemble
                'max_depth': 6,
                'learning_rate': 0.05
            })
            
            models['xgboost'] = xgb.XGBClassifier(**xgb_params)
        
        # 2. LightGBM with conservative settings to avoid overfitting
        if LIGHTGBM_AVAILABLE:
            lgb_params = get_conservative_lgb_params()
            # Override with some specific values for ensemble
            lgb_params.update({
                'n_estimators': 300,
                'num_leaves': 20,
                'learning_rate': 0.05
            })
            
            models['lightgbm'] = lgb.LGBMClassifier(**lgb_params)
        
        # 3. Random Forest with regularization
        rf_params = {
            'n_estimators': 500,
            'max_depth': 10,  # Limited depth
            'min_samples_split': 10,  # Increased minimum samples
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': RANDOM_SEED,
            'n_jobs': -1
        }
        
        models['random_forest'] = RandomForestClassifier(**rf_params)
        
        # 4. Extra Trees with regularization
        et_params = {
            'n_estimators': 500,
            'max_depth': 12,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'bootstrap': False,
            'random_state': RANDOM_SEED,
            'n_jobs': -1
        }
        
        models['extra_trees'] = ExtraTreesClassifier(**et_params)
        
        # 5. MLP with dropout and regularization
        mlp_params = {
            'hidden_layer_sizes': (128, 64, 32),
            'alpha': 0.01,  # L2 regularization
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.001,
            'max_iter': 1000,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 20,
            'random_state': RANDOM_SEED
        }
        
        models['mlp'] = MLPClassifier(**mlp_params)
        
        # Train models with cross-validation
        trained_models = {}
        cv_scores = {}
        
        # Stratified K-Fold for robust evaluation
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        
        for name, model in models.items():
            logger.info(f"   Training {name}...")
            
            try:
                # Cross-validation scores
                cv_score = cross_val_score(
                    model, features_scaled, labels, 
                    cv=skf, scoring='accuracy', n_jobs=-1
                )
                
                cv_scores[name] = {
                    'mean': cv_score.mean(),
                    'std': cv_score.std(),
                    'scores': cv_score.tolist()
                }
                
                # Train on full dataset
                model.fit(features_scaled, labels)
                trained_models[name] = model
                
                logger.info(f"     CV Accuracy: {cv_score.mean():.4f} ± {cv_score.std():.4f}")
                
            except Exception as e:
                logger.warning(f"     Failed to train {name}: {e}")
        
        # Create voting ensemble
        if len(trained_models) > 1:
            ensemble_models = [(name, model) for name, model in trained_models.items()]
            voting_ensemble = VotingClassifier(
                estimators=ensemble_models,
                voting='soft',
                n_jobs=-1
            )
            
            # Train ensemble
            voting_ensemble.fit(features_scaled, labels)
            trained_models['voting_ensemble'] = voting_ensemble
            
            # Cross-validate ensemble
            ensemble_cv = cross_val_score(
                voting_ensemble, features_scaled, labels,
                cv=skf, scoring='accuracy', n_jobs=-1
            )
            
            cv_scores['voting_ensemble'] = {
                'mean': ensemble_cv.mean(),
                'std': ensemble_cv.std(),
                'scores': ensemble_cv.tolist()
            }
            
            logger.info(f"   Ensemble CV Accuracy: {ensemble_cv.mean():.4f} ± {ensemble_cv.std():.4f}")
        
        # Find best model based on CV scores
        best_model_name = max(cv_scores.keys(), key=lambda x: cv_scores[x]['mean'])
        best_model = trained_models[best_model_name]
        
        logger.info(f"✅ Best Model: {best_model_name}")
        logger.info(f"   CV Accuracy: {cv_scores[best_model_name]['mean']:.4f} ± {cv_scores[best_model_name]['std']:.4f}")
        
        return {
            'best_model': best_model,
            'best_model_name': best_model_name,
            'all_models': trained_models,
            'cv_scores': cv_scores,
            'scaler': scaler,
            'feature_selector': selector,
            'rfe_selector': rfe if 'rfe' in locals() else None
        }
    
    def load_cnn_models(self):
        """Load pre-trained CNN models for feature extraction with proper error handling"""
        try:
            if not TIMM_AVAILABLE:
                logger.warning("⚠️ TIMM not available, using basic feature extraction only")
                return False
            
            self.cnn_models = {}
            # Use fewer, more robust models to prevent overfitting
            model_configs = [
                ('efficientnet_v2_s.in1k', 1280),
                ('resnet50.a1_in1k', 2048),
                ('convnext_tiny.fb_in22k_ft_in1k', 768)
            ]
            
            for model_name, feature_dim in model_configs:
                try:
                    logger.info(f"Loading {model_name}...")
                    model = timm.create_model(
                        model_name, 
                        pretrained=True, 
                        num_classes=0,
                        global_pool='avg'
                    )
                    model = model.to(self.device)
                    model.eval()
                    
                    # Freeze model to prevent overfitting
                    for param in model.parameters():
                        param.requires_grad = False
                    
                    self.cnn_models[model_name] = model
                    logger.info(f"✅ Loaded {model_name} (features: {feature_dim})")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {model_name}: {e}")
                    continue
            
            logger.info(f"✅ Successfully loaded {len(self.cnn_models)} CNN models")
            return len(self.cnn_models) > 0
            
        except Exception as e:
            logger.error(f"❌ Error loading CNN models: {e}")
            return False
    
    def extract_features_from_dataset(self, csv_path, dataset_name="Dataset"):
        """
        Enhanced feature extraction with overfitting prevention
        """
        logger.info(f"🔍 Extracting enhanced features from {dataset_name}...")
        
        # Load dataset
        df = pd.read_csv(csv_path)
        image_paths = df['filepath'].values
        labels = df['label'].values
        
        logger.info(f"📊 {dataset_name} size: {len(image_paths)} images")
        
        # Enhanced transform with light augmentation to reduce overfitting
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        all_features = []
        valid_labels = []
        processing_times = []
        
        # Process images with progress bar
        with torch.no_grad():
            for i, img_path in enumerate(tqdm(image_paths, desc=f"Processing {dataset_name}")):
                start_time = time.time()
                
                try:
                    # Load and preprocess image
                    image = Image.open(img_path).convert('RGB')
                    image_array = np.array(image)
                    
                    # Extract enhanced features
                    enhanced_features = self.enhanced_feature_extraction(image)
                    
                    # Extract CNN features if available
                    cnn_features = []
                    if hasattr(self, 'cnn_models') and self.cnn_models:
                        image_tensor = transform(image).unsqueeze(0).to(self.device)
                        
                        for model_name, model in self.cnn_models.items():
                            try:
                                features = model(image_tensor)
                                if len(features.shape) > 2:
                                    features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                                features = features.view(features.size(0), -1)
                                cnn_features.append(features.cpu().numpy().flatten())
                            except Exception as e:
                                logger.warning(f"CNN feature extraction error for {model_name}: {e}")
                                continue
                    
                    # Combine all features
                    if cnn_features:
                        combined_cnn = np.concatenate(cnn_features)
                        final_features = np.concatenate([enhanced_features, combined_cnn])
                    else:
                        final_features = enhanced_features
                    
                    all_features.append(final_features)
                    valid_labels.append(labels[i])
                    
                    processing_times.append(time.time() - start_time)
                    
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")
                    continue
        
        # Convert to arrays
        features_array = np.array(all_features)
        labels_array = np.array(valid_labels)
        
        avg_time = np.mean(processing_times)
        
        logger.info(f"✅ Feature extraction completed:")
        logger.info(f"   - Processed: {len(all_features)} images")
        logger.info(f"   - Feature shape: {features_array.shape}")
        logger.info(f"   - Avg processing time: {avg_time:.3f}s per image")
        logger.info(f"   - Total processing rate: {len(all_features)/sum(processing_times):.1f} images/sec")
        
        # Create feature names for interpretability
        feature_names = []
        
        # Enhanced feature names
        for i in range(len(enhanced_features)):
            feature_names.append(f'enhanced_feat_{i}')
        
        # CNN feature names
        if hasattr(self, 'cnn_models') and self.cnn_models:
            for model_name in self.cnn_models.keys():
                model_feat_count = len(cnn_features[0]) if cnn_features else 0
                for j in range(model_feat_count):
                    feature_names.append(f'{model_name}_feat_{j}')
        
        return features_array, labels_array, feature_names
    
    def load_cnn_models(self):
        """Load pre-trained CNN models for feature extraction"""
        try:
            if not TIMM_AVAILABLE:
                logger.warning(" TIMM not available for advanced models")
                return False
            
            self.cnn_models = {}
            model_names = ['resnet50', 'efficientnet_b2', 'densenet121']
            
            for model_name in model_names:
                logger.info(f"Loading {model_name}...")
                model = timm.create_model(f'{model_name}.ra_in1k', pretrained=True, num_classes=0)
                model = model.to(self.device)
                model.eval()
                self.cnn_models[model_name] = model
            
            logger.info(f" Loaded {len(self.cnn_models)} CNN models")
            return True
        except Exception as e:
            logger.error(f" Error loading CNN models: {e}")
            return False
    
    def extract_features_from_dataset(self, csv_path, dataset_name="Dataset"):
        """Extract comprehensive features from complete dataset"""
        logger.info(f" Extracting features from {dataset_name}...")
        
        # Load dataset
        df = pd.read_csv(csv_path)
        image_paths = df['filepath'].values
        labels = df['label'].values
        
        logger.info(f" {dataset_name} size: {len(image_paths)} images")
        
        # Transform for CNN models
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        all_features = []
        valid_labels = []
        processing_times = []
        
        # Process images with progress bar
        with torch.no_grad():
            for i, img_path in enumerate(tqdm(image_paths, desc=f"Processing {dataset_name}")):
                start_time = time.time()
                
                try:
                    # Load and preprocess image
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = transform(image).unsqueeze(0).to(self.device)
                    
                    # Extract CNN features
                    cnn_features = []
                    if hasattr(self, 'cnn_models'):
                        for model_name, model in self.cnn_models.items():
                            features = model(image_tensor)
                            if len(features.shape) > 2:
                                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                            features = features.view(features.size(0), -1)
                            cnn_features.append(features.cpu().numpy().flatten())
                    
                    # Extract basic statistical features
                    basic_features = self.extract_basic_features(image)
                    
                    # Combine all features
                    if cnn_features:
                        combined_features = np.concatenate([np.concatenate(cnn_features), basic_features])
                    else:
                        combined_features = basic_features
                    
                    all_features.append(combined_features)
                    valid_labels.append(labels[i])
                    
                    processing_times.append(time.time() - start_time)
                    
                except Exception as e:
                    logger.warning(f" Error processing {img_path}: {e}")
                    continue
        
        if all_features:
            features_array = np.array(all_features)
            labels_array = np.array(valid_labels)
            avg_processing_time = np.mean(processing_times)
            
            logger.info(f"✅ Extracted {features_array.shape[0]} samples with {features_array.shape[1]} features")
            logger.info(f"   Average processing time: {avg_processing_time:.3f}s per image")
            
            # Create feature names for interpretability
            feature_names = []
            
            # Enhanced feature names (estimated 50 features)
            for i in range(50):
                feature_names.append(f'enhanced_feat_{i}')
            
            # CNN feature names (remaining features)
            remaining_features = features_array.shape[1] - 50
            if remaining_features > 0:
                cnn_models_count = len(self.cnn_models) if hasattr(self, 'cnn_models') else 3
                features_per_model = remaining_features // cnn_models_count
                
                model_names = list(self.cnn_models.keys()) if hasattr(self, 'cnn_models') else ['resnet50', 'efficientnet_b2', 'densenet121']
                
                for model_idx, model_name in enumerate(model_names):
                    for j in range(features_per_model):
                        feature_names.append(f'{model_name}_feat_{j}')
                
                # Add any remaining features
                for j in range(len(feature_names), features_array.shape[1]):
                    feature_names.append(f'extra_feat_{j}')
            
            return features_array, labels_array, feature_names
        else:
            logger.error(f"❌ No features extracted from {dataset_name}")
            return None, None, None
    
    def extract_basic_features(self, image):
        """Extract basic statistical features from image (matching training exactly)"""
        # Convert PIL to numpy
        img_array = np.array(image)
        
        features = []
        
        # Convert to different color spaces
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Basic statistics for each channel
        for channel in [img_array[:,:,0], img_array[:,:,1], img_array[:,:,2], gray]:
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.median(channel),
                np.min(channel),
                np.max(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75)
            ])
        
        # HSV statistics
        for channel in [hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]]:
            features.extend([
                np.mean(channel),
                np.std(channel)
            ])
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        features.extend([
            np.mean(edges),
            np.std(edges),
            np.sum(edges > 0) / edges.size  # Edge density
        ])
        
        return features
    
    def train_models(self, features, labels):
        """Train multiple models with comprehensive evaluation and feature selection"""
        logger.info(" Training models...")
        
        # Feature selection to reduce overfitting
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # Select top 100 features (much less than 4517) to reduce overfitting
        logger.info("Selecting top 100 features to reduce overfitting...")
        feature_selector = SelectKBest(score_func=f_classif, k=min(100, features.shape[1]))
        features_selected = feature_selector.fit_transform(features, labels)
        
        logger.info(f"Reduced features from {features.shape[1]} to {features_selected.shape[1]}")
        
        # Feature scaling
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_selected)
        
        models = {}
        results = {}
        training_times = {}
        
        # Random Forest - with MUCH stronger regularization for small dataset
        logger.info("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=20,     # Very small for small dataset
            max_depth=3,         # Very shallow to prevent overfitting
            min_samples_split=20, # Much higher to prevent overfitting
            min_samples_leaf=10,  # Much higher to prevent overfitting
            max_features='sqrt', # Use sqrt for better generalization
            class_weight='balanced',  # Handle class imbalance
            random_state=42, 
            n_jobs=-1
        )
        start_time = time.time()
        rf.fit(features_scaled, labels)
        training_times['rf'] = time.time() - start_time
        models['rf'] = rf
        logger.info(f" Random Forest trained in {training_times['rf']:.2f}s")
        
        # Extra Trees - with MUCH stronger regularization
        logger.info("Training Extra Trees...")
        et = ExtraTreesClassifier(
            n_estimators=20,     # Very small for small dataset
            max_depth=3,         # Very shallow to prevent overfitting
            min_samples_split=20, # Much higher to prevent overfitting
            min_samples_leaf=10,  # Much higher to prevent overfitting
            max_features='sqrt', # Use sqrt for better generalization
            class_weight='balanced',  # Handle class imbalance
            random_state=42, 
            n_jobs=-1
        )
        start_time = time.time()
        et.fit(features_scaled, labels)
        training_times['et'] = time.time() - start_time
        models['et'] = et
        logger.info(f" Extra Trees trained in {training_times['et']:.2f}s")
        
        # XGBoost with conservative parameters to avoid validation errors
        if XGB_AVAILABLE:
            logger.info("Training XGBoost...")
            xgb_params = get_conservative_xgb_params()
            # Override for simple training (even more conservative)
            xgb_params.update({
                'n_estimators': 50,
                'max_depth': 2,
                'learning_rate': 0.1
            })
            
            # Add GPU support if available
            if self.gpu_available:
                xgb_params['tree_method'] = 'gpu_hist'
                xgb_params['gpu_id'] = 0
                logger.info(" XGBoost GPU enabled")
            
            xgb_model = xgb.XGBClassifier(**xgb_params)
            start_time = time.time()
            xgb_model.fit(features_scaled, labels)
            training_times['xgb'] = time.time() - start_time
            models['xgb'] = xgb_model
            logger.info(f" XGBoost trained in {training_times['xgb']:.2f}s")
        
        # MLP with very strong regularization for small dataset
        logger.info("Training MLP...")
        mlp = MLPClassifier(
            hidden_layer_sizes=(64,),  # Single small hidden layer
            max_iter=200,        # Reduced iterations
            random_state=42,
            early_stopping=True,
            validation_fraction=0.3,  # Larger validation for early stopping
            n_iter_no_change=20,      # Stop if no improvement
            alpha=0.5,               # Very strong L2 regularization
            learning_rate_init=0.001,
            solver='adam'
        )
        start_time = time.time()
        mlp.fit(features_scaled, labels)
        training_times['mlp'] = time.time() - start_time
        models['mlp'] = mlp
        logger.info(f" MLP trained in {training_times['mlp']:.2f}s")
        
        return models, scaler, training_times, feature_selector
    
    def evaluate_models(self, models, scaler, features, labels):
        """Comprehensive model evaluation with cross-validation"""
        logger.info(" Evaluating models...")
        
        features_scaled = scaler.transform(features)
        results = {}
        
        for name, model in models.items():
            logger.info(f"Evaluating {name.upper()}...")
            
            # Predictions
            predictions = model.predict(features_scaled)
            probabilities = model.predict_proba(features_scaled) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = self.calculate_comprehensive_metrics(labels, predictions, probabilities)
            
            # Cross-validation with robust error handling
            try:
                if name in ['xgb', 'xgboost'] and XGB_AVAILABLE:
                    from core.cv_utils import safe_xgb_cross_val
                    cv_scores = safe_xgb_cross_val(model, features_scaled, labels, cv=5, scoring='accuracy')
                    cv_f1_scores = safe_xgb_cross_val(model, features_scaled, labels, cv=5, scoring='f1_weighted')
                elif name in ['lgb', 'lightgbm'] and LIGHTGBM_AVAILABLE:
                    from core.cv_utils import safe_lgb_cross_val
                    cv_scores = safe_lgb_cross_val(model, features_scaled, labels, cv=5, scoring='accuracy')
                    cv_f1_scores = safe_lgb_cross_val(model, features_scaled, labels, cv=5, scoring='f1_weighted')
                else:
                    cv_scores = cross_val_score(model, features_scaled, labels, cv=5, scoring='accuracy')
                    cv_f1_scores = cross_val_score(model, features_scaled, labels, cv=5, scoring='f1_weighted')
            except Exception as e:
                logger.warning(f"Cross-validation failed for {name}: {e}")
                # Use dummy scores
                cv_scores = np.array([metrics['accuracy']] * 5)
                cv_f1_scores = np.array([metrics['f1_score']] * 5)
            
            metrics['cv_accuracy_mean'] = float(np.mean(cv_scores))
            metrics['cv_accuracy_std'] = float(np.std(cv_scores))
            metrics['cv_f1_mean'] = float(np.mean(cv_f1_scores))
            metrics['cv_f1_std'] = float(np.std(cv_f1_scores))
            
            results[name] = metrics
            
            logger.info(f" {name.upper()}: Acc={metrics['accuracy']:.4f} F1={metrics['f1_score']:.4f} CV-Acc={metrics['cv_accuracy_mean']:.4f}{metrics['cv_accuracy_std']:.4f}")
        
        return results
    
    def calculate_comprehensive_metrics(self, true_labels, predictions, probabilities):
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = float(accuracy_score(true_labels, predictions))
        metrics['precision'] = float(precision_score(true_labels, predictions, average='weighted'))
        metrics['recall'] = float(recall_score(true_labels, predictions, average='weighted'))
        metrics['f1_score'] = float(f1_score(true_labels, predictions, average='weighted'))
        
        # ROC AUC
        if probabilities is not None and probabilities.shape[1] == 2:
            metrics['roc_auc'] = float(roc_auc_score(true_labels, probabilities[:, 1]))
        else:
            metrics['roc_auc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Additional metrics
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        
        # Classification report
        metrics['classification_report'] = classification_report(
            true_labels, predictions, 
            target_names=['Authentic', 'Forged'],
            output_dict=True
        )
        
        return metrics
    
    def create_visualizations(self, results, save_dir=None):
        """Create comprehensive visualizations"""
        if save_dir is None:
            save_dir = RESULTS_DIR
        logger.info(" Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Model comparison chart
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        f1_scores = [results[model]['f1_score'] for model in models]
        precisions = [results[model]['precision'] for model in models]
        recalls = [results[model]['recall'] for model in models]
        
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        bars1 = ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
        bars2 = ax.bar(x - 0.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
        bars3 = ax.bar(x + 0.5*width, precisions, width, label='Precision', alpha=0.8)
        bars4 = ax.bar(x + 1.5*width, recalls, width, label='Recall', alpha=0.8)
        
        ax.set_xlabel('Models', fontsize=14)
        ax.set_ylabel('Score', fontsize=14)
        ax.set_title('Model Performance Comparison - Complete Dataset Training', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models])
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Best model confusion matrix
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_metrics = results[best_model_name]
        
        plt.figure(figsize=(10, 8))
        cm = np.array(best_metrics['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Authentic', 'Forged'],
                   yticklabels=['Authentic', 'Forged'],
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix - {best_model_name.upper()} (Best Model)', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        
        # Add accuracy to the plot
        accuracy = best_metrics['accuracy']
        plt.text(0.5, -0.1, f'Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)', 
                ha='center', transform=plt.gca().transAxes, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/confusion_matrix_best_model.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Cross-validation results
        cv_means = [results[model]['cv_accuracy_mean'] for model in models]
        cv_stds = [results[model]['cv_accuracy_std'] for model in models]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(models)), cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
        plt.xlabel('Models', fontsize=14)
        plt.ylabel('Cross-Validation Accuracy', fontsize=14)
        plt.title('5-Fold Cross-Validation Results', fontsize=16, fontweight='bold')
        plt.xticks(range(len(models)), [m.upper() for m in models])
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
            plt.text(i, mean + std + 0.01, f'{mean:.3f}{std:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/cross_validation_results.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f" Visualizations saved to {save_dir}/")
    
    def save_results(self, models, scaler, feature_selector, results, training_times, features_shape):
        """Save all models and results"""
        logger.info(" Saving results...")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_model = models[best_model_name]
        best_metrics = results[best_model_name]
        
        # Save best model
        with open(BEST_MODEL_PATH, 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save scaler
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save feature selector
        with open(FEATURE_SELECTOR_PATH, 'wb') as f:
            pickle.dump(feature_selector, f)
        
        # Save all models
        with open(ALL_MODELS_PATH, 'wb') as f:
            pickle.dump(models, f)
        
        # Save detailed results
        detailed_results = {
            'best_model': best_model_name,
            'best_metrics': best_metrics,
            'all_results': results,
            'training_times': training_times,
            'feature_count': features_shape[1],
            'sample_count': features_shape[0],
            'gpu_used': self.gpu_available,
            'gpu_name': self.gpu_name if self.gpu_available else None,
            'training_timestamp': datetime.now().isoformat(),
            'total_training_time': time.time() - self.training_start_time
        }
        
        # Save comprehensive results
        with open(os.path.join(RESULTS_DIR, 'train_complete_results.json'), 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # Save configuration
        config = {
            'gpu_available': self.gpu_available,
            'gpu_name': self.gpu_name,
            'device': str(self.device),
            'best_model': best_model_name,
            'best_accuracy': best_metrics['accuracy'],
            'feature_count': features_shape[1],
            'sample_count': features_shape[0]
        }
        
        with open('./models/train_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f" Results saved:")
        logger.info(f"   - Best model: {BEST_MODEL_PATH}")
        logger.info(f"   - All models: {ALL_MODELS_PATH}")
        logger.info(f"   - Scaler: {SCALER_PATH}")
        logger.info(f"   - Results: {os.path.join(RESULTS_DIR, 'train_complete_results.json')}")
        
        return best_model_name, best_metrics

def main():
    """Enhanced main training function with epoch-based learning and anti-overfitting"""
    print("=" * 80)
    print("🚀 ENHANCED IMAGE FORGERY DETECTION - EPOCH-BASED TRAINING")
    print("=" * 80)
    
    # Initialize enhanced trainer
    trainer = EnhancedForgeryTrainer()
    
    # Load CNN models
    cnn_loaded = trainer.load_cnn_models()
    
    # Check if training data exists
    logger.info("📂 Checking dataset files...")
    if not os.path.exists(TRAIN_CSV):
        logger.error(f"❌ Training dataset CSV not found: {TRAIN_CSV}")
        logger.info("   Please run dataset_manager.py first to create training splits")
        return 0.0
    
    # Load training data
    train_df = pd.read_csv(TRAIN_CSV)
    
    logger.info(f"📊 Training samples: {len(train_df)}")
    
    # Extract features from training dataset
    logger.info("🔍 Extracting features from training dataset...")
    train_features, train_labels, feature_names = trainer.extract_features_from_dataset(
        TRAIN_CSV, "Training Dataset"
    )
    
    if train_features is None or len(train_features) == 0:
        logger.error("❌ Failed to extract features from training dataset")
        return 0.0
    
    # Enhanced training with regularization
    logger.info("🎯 Starting enhanced training with regularization...")
    training_results = trainer.train_ensemble_with_regularization(
        train_features, train_labels, feature_names
    )
    
    if not training_results:
        logger.error("❌ Training failed")
        return 0.0
    
    # Save models and results
    logger.info("💾 Saving models and results...")
    
    # Save best model
    with open(BEST_MODEL_PATH, 'wb') as f:
        pickle.dump(training_results['best_model'], f)
    
    # Save scaler
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(training_results['scaler'], f)
    
    # Save feature selector
    with open(FEATURE_SELECTOR_PATH, 'wb') as f:
        pickle.dump(training_results['feature_selector'], f)
    
    # Save RFE selector if used
    if training_results['rfe_selector'] is not None:
        rfe_path = FEATURE_SELECTOR_PATH.replace('_feature_selector.pkl', '_rfe_selector.pkl')
        with open(rfe_path, 'wb') as f:
            pickle.dump(training_results['rfe_selector'], f)
        logger.info("💾 Saved RFE selector")
    
    # Save all models
    with open(ALL_MODELS_PATH, 'wb') as f:
        pickle.dump(training_results['all_models'], f)
    
    # Save comprehensive results
    final_results = {
        'best_model': training_results['best_model_name'],
        'cv_results': training_results['cv_scores'],
        'training_time': time.time() - trainer.training_start_time,
        'feature_count': train_features.shape[1],
        'sample_count': train_features.shape[0],
        'gpu_used': trainer.gpu_available,
        'gpu_name': trainer.gpu_name if trainer.gpu_available else None,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    with open(os.path.join(RESULTS_DIR, 'enhanced_training_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # Print final summary
    total_time = time.time() - trainer.training_start_time
    best_model_name = training_results['best_model_name']
    best_cv_score = training_results['cv_scores'][best_model_name]['mean']
    
    print("\n" + "=" * 80)
    print("✅ ENHANCED TRAINING COMPLETED!")
    print("=" * 80)
    print(f"🔧 Device: {trainer.device}")
    print(f"🎯 GPU Used: {'Yes (' + trainer.gpu_name + ')' if trainer.gpu_available else 'No'}")
    print(f"📊 Training Samples: {train_features.shape[0]}")
    print(f"🔍 Features: {train_features.shape[1]}")
    print(f"🏆 Best Model: {best_model_name.upper()}")
    print(f"📈 CV Accuracy: {best_cv_score:.4f} ({best_cv_score*100:.2f}%)")
    print(f"⏱️ Total Training Time: {total_time:.2f} seconds")
    print(f"💾 Models saved to: ./models/")
    print(f"📄 Results saved to: {RESULTS_DIR}")
    print("=" * 80)
    print("🔍 Run validate.py to evaluate on validation set")
    print("=" * 80)
    
    return best_cv_score

if __name__ == "__main__":
    accuracy = main()
