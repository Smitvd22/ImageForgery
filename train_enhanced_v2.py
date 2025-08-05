#!/usr/bin/env python3
"""
🚀 Enhanced Image Forgery Detection Training v2.0
Automatic improvements based on test results analysis
Target: 85%+ accuracy with balanced performance
"""

import os
import sys
import time
import json
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Scientific computing
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import cv2

# Machine learning
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, 
    GradientBoostingClassifier, VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# Advanced models
try:
    import xgboost as XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_training_v2.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import configuration
from core.config import *

class EnhancedForgeryTrainer:
    """Enhanced trainer with automatic improvements based on test analysis"""
    
    def __init__(self):
        self.device = DEVICE
        self.gpu_available = GPU_AVAILABLE
        self.gpu_name = GPU_NAME
        
        # Initialize models
        self.models = {}
        self.cnn_models = ['resnet50', 'efficientnet_b2', 'densenet121']
        self.loaded_models = {}
        
        # Class balancing
        self.class_weights = None
        
        logger.info(f"Device: {self.device}")
        if self.gpu_available:
            logger.info(f"GPU: {self.gpu_name}")
            # Optimize GPU memory
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        
    def load_models(self):
        """Load pre-trained CNN models"""
        try:
            import torchvision.models as models
            
            cnn_models = ['resnet50', 'efficientnet_b2', 'densenet121']
            
            for model_name in cnn_models:
                try:
                    if model_name == 'resnet50':
                        model = models.resnet50(pretrained=True)
                        model.fc = nn.Identity()  # Remove final layer
                    elif model_name == 'efficientnet_b2':
                        model = models.efficientnet_b2(pretrained=True)
                        model.classifier = nn.Identity()
                    elif model_name == 'densenet121':
                        model = models.densenet121(pretrained=True)
                        model.classifier = nn.Identity()
                    
                    model = model.to(self.device)
                    model.eval()
                    self.loaded_models[model_name] = model
                    logger.info(f"[OK] Loaded {model_name} on {self.device}")
                    
                except Exception as e:
                    logger.warning(f"[WARNING] Could not load {model_name}: {e}")
            
            return len(self.loaded_models) > 0
            
        except Exception as e:
            logger.error(f"[ERROR] Error loading CNN models: {e}")
            return False
    
    def extract_enhanced_features_batch(self, image_paths, labels, batch_size=8):
        """Enhanced feature extraction with better CNN features and augmentation"""
        all_features = []
        
        # Transform for CNN models with enhanced preprocessing
        transform_basic = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Additional transform for data augmentation during training
        transform_augmented = T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Enhanced feature extraction from {len(image_paths)} images...")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing enhanced batches"):
                batch_paths = image_paths[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
                batch_features = []
                
                for image_path, label in zip(batch_paths, batch_labels):
                    try:
                        # Load and preprocess image
                        image = Image.open(image_path).convert('RGB')
                        
                        # Extract features from multiple transformations for robustness
                        all_img_features = []
                        
                        # Basic transformation
                        image_tensor = transform_basic(image).unsqueeze(0).to(self.device)
                        
                        # Extract CNN features if available
                        if self.loaded_models:
                            for model_name, model in self.loaded_models.items():
                                try:
                                    features = model(image_tensor)
                                    features = features.view(features.size(0), -1)
                                    # Apply global average pooling to reduce dimensionality
                                    if features.shape[1] > 2048:
                                        features = torch.mean(features.view(features.size(0), -1, 1), dim=2)
                                    all_img_features.append(features.cpu().numpy().flatten())
                                except Exception as e:
                                    logger.warning(f"[WARNING] Error with {model_name}: {e}")
                        
                        # Enhanced basic features with multiple scales
                        enhanced_basic = self.extract_enhanced_basic_features(image)
                        all_img_features.extend(enhanced_basic)
                        
                        # For forged images (class 1), add augmented features for better detection
                        if label == 1 and len(all_img_features) > 0:
                            try:
                                aug_image_tensor = transform_augmented(image).unsqueeze(0).to(self.device)
                                if len(self.loaded_models) > 0:
                                    # Extract features from one CNN model for augmented version
                                    first_model = list(self.loaded_models.values())[0]
                                    aug_features = first_model(aug_image_tensor)
                                    aug_features = aug_features.view(aug_features.size(0), -1)
                                    if aug_features.shape[1] > 1024:
                                        aug_features = torch.mean(aug_features.view(aug_features.size(0), -1, 1), dim=2)
                                    # Add augmented features (weighted lower)
                                    aug_features_np = aug_features.cpu().numpy().flatten() * 0.3
                                    all_img_features.append(aug_features_np)
                            except Exception:
                                pass  # Skip if augmentation fails
                        
                        # Combine all features
                        if all_img_features:
                            combined_features = np.concatenate(all_img_features)
                        else:
                            # Fallback to basic features only
                            combined_features = np.array(enhanced_basic).flatten()
                        
                        batch_features.append(combined_features)
                        
                    except Exception as e:
                        logger.error(f"[ERROR] Error processing {image_path}: {e}")
                        # Add zero features for failed images
                        if batch_features:
                            feature_dim = len(batch_features[0])
                        else:
                            feature_dim = 4517  # Default feature count
                        batch_features.append(np.zeros(feature_dim))
                
                all_features.extend(batch_features)
                
                # Clear GPU cache periodically
                if self.gpu_available and (i + batch_size) % (batch_size * 4) == 0:
                    torch.cuda.empty_cache()
        
        if all_features:
            # Ensure all feature vectors have the same length
            min_length = min(len(f) for f in all_features)
            all_features = [f[:min_length] for f in all_features]
            return np.array(all_features)
        else:
            logger.error("[ERROR] No features extracted")
            return None
    
    def extract_enhanced_basic_features(self, image):
        """Extract enhanced basic features with multiple scales and improved descriptors"""
        # Convert PIL to numpy
        img_array = np.array(image)
        
        features = []
        
        # Multi-scale analysis
        scales = [1.0, 0.75, 0.5]
        
        for scale in scales:
            if scale != 1.0:
                h, w = img_array.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_img = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                scaled_img = img_array
            
            # Color space analysis
            gray = cv2.cvtColor(scaled_img, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(scaled_img, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(scaled_img, cv2.COLOR_RGB2LAB)
            
            # Enhanced statistics for each channel
            for channel in [scaled_img[:,:,0], scaled_img[:,:,1], scaled_img[:,:,2], gray]:
                features.extend([
                    np.mean(channel),
                    np.std(channel),
                    np.median(channel),
                    np.min(channel),
                    np.max(channel),
                    np.percentile(channel, 25),
                    np.percentile(channel, 75)
                ])
            
            # HSV features
            for channel in [hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]]:
                features.extend([
                    np.mean(channel),
                    np.std(channel)
                ])
            
            # LAB features
            for channel in [lab[:,:,0], lab[:,:,1], lab[:,:,2]]:
                features.extend([
                    np.mean(channel),
                    np.std(channel)
                ])
            
            # Enhanced edge analysis
            edges = cv2.Canny(gray, 50, 150)
            features.extend([
                np.mean(edges),
                np.std(edges),
                np.sum(edges > 0) / edges.size,  # Edge density
            ])
            
            # Texture features
            try:
                # Sobel gradients
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                features.extend([
                    np.mean(np.abs(sobel_x)),
                    np.std(sobel_x),
                    np.mean(np.abs(sobel_y)),
                    np.std(sobel_y)
                ])
                
                # Laplacian
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                features.extend([
                    np.mean(np.abs(laplacian)),
                    np.std(laplacian)
                ])
            except Exception:
                # Add zeros if texture analysis fails
                features.extend([0.0] * 6)
        
        return features
    
    def calculate_class_weights(self, labels):
        """Calculate class weights for imbalanced dataset"""
        unique_classes = np.unique(labels)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
        weight_dict = dict(zip(unique_classes, class_weights))
        
        logger.info(f"Class weights calculated: {weight_dict}")
        return weight_dict
    
    def train_enhanced_models(self, train_features, train_labels, val_features, val_labels):
        """Train enhanced models with class balancing and improved parameters"""
        
        # Calculate class weights
        self.class_weights = self.calculate_class_weights(train_labels)
        
        models = {}
        results = {}
        
        # Enhanced Random Forest with class balancing
        logger.info("Training Enhanced Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=500,  # Increased
            max_depth=20,      # Increased
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=self.class_weights,
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            max_features='sqrt'
        )
        start_time = time.time()
        rf.fit(train_features, train_labels)
        rf_time = time.time() - start_time
        models['rf'] = rf
        logger.info(f"[OK] Enhanced Random Forest trained in {rf_time:.2f}s")
        
        # Enhanced Extra Trees with class balancing
        logger.info("Training Enhanced Extra Trees...")
        et = ExtraTreesClassifier(
            n_estimators=500,  # Increased
            max_depth=20,      # Increased
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=self.class_weights,
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            max_features='sqrt'
        )
        start_time = time.time()
        et.fit(train_features, train_labels)
        et_time = time.time() - start_time
        models['et'] = et
        logger.info(f"[OK] Enhanced Extra Trees trained in {et_time:.2f}s")
        
        # Enhanced Gradient Boosting
        logger.info("Training Enhanced Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=300,  # Increased
            max_depth=12,      # Increased
            learning_rate=0.05, # Decreased for better convergence
            subsample=0.8,
            random_state=42
        )
        start_time = time.time()
        gb.fit(train_features, train_labels)
        gb_time = time.time() - start_time
        models['gb'] = gb
        logger.info(f"[OK] Enhanced Gradient Boosting trained in {gb_time:.2f}s")
        
        # Enhanced XGBoost with GPU if available
        if XGBOOST_AVAILABLE:
            logger.info("Training Enhanced XGBoost...")
            try:
                # Calculate scale_pos_weight for class imbalance
                pos_weight = len(train_labels[train_labels == 0]) / len(train_labels[train_labels == 1])
                
                xgb = XGBClassifier(
                    n_estimators=1000,  # Increased
                    max_depth=15,       # Increased
                    learning_rate=0.05, # Decreased
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=pos_weight,  # Handle class imbalance
                    tree_method='gpu_hist' if self.gpu_available else 'hist',
                    gpu_id=0 if self.gpu_available else None,
                    random_state=42,
                    early_stopping_rounds=50,
                    eval_metric=['logloss', 'auc']
                )
                
                start_time = time.time()
                xgb.fit(
                    train_features, train_labels,
                    eval_set=[(val_features, val_labels)],
                    verbose=False
                )
                xgb_time = time.time() - start_time
                models['xgb'] = xgb
                logger.info(f"[OK] Enhanced XGBoost trained in {xgb_time:.2f}s")
                
            except Exception as e:
                logger.warning(f"[WARNING] XGBoost training failed: {e}")
        
        # Enhanced MLP with better architecture
        logger.info("Training Enhanced MLP...")
        mlp = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128) if self.gpu_available else (256, 128, 64),
            max_iter=1000,  # Increased
            learning_rate_init=0.001,  # Decreased
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            alpha=0.001,  # L2 regularization
            batch_size=32 if self.gpu_available else 16
        )
        start_time = time.time()
        mlp.fit(train_features, train_labels)
        mlp_time = time.time() - start_time
        models['mlp'] = mlp
        logger.info(f"[OK] Enhanced MLP trained in {mlp_time:.2f}s")
        
        # Create Enhanced Ensemble Model
        logger.info("Creating Enhanced Ensemble Model...")
        ensemble_models = [(name, model) for name, model in models.items()]
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft',  # Use probability voting
            n_jobs=-1
        )
        start_time = time.time()
        ensemble.fit(train_features, train_labels)
        ensemble_time = time.time() - start_time
        models['ensemble'] = ensemble
        logger.info(f"[OK] Enhanced Ensemble trained in {ensemble_time:.2f}s")
        
        return models
    
    def evaluate_enhanced_models(self, models, val_features, val_labels):
        """Enhanced evaluation with focus on balanced metrics"""
        
        logger.info("Evaluating enhanced models...")
        best_acc = 0
        best_model_name = ""
        best_metrics = {}
        best_model = None
        
        for name, model in models.items():
            try:
                # Predictions
                val_pred = model.predict(val_features)
                val_prob = model.predict_proba(val_features)
                
                # Calculate metrics with focus on balanced performance
                accuracy = accuracy_score(val_labels, val_pred)
                precision = precision_score(val_labels, val_pred, average='weighted')
                recall = recall_score(val_labels, val_pred, average='weighted')
                f1 = f1_score(val_labels, val_pred, average='weighted')
                
                # Calculate per-class metrics for balance assessment
                precision_per_class = precision_score(val_labels, val_pred, average=None)
                recall_per_class = recall_score(val_labels, val_pred, average=None)
                
                # Balanced accuracy (accounts for class imbalance better)
                balanced_acc = (recall_per_class[0] + recall_per_class[1]) / 2
                
                # Custom score emphasizing balance and sensitivity
                # Penalize models with poor sensitivity (forged detection)
                sensitivity = recall_per_class[1]  # Recall for forged class
                specificity = recall_per_class[0]  # Recall for authentic class
                
                # Balanced score that emphasizes both sensitivity and overall accuracy
                balanced_score = (accuracy * 0.5) + (balanced_acc * 0.3) + (sensitivity * 0.2)
                
                results = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'balanced_accuracy': balanced_acc,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'balanced_score': balanced_score
                }
                
                logger.info(f"{name.upper()}: Acc={accuracy:.4f}, Bal_Acc={balanced_acc:.4f}, "
                           f"Sens={sensitivity:.4f}, Spec={specificity:.4f}, Score={balanced_score:.4f}")
                
                # Select best model based on balanced score (prioritizing sensitivity)
                if balanced_score > best_acc and sensitivity > 0.7:  # Minimum sensitivity threshold
                    best_acc = balanced_score
                    best_model_name = name
                    best_metrics = results
                    best_model = model
                
            except Exception as e:
                logger.error(f"[ERROR] Error evaluating {name}: {e}")
        
        return best_model, best_model_name, best_metrics

def main():
    """Main enhanced training function"""
    print("=" * 80)
    print("ENHANCED IMAGE FORGERY DETECTION TRAINING v2.0")
    print("Automatic improvements based on test analysis")
    print("=" * 80)
    
    # Initialize trainer
    trainer = EnhancedForgeryTrainer()
    
    # Load CNN models
    logger.info("Loading pre-trained models...")
    trainer.load_models()
    
    # Load dataset
    logger.info("Loading dataset...")
    try:
        # Load training data
        train_df = pd.read_csv(TRAIN_CSV)
        val_df = pd.read_csv(VAL_CSV)
        test_df = pd.read_csv(TEST_CSV)
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
    except Exception as e:
        logger.error(f"[ERROR] Error loading dataset: {e}")
        return False
    
    # Extract enhanced features
    logger.info("Extracting enhanced training features...")
    train_features = trainer.extract_enhanced_features_batch(
        train_df['filepath'].values, train_df['label'].values
    )
    if train_features is None:
        logger.error("[ERROR] Failed to extract training features")
        return False
    
    logger.info("Extracting enhanced validation features...")
    val_features = trainer.extract_enhanced_features_batch(
        val_df['filepath'].values, val_df['label'].values
    )
    if val_features is None:
        logger.error("[ERROR] Failed to extract validation features")
        return False
    
    # Feature scaling
    logger.info("Scaling features...")
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    
    # Train enhanced models
    logger.info("Training enhanced models...")
    models = trainer.train_enhanced_models(
        train_features_scaled, train_df['label'].values,
        val_features_scaled, val_df['label'].values
    )
    
    # Evaluate models
    logger.info("Evaluating enhanced models...")
    best_model, best_model_name, best_metrics = trainer.evaluate_enhanced_models(
        models, val_features_scaled, val_df['label'].values
    )
    
    if best_model is None:
        logger.error("[ERROR] No suitable model found")
        return False
    
    logger.info(f"Best model: {best_model_name.upper()}")
    logger.info(f"Best metrics: {best_metrics}")
    
    # Save enhanced models
    logger.info("Saving enhanced models...")
    os.makedirs('./models', exist_ok=True)
    
    # Save best model
    with open('./models/enhanced_v2_best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save scaler
    with open('./models/enhanced_v2_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save all models
    with open('./models/enhanced_v2_all_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    # Save results
    with open('./models/enhanced_v2_results.json', 'w') as f:
        json.dump(best_metrics, f, indent=2)
    
    # Save configuration
    config = {
        'gpu_available': trainer.gpu_available,
        'gpu_name': trainer.gpu_name,
        'device': str(trainer.device),
        'best_model': best_model_name,
        'best_metrics': best_metrics,
        'feature_count': train_features_scaled.shape[1],
        'sample_count': len(train_df['label'].values),
        'class_weights': trainer.class_weights,
        'version': '2.0_enhanced'
    }
    with open('./models/enhanced_v2_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 80)
    print("ENHANCED TRAINING v2.0 COMPLETED!")
    print("=" * 80)
    print(f"Device: {trainer.device}")
    print(f"GPU Used: {'Yes' if trainer.gpu_available else 'No'}")
    if trainer.gpu_available:
        print(f"GPU: {trainer.gpu_name}")
    print(f"Best Model: {best_model_name.upper()}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
    print(f"Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}")
    print(f"Sensitivity (Forged Detection): {best_metrics['sensitivity']:.4f}")
    print(f"Specificity (Authentic Detection): {best_metrics['specificity']:.4f}")
    print(f"F1-Score: {best_metrics['f1']:.4f}")
    print(f"Features: {train_features_scaled.shape[1]}")
    print(f"Models saved to ./models/enhanced_v2_*")
    print("=" * 80)
    print("Ready for testing with enhanced model!")
    
    return best_metrics['accuracy'] >= 0.85

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
