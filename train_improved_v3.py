#!/usr/bin/env python3
"""
🚀 Improved Image Forgery Detection Training v3.0
Focused improvements based on analysis with consistent feature extraction
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
    GradientBoostingClassifier, VotingClassifier,
    AdaBoostClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# Advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('improved_training_v3.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import configuration
from core.config import *

class ImprovedForgeryTrainer:
    """Improved trainer with consistent feature extraction and balanced approach"""
    
    def __init__(self):
        self.device = DEVICE
        self.gpu_available = GPU_AVAILABLE
        self.gpu_name = GPU_NAME
        
        # Initialize models
        self.loaded_models = {}
        self.class_weights = None
        
        logger.info(f"Device: {self.device}")
        if self.gpu_available:
            logger.info(f"GPU: {self.gpu_name}")
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
                        model.fc = nn.Identity()
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
    
    def extract_features_simple_consistent(self, image_paths, labels):
        """Extract features with consistent methodology - matching original training"""
        from PIL import Image
        
        # Load models if not already loaded
        if not self.loaded_models:
            self.load_models()
        
        # Transform - keep exactly same as original
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        all_features = []
        valid_labels = []
        
        logger.info("Extracting consistent CNN features...")
        
        with torch.no_grad():
            for i, (image_path, label) in tqdm(enumerate(zip(image_paths, labels)), 
                                             total=len(image_paths), desc="Processing images"):
                try:
                    # Load and preprocess image
                    image = Image.open(image_path).convert('RGB')
                    image_tensor = transform(image).unsqueeze(0).to(self.device)
                    
                    # Extract CNN features
                    cnn_features = []
                    if self.loaded_models:
                        for model_name, model in self.loaded_models.items():
                            try:
                                features = model(image_tensor)
                                features = features.view(features.size(0), -1)
                                cnn_features.append(features.cpu().numpy().flatten())
                            except Exception as e:
                                logger.warning(f"[WARNING] Error with {model_name}: {e}")
                    
                    # Extract basic features - exactly same as original
                    basic_features = self.extract_basic_features(image)
                    
                    # Combine features - exactly same methodology
                    combined_features = []
                    if cnn_features:
                        combined_features.extend(np.concatenate(cnn_features))
                    combined_features.extend(basic_features)
                    
                    all_features.append(combined_features)
                    valid_labels.append(label)
                    
                    # Clear GPU cache periodically
                    if self.gpu_available and i % 32 == 0:
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.error(f"[ERROR] Error processing {image_path}: {e}")
                    continue
        
        if all_features:
            # Ensure consistent feature length
            min_length = min(len(f) for f in all_features)
            all_features = [f[:min_length] for f in all_features]
            return np.array(all_features), np.array(valid_labels)
        else:
            logger.error("[ERROR] No features extracted")
            return None, None
    
    def extract_basic_features(self, image):
        """Extract basic statistical features - exactly matching original training"""
        # Convert PIL to numpy
        img_array = np.array(image)
        
        features = []
        
        # Convert to different color spaces
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Basic statistics for each channel - exactly same as original
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
        
        # HSV statistics - exactly same as original
        for channel in [hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]]:
            features.extend([
                np.mean(channel),
                np.std(channel)
            ])
        
        # Edge features - exactly same as original
        edges = cv2.Canny(gray, 50, 150)
        features.extend([
            np.mean(edges),
            np.std(edges),
            np.sum(edges > 0) / edges.size  # Edge density
        ])
        
        return features
    
    def calculate_class_weights(self, labels):
        """Calculate class weights for imbalanced dataset"""
        unique_classes = np.unique(labels)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
        weight_dict = {int(k): float(v) for k, v in zip(unique_classes, class_weights)}
        
        logger.info(f"Class weights calculated: {weight_dict}")
        return weight_dict
    
    def train_improved_models(self, train_features, train_labels, val_features, val_labels):
        """Train improved models with focus on balance and robustness"""
        
        # Calculate class weights
        self.class_weights = self.calculate_class_weights(train_labels)
        
        models = {}
        
        # Improved Random Forest with balanced settings
        logger.info("Training Improved Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            max_features='sqrt',
            bootstrap=True
        )
        start_time = time.time()
        rf.fit(train_features, train_labels)
        rf_time = time.time() - start_time
        models['rf'] = rf
        logger.info(f"[OK] Improved Random Forest trained in {rf_time:.2f}s")
        
        # Improved Extra Trees
        logger.info("Training Improved Extra Trees...")
        et = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            max_features='sqrt',
            bootstrap=True
        )
        start_time = time.time()
        et.fit(train_features, train_labels)
        et_time = time.time() - start_time
        models['et'] = et
        logger.info(f"[OK] Improved Extra Trees trained in {et_time:.2f}s")
        
        # Improved Gradient Boosting
        logger.info("Training Improved Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.9,
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        start_time = time.time()
        gb.fit(train_features, train_labels)
        gb_time = time.time() - start_time
        models['gb'] = gb
        logger.info(f"[OK] Improved Gradient Boosting trained in {gb_time:.2f}s")
        
        # XGBoost with careful class balancing
        if XGBOOST_AVAILABLE:
            logger.info("Training Improved XGBoost...")
            try:
                pos_weight = len(train_labels[train_labels == 0]) / len(train_labels[train_labels == 1])
                
                xgb_model = xgb.XGBClassifier(
                    n_estimators=500,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    scale_pos_weight=pos_weight,
                    tree_method='gpu_hist' if self.gpu_available else 'hist',
                    gpu_id=0 if self.gpu_available else None,
                    random_state=42,
                    early_stopping_rounds=50,
                    eval_metric=['logloss', 'auc']
                )
                
                start_time = time.time()
                xgb_model.fit(
                    train_features, train_labels,
                    eval_set=[(val_features, val_labels)],
                    verbose=False
                )
                xgb_time = time.time() - start_time
                models['xgb'] = xgb_model
                logger.info(f"[OK] Improved XGBoost trained in {xgb_time:.2f}s")
                
            except Exception as e:
                logger.warning(f"[WARNING] XGBoost training failed: {e}")
        
        # Improved MLP
        logger.info("Training Improved MLP...")
        mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=500,
            learning_rate_init=0.001,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            alpha=0.0001,
            batch_size=min(32, len(train_features) // 10)
        )
        start_time = time.time()
        mlp.fit(train_features, train_labels)
        mlp_time = time.time() - start_time
        models['mlp'] = mlp
        logger.info(f"[OK] Improved MLP trained in {mlp_time:.2f}s")
        
        # AdaBoost for diversity
        logger.info("Training AdaBoost...")
        ada = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.8,
            random_state=42
        )
        start_time = time.time()
        ada.fit(train_features, train_labels)
        ada_time = time.time() - start_time
        models['ada'] = ada
        logger.info(f"[OK] AdaBoost trained in {ada_time:.2f}s")
        
        # Logistic Regression with balanced class weights
        logger.info("Training Balanced Logistic Regression...")
        lr = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            C=1.0
        )
        start_time = time.time()
        lr.fit(train_features, train_labels)
        lr_time = time.time() - start_time
        models['lr'] = lr
        logger.info(f"[OK] Balanced Logistic Regression trained in {lr_time:.2f}s")
        
        # Balanced SVM
        logger.info("Training Balanced SVM...")
        svm = SVC(
            kernel='rbf',
            class_weight='balanced',
            random_state=42,
            probability=True,
            C=1.0,
            gamma='scale'
        )
        start_time = time.time()
        svm.fit(train_features, train_labels)
        svm_time = time.time() - start_time
        models['svm'] = svm
        logger.info(f"[OK] Balanced SVM trained in {svm_time:.2f}s")
        
        return models
    
    def evaluate_improved_models(self, models, val_features, val_labels):
        """Evaluate models with focus on balanced performance"""
        
        logger.info("Evaluating improved models...")
        best_score = 0
        best_model_name = ""
        best_metrics = {}
        best_model = None
        
        for name, model in models.items():
            try:
                # Predictions
                val_pred = model.predict(val_features)
                val_prob = model.predict_proba(val_features)
                
                # Calculate metrics
                accuracy = accuracy_score(val_labels, val_pred)
                precision = precision_score(val_labels, val_pred, average='weighted')
                recall = recall_score(val_labels, val_pred, average='weighted')
                f1 = f1_score(val_labels, val_pred, average='weighted')
                
                # Per-class metrics for balance assessment
                precision_per_class = precision_score(val_labels, val_pred, average=None)
                recall_per_class = recall_score(val_labels, val_pred, average=None)
                
                # Balanced accuracy
                balanced_acc = (recall_per_class[0] + recall_per_class[1]) / 2
                
                # Sensitivity and specificity
                sensitivity = recall_per_class[1] if len(recall_per_class) > 1 else 0  # Recall for forged class
                specificity = recall_per_class[0]  # Recall for authentic class
                
                # Custom balanced score emphasizing both classes
                # Penalty for very low sensitivity (< 0.6) or specificity (< 0.6)
                sensitivity_penalty = max(0, 0.6 - sensitivity)
                specificity_penalty = max(0, 0.6 - specificity)
                penalty = (sensitivity_penalty + specificity_penalty) * 0.5
                
                balanced_score = accuracy * 0.4 + balanced_acc * 0.4 + f1 * 0.2 - penalty
                
                results = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'balanced_accuracy': float(balanced_acc),
                    'sensitivity': float(sensitivity),
                    'specificity': float(specificity),
                    'balanced_score': float(balanced_score)
                }
                
                logger.info(f"{name.upper()}: Acc={accuracy:.4f}, Bal_Acc={balanced_acc:.4f}, "
                           f"Sens={sensitivity:.4f}, Spec={specificity:.4f}, Score={balanced_score:.4f}")
                
                # Select best model based on balanced score with minimum sensitivity threshold
                if balanced_score > best_score and sensitivity > 0.5 and specificity > 0.5:
                    best_score = balanced_score
                    best_model_name = name
                    best_metrics = results
                    best_model = model
                
            except Exception as e:
                logger.error(f"[ERROR] Error evaluating {name}: {e}")
        
        # If no model meets criteria, select best by accuracy
        if best_model is None:
            logger.warning("[WARNING] No model meets balance criteria, selecting best by accuracy")
            for name, model in models.items():
                try:
                    val_pred = model.predict(val_features)
                    accuracy = accuracy_score(val_labels, val_pred)
                    if accuracy > best_score:
                        best_score = accuracy
                        best_model_name = name
                        best_model = model
                        best_metrics = {'accuracy': float(accuracy)}
                except Exception:
                    continue
        
        return best_model, best_model_name, best_metrics

def main():
    """Main improved training function"""
    print("=" * 80)
    print("IMPROVED IMAGE FORGERY DETECTION TRAINING v3.0")
    print("Focused improvements with consistent feature extraction")
    print("=" * 80)
    
    # Initialize trainer
    trainer = ImprovedForgeryTrainer()
    
    # Load CNN models
    logger.info("Loading pre-trained models...")
    trainer.load_models()
    
    # Load dataset
    logger.info("Loading dataset...")
    try:
        train_df = pd.read_csv(TRAIN_CSV)
        val_df = pd.read_csv(VAL_CSV)
        test_df = pd.read_csv(TEST_CSV)
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
    except Exception as e:
        logger.error(f"[ERROR] Error loading dataset: {e}")
        return False
    
    # Extract consistent features
    logger.info("Extracting consistent training features...")
    train_features, train_labels = trainer.extract_features_simple_consistent(
        train_df['filepath'].values, train_df['label'].values
    )
    if train_features is None:
        logger.error("[ERROR] Failed to extract training features")
        return False
    
    logger.info("Extracting consistent validation features...")
    val_features, val_labels = trainer.extract_features_simple_consistent(
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
    
    logger.info(f"Feature dimensions: Train={train_features_scaled.shape}, Val={val_features_scaled.shape}")
    
    # Train improved models
    logger.info("Training improved models...")
    models = trainer.train_improved_models(
        train_features_scaled, train_labels,
        val_features_scaled, val_labels
    )
    
    # Evaluate models
    logger.info("Evaluating improved models...")
    best_model, best_model_name, best_metrics = trainer.evaluate_improved_models(
        models, val_features_scaled, val_labels
    )
    
    if best_model is None:
        logger.error("[ERROR] No suitable model found")
        return False
    
    logger.info(f"Best model: {best_model_name.upper()}")
    logger.info(f"Best metrics: {best_metrics}")
    
    # Save improved models
    logger.info("Saving improved models...")
    os.makedirs('./models', exist_ok=True)
    
    # Save best model
    with open('./models/improved_v3_best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save scaler
    with open('./models/improved_v3_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save all models
    with open('./models/improved_v3_all_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    # Save results
    with open('./models/improved_v3_results.json', 'w') as f:
        json.dump(best_metrics, f, indent=2)
    
    # Save configuration
    config = {
        'gpu_available': trainer.gpu_available,
        'gpu_name': trainer.gpu_name,
        'device': str(trainer.device),
        'best_model': best_model_name,
        'best_metrics': best_metrics,
        'feature_count': int(train_features_scaled.shape[1]),
        'sample_count': int(len(train_labels)),
        'class_weights': trainer.class_weights,
        'version': '3.0_improved'
    }
    with open('./models/improved_v3_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 80)
    print("IMPROVED TRAINING v3.0 COMPLETED!")
    print("=" * 80)
    print(f"Device: {trainer.device}")
    print(f"GPU Used: {'Yes' if trainer.gpu_available else 'No'}")
    if trainer.gpu_available:
        print(f"GPU: {trainer.gpu_name}")
    print(f"Best Model: {best_model_name.upper()}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
    if 'balanced_accuracy' in best_metrics:
        print(f"Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}")
        print(f"Sensitivity (Forged Detection): {best_metrics['sensitivity']:.4f}")
        print(f"Specificity (Authentic Detection): {best_metrics['specificity']:.4f}")
        print(f"F1-Score: {best_metrics['f1']:.4f}")
    print(f"Features: {train_features_scaled.shape[1]}")
    print(f"Models saved to ./models/improved_v3_*")
    print("=" * 80)
    print("Ready for testing with improved model!")
    
    return best_metrics['accuracy'] >= 0.85

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
