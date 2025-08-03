#!/usr/bin/env python3
"""
🚀 Optimized Image Forgery Detection Training
GPU-accelerated with CPU fallback, simplified and clean
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import pickle
import json
import time
from pathlib import Path
from tqdm import tqdm
import logging

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import configuration
from core.config import *

class OptimizedForgeryTrainer:
    """Optimized trainer with automatic GPU/CPU handling"""
    
    def __init__(self):
        self.device = DEVICE
        self.gpu_available = GPU_AVAILABLE
        self.gpu_name = GPU_NAME
        
        # Initialize models
        self.models = {}
        self.cnn_models = ['resnet50', 'efficientnet_b2', 'densenet121']
        
        logger.info(f"🎮 Device: {self.device}")
        if self.gpu_available:
            logger.info(f"🚀 GPU: {self.gpu_name}")
        else:
            logger.info("💻 Using CPU")
    
    def load_models(self):
        """Load pre-trained CNN models"""
        try:
            import timm
            import torch
            import torch.nn as nn
            
            self.loaded_models = {}
            for model_name in self.cnn_models:
                logger.info(f"Loading {model_name}...")
                model = timm.create_model(f'{model_name}.ra_in1k', pretrained=True, num_classes=0)
                model = model.to(self.device)
                model.eval()
                self.loaded_models[model_name] = model
                logger.info(f"✅ {model_name} loaded")
            
            return True
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    def extract_features_simple(self, image_paths, labels):
        """Extract features using simple, reliable method"""
        import torch
        import timm
        from PIL import Image
        import torchvision.transforms as T
        
        # Load models if not already loaded
        if not hasattr(self, 'loaded_models'):
            if not self.load_models():
                return None, None
        
        # Transform
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        all_features = []
        valid_labels = []
        
        logger.info("🔧 Extracting CNN features...")
        
        with torch.no_grad():
            for i, img_path in enumerate(tqdm(image_paths, desc="CNN Features")):
                try:
                    # Load and process image
                    image = Image.open(img_path).convert('RGB')
                    img_tensor = transform(image).unsqueeze(0).to(self.device)
                    
                    # Extract features from all models
                    features = []
                    for model_name, model in self.loaded_models.items():
                        feat = model(img_tensor).cpu().numpy().flatten()
                        features.extend(feat)
                    
                    # Add basic image features
                    basic_features = self.extract_basic_features(image)
                    features.extend(basic_features)
                    
                    all_features.append(features)
                    valid_labels.append(labels[i])
                    
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")
                    continue
        
        if all_features:
            features_array = np.array(all_features)
            logger.info(f"✅ Extracted {features_array.shape[0]} samples with {features_array.shape[1]} features")
            return features_array, np.array(valid_labels)
        else:
            return None, None
    
    def extract_basic_features(self, image):
        """Extract basic statistical features from image"""
        import cv2
        import numpy as np
        
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

def main():
    """Main training function"""
    print("=" * 80)
    print("🚀 OPTIMIZED IMAGE FORGERY DETECTION TRAINING")
    print("=" * 80)
    
    # Initialize trainer
    trainer = OptimizedForgeryTrainer()
    
    # Load dataset
    logger.info("📂 Loading dataset...")
    try:
        train_df = pd.read_csv('./data/train_labels.csv')
        val_df = pd.read_csv('./data/val_labels.csv')
        test_df = pd.read_csv('./data/test_labels.csv')
        logger.info(f"📊 Dataset: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    except Exception as e:
        logger.error(f"❌ Error loading dataset: {e}")
        return
    
    # Extract features
    logger.info("🔧 Extracting training features...")
    train_features, train_labels = trainer.extract_features_simple(
        train_df['filepath'].values, train_df['label'].values
    )
    if train_features is None:
        logger.error("❌ Failed to extract training features")
        return
    
    logger.info("🔧 Extracting validation features...")
    val_features, val_labels = trainer.extract_features_simple(
        val_df['filepath'].values, val_df['label'].values
    )
    
    logger.info("🔧 Extracting test features...")
    test_features, test_labels = trainer.extract_features_simple(
        test_df['filepath'].values, test_df['label'].values
    )
    
    # Train models
    logger.info("🎯 Training models...")
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.preprocessing import StandardScaler
    
    # Feature scaling
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)
    
    models = {}
    results = {}
    
    # Random Forest
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    start_time = time.time()
    rf.fit(train_features_scaled, train_labels)
    rf_time = time.time() - start_time
    models['rf'] = rf
    logger.info(f"✅ Random Forest trained in {rf_time:.2f}s")
    
    # Extra Trees
    logger.info("Training Extra Trees...")
    et = ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    start_time = time.time()
    et.fit(train_features_scaled, train_labels)
    et_time = time.time() - start_time
    models['et'] = et
    logger.info(f"✅ Extra Trees trained in {et_time:.2f}s")
    
    # Gradient Boosting
    logger.info("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=8, learning_rate=0.1, random_state=42)
    start_time = time.time()
    gb.fit(train_features_scaled, train_labels)
    gb_time = time.time() - start_time
    models['gb'] = gb
    logger.info(f"✅ Gradient Boosting trained in {gb_time:.2f}s")
    
    # XGBoost with GPU if available
    try:
        import xgboost as xgb
        logger.info("Training XGBoost...")
        
        xgb_params = {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Add GPU support if available
        if trainer.gpu_available:
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['gpu_id'] = 0
            logger.info("🚀 XGBoost GPU enabled")
        
        xgb_model = xgb.XGBClassifier(**xgb_params)
        start_time = time.time()
        xgb_model.fit(train_features_scaled, train_labels)
        xgb_time = time.time() - start_time
        models['xgb'] = xgb_model
        logger.info(f"✅ XGBoost trained in {xgb_time:.2f}s")
    except ImportError:
        logger.warning("⚠️ XGBoost not available")
    
    # MLP
    logger.info("Training MLP...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64) if trainer.gpu_available else (128, 64),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    start_time = time.time()
    mlp.fit(train_features_scaled, train_labels)
    mlp_time = time.time() - start_time
    models['mlp'] = mlp
    logger.info(f"✅ MLP trained in {mlp_time:.2f}s")
    
    # Evaluate models
    logger.info("📊 Evaluating models...")
    best_acc = 0
    best_model_name = ""
    best_metrics = {}
    
    for name, model in models.items():
        # Validation predictions
        val_pred = model.predict(val_features_scaled)
        val_acc = accuracy_score(val_labels, val_pred)
        val_f1 = f1_score(val_labels, val_pred, average='weighted')
        val_precision = precision_score(val_labels, val_pred, average='weighted')
        val_recall = recall_score(val_labels, val_pred, average='weighted')
        
        results[name] = {
            'accuracy': val_acc,
            'f1': val_f1,
            'precision': val_precision,
            'recall': val_recall
        }
        
        logger.info(f"✅ {name.upper()}: Acc={val_acc:.4f} F1={val_f1:.4f} Prec={val_precision:.4f} Rec={val_recall:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_name = name
            best_metrics = results[name]
    
    logger.info(f"🏆 Best model: {best_model_name.upper()}")
    logger.info(f"📊 Best metrics: {best_metrics}")
    
    # Save models
    logger.info("💾 Saving models...")
    os.makedirs('./models', exist_ok=True)
    
    # Save best model
    with open('./models/optimized_best_model.pkl', 'wb') as f:
        pickle.dump(models[best_model_name], f)
    
    # Save scaler
    with open('./models/optimized_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save all models
    with open('./models/optimized_all_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    # Save results
    with open('./models/optimized_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save configuration
    config = {
        'gpu_available': trainer.gpu_available,
        'gpu_name': trainer.gpu_name,
        'device': str(trainer.device),
        'best_model': best_model_name,
        'best_metrics': best_metrics,
        'feature_count': train_features.shape[1],
        'sample_count': len(train_labels)
    }
    with open('./models/optimized_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 80)
    print("🎉 OPTIMIZED TRAINING COMPLETED!")
    print("=" * 80)
    print(f"🎮 Device: {trainer.device}")
    print(f"🚀 GPU Used: {'✅ Yes' if trainer.gpu_available else '❌ No'}")
    if trainer.gpu_available:
        print(f"🎯 GPU: {trainer.gpu_name}")
    print(f"🏆 Best Model: {best_model_name.upper()}")
    print(f"📊 Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"📊 F1-Score: {best_metrics['f1']:.4f}")
    print(f"📊 Precision: {best_metrics['precision']:.4f}")
    print(f"📊 Recall: {best_metrics['recall']:.4f}")
    print(f"🔧 Features: {train_features.shape[1]}")
    print(f"💾 Models saved to ./models/")
    print("=" * 80)
    print("🚀 Ready for prediction with optimized_predict.py!")

if __name__ == "__main__":
    main()
