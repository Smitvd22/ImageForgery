#!/usr/bin/env python3
"""
🚀 Image Forgery Detection Training on Complete Dataset
GPU-accelerated training with comprehensive evaluation and visualization
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

import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
from PIL import Image
import cv2

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

# XGBoost with GPU support
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import configuration
from core.config import *
from core.models import TIMM_AVAILABLE

class CompleteForgeryTrainer:
    """Complete dataset trainer with GPU acceleration and comprehensive evaluation"""
    
    def __init__(self):
        self.device = DEVICE
        self.gpu_available = GPU_AVAILABLE
        self.gpu_name = GPU_NAME
        
        # Initialize results storage
        self.results = {}
        self.training_start_time = time.time()
        
        logger.info(f"🎮 Device: {self.device}")
        if self.gpu_available:
            logger.info(f"🚀 GPU: {self.gpu_name}")
            logger.info(f"📊 GPU Memory: {GPU_MEMORY:.1f} GB")
        else:
            logger.info("💻 Using CPU")
        
        # Setup directories
        os.makedirs('./models', exist_ok=True)
        os.makedirs('./results', exist_ok=True)
    
    def load_cnn_models(self):
        """Load pre-trained CNN models for feature extraction"""
        try:
            if not TIMM_AVAILABLE:
                logger.warning("⚠️ TIMM not available for advanced models")
                return False
            
            self.cnn_models = {}
            model_names = ['resnet50', 'efficientnet_b2', 'densenet121']
            
            for model_name in model_names:
                logger.info(f"Loading {model_name}...")
                model = timm.create_model(f'{model_name}.ra_in1k', pretrained=True, num_classes=0)
                model = model.to(self.device)
                model.eval()
                self.cnn_models[model_name] = model
            
            logger.info(f"✅ Loaded {len(self.cnn_models)} CNN models")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading CNN models: {e}")
            return False
    
    def extract_features_from_dataset(self, csv_path, dataset_name="Dataset"):
        """Extract comprehensive features from complete dataset"""
        logger.info(f"🔧 Extracting features from {dataset_name}...")
        
        # Load dataset
        df = pd.read_csv(csv_path)
        image_paths = df['filepath'].values
        labels = df['label'].values
        
        logger.info(f"📊 {dataset_name} size: {len(image_paths)} images")
        
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
                    logger.warning(f"⚠️ Error processing {img_path}: {e}")
                    continue
        
        if all_features:
            features_array = np.array(all_features)
            labels_array = np.array(valid_labels)
            avg_processing_time = np.mean(processing_times)
            
            logger.info(f"✅ Extracted {features_array.shape[0]} samples with {features_array.shape[1]} features")
            logger.info(f"⏱️ Average processing time: {avg_processing_time:.3f}s per image")
            
            return features_array, labels_array
        else:
            logger.error(f"❌ No features extracted from {dataset_name}")
            return None, None
    
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
        """Train multiple models with comprehensive evaluation"""
        logger.info("🎯 Training models...")
        
        # Feature scaling
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        models = {}
        results = {}
        training_times = {}
        
        # Random Forest
        logger.info("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
        start_time = time.time()
        rf.fit(features_scaled, labels)
        training_times['rf'] = time.time() - start_time
        models['rf'] = rf
        logger.info(f"✅ Random Forest trained in {training_times['rf']:.2f}s")
        
        # Extra Trees
        logger.info("Training Extra Trees...")
        et = ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
        start_time = time.time()
        et.fit(features_scaled, labels)
        training_times['et'] = time.time() - start_time
        models['et'] = et
        logger.info(f"✅ Extra Trees trained in {training_times['et']:.2f}s")
        
        # XGBoost with GPU if available
        if XGB_AVAILABLE:
            logger.info("Training XGBoost...")
            xgb_params = {
                'n_estimators': 500,
                'max_depth': 10,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Add GPU support if available
            if self.gpu_available:
                xgb_params['tree_method'] = 'gpu_hist'
                xgb_params['gpu_id'] = 0
                logger.info("🚀 XGBoost GPU enabled")
            
            xgb_model = xgb.XGBClassifier(**xgb_params)
            start_time = time.time()
            xgb_model.fit(features_scaled, labels)
            training_times['xgb'] = time.time() - start_time
            models['xgb'] = xgb_model
            logger.info(f"✅ XGBoost trained in {training_times['xgb']:.2f}s")
        
        # MLP with GPU optimization
        logger.info("Training MLP...")
        mlp = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128) if self.gpu_available else (256, 128),
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            alpha=0.001
        )
        start_time = time.time()
        mlp.fit(features_scaled, labels)
        training_times['mlp'] = time.time() - start_time
        models['mlp'] = mlp
        logger.info(f"✅ MLP trained in {training_times['mlp']:.2f}s")
        
        return models, scaler, training_times
    
    def evaluate_models(self, models, scaler, features, labels):
        """Comprehensive model evaluation with cross-validation"""
        logger.info("📊 Evaluating models...")
        
        features_scaled = scaler.transform(features)
        results = {}
        
        for name, model in models.items():
            logger.info(f"Evaluating {name.upper()}...")
            
            # Predictions
            predictions = model.predict(features_scaled)
            probabilities = model.predict_proba(features_scaled) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = self.calculate_comprehensive_metrics(labels, predictions, probabilities)
            
            # Cross-validation
            cv_scores = cross_val_score(model, features_scaled, labels, cv=5, scoring='accuracy')
            cv_f1_scores = cross_val_score(model, features_scaled, labels, cv=5, scoring='f1_weighted')
            
            metrics['cv_accuracy_mean'] = float(np.mean(cv_scores))
            metrics['cv_accuracy_std'] = float(np.std(cv_scores))
            metrics['cv_f1_mean'] = float(np.mean(cv_f1_scores))
            metrics['cv_f1_std'] = float(np.std(cv_f1_scores))
            
            results[name] = metrics
            
            logger.info(f"✅ {name.upper()}: Acc={metrics['accuracy']:.4f} F1={metrics['f1_score']:.4f} CV-Acc={metrics['cv_accuracy_mean']:.4f}±{metrics['cv_accuracy_std']:.4f}")
        
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
    
    def create_visualizations(self, results, save_dir="./results"):
        """Create comprehensive visualizations"""
        logger.info("📊 Creating visualizations...")
        
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
            plt.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/cross_validation_results.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"✅ Visualizations saved to {save_dir}/")
    
    def save_results(self, models, scaler, results, training_times, features_shape):
        """Save all models and results"""
        logger.info("💾 Saving results...")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_model = models[best_model_name]
        best_metrics = results[best_model_name]
        
        # Save best model
        with open('./models/train_best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save scaler
        with open('./models/train_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save all models
        with open('./models/train_all_models.pkl', 'wb') as f:
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
        
        with open('./results/train_complete_results.json', 'w') as f:
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
        
        logger.info(f"✅ Results saved:")
        logger.info(f"   - Best model: ./models/train_best_model.pkl")
        logger.info(f"   - All models: ./models/train_all_models.pkl")
        logger.info(f"   - Scaler: ./models/train_scaler.pkl")
        logger.info(f"   - Results: ./results/train_complete_results.json")
        
        return best_model_name, best_metrics

def main():
    """Main training function"""
    print("=" * 80)
    print("🚀 IMAGE FORGERY DETECTION - COMPLETE DATASET TRAINING")
    print("=" * 80)
    
    # Initialize trainer
    trainer = CompleteForgeryTrainer()
    
    # Load CNN models
    trainer.load_cnn_models()
    
    # Load complete dataset
    logger.info("📂 Loading complete dataset...")
    if not os.path.exists('./data/labels.csv'):
        logger.error("❌ Complete dataset CSV not found. Please ensure data/labels.csv exists.")
        return
    
    # Extract features from complete dataset
    features, labels = trainer.extract_features_from_dataset('./data/labels.csv', "Complete Dataset")
    
    if features is None:
        logger.error("❌ Failed to extract features from complete dataset")
        return
    
    # Train models
    models, scaler, training_times = trainer.train_models(features, labels)
    
    # Evaluate models
    results = trainer.evaluate_models(models, scaler, features, labels)
    
    # Create visualizations
    trainer.create_visualizations(results)
    
    # Save results
    best_model_name, best_metrics = trainer.save_results(models, scaler, results, training_times, features.shape)
    
    # Print final summary
    total_time = time.time() - trainer.training_start_time
    
    print("\n" + "=" * 80)
    print("🎉 COMPLETE DATASET TRAINING FINISHED!")
    print("=" * 80)
    print(f"🎮 Device: {trainer.device}")
    print(f"🚀 GPU Used: {'✅ Yes (' + trainer.gpu_name + ')' if trainer.gpu_available else '❌ No'}")
    print(f"📊 Dataset Size: {features.shape[0]} samples")
    print(f"🔧 Features: {features.shape[1]}")
    print(f"🏆 Best Model: {best_model_name.upper()}")
    print(f"📊 Best Accuracy: {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
    print(f"📊 Best F1-Score: {best_metrics['f1_score']:.4f}")
    print(f"📊 Best Precision: {best_metrics['precision']:.4f}")
    print(f"📊 Best Recall: {best_metrics['recall']:.4f}")
    if best_metrics['roc_auc']:
        print(f"📊 Best ROC AUC: {best_metrics['roc_auc']:.4f}")
    print(f"⏱️ Total Training Time: {total_time:.2f} seconds")
    print(f"💾 Models saved to: ./models/")
    print(f"📊 Results saved to: ./results/")
    print("=" * 80)
    
    return best_metrics['accuracy']

if __name__ == "__main__":
    accuracy = main()
