#!/usr/bin/env python3
"""
🚀 Complete Image Forgery Detection Process
GPU-accelerated training with proper train/validation/test splits
Comprehensive evaluation and visualization
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
    confusion_matrix, classification_report, average_precision_score
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

# Import configuration and dataset utilities
from core.config import *
from core.dataset import create_dataset_splits
from core.models import TIMM_AVAILABLE

class CompleteForgeryProcessor:
    """Complete forgery detection process with proper train/validation/test workflow"""
    
    def __init__(self):
        self.device = DEVICE
        self.gpu_available = GPU_AVAILABLE
        self.gpu_name = GPU_NAME
        
        # Initialize results storage
        self.results = {}
        self.process_start_time = time.time()
        
        logger.info(f"🎮 Device: {self.device}")
        if self.gpu_available:
            logger.info(f"🚀 GPU: {self.gpu_name}")
            logger.info(f"📊 GPU Memory: {GPU_MEMORY:.1f} GB")
        else:
            logger.info("💻 Using CPU")
        
        # Setup directories
        os.makedirs('./models', exist_ok=True)
        os.makedirs('./results', exist_ok=True)
        os.makedirs('./data', exist_ok=True)
    
    def prepare_dataset(self):
        """Prepare dataset with proper train/validation/test splits"""
        logger.info("📂 Preparing dataset with train/validation/test splits...")
        
        # Check if split files already exist
        if all(os.path.exists(path) for path in [TRAIN_CSV, VAL_CSV, TEST_CSV]):
            logger.info("✅ Dataset splits already exist")
            
            # Load and display split information
            train_df = pd.read_csv(TRAIN_CSV)
            val_df = pd.read_csv(VAL_CSV)
            test_df = pd.read_csv(TEST_CSV)
            
            logger.info(f"📊 Dataset splits:")
            logger.info(f"   - Training: {len(train_df)} samples ({len(train_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
            logger.info(f"   - Validation: {len(val_df)} samples ({len(val_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
            logger.info(f"   - Test: {len(test_df)} samples ({len(test_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
            
            return train_df, val_df, test_df
        else:
            logger.info("🔄 Creating new dataset splits...")
            
            # Create splits using the dataset utility
            train_data, val_data, test_data = create_dataset_splits(
                authentic_dir=AUTHENTIC_DIR,
                forged_dir=FORGED_DIR,
                train_split=TRAIN_SPLIT,
                val_split=VAL_SPLIT,
                test_split=TEST_SPLIT,
                random_seed=RANDOM_SEED
            )
            
            logger.info("✅ Dataset splits created successfully")
            return train_data, val_data, test_data
    
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
        """Extract comprehensive features from dataset"""
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
        """Extract basic statistical features from image"""
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
    
    def train_models(self, train_features, train_labels, val_features, val_labels):
        """Train multiple models with validation"""
        logger.info("🎯 Training models...")
        
        # Feature scaling
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        val_features_scaled = scaler.transform(val_features)
        
        models = {}
        results = {}
        training_times = {}
        
        # Random Forest
        logger.info("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
        start_time = time.time()
        rf.fit(train_features_scaled, train_labels)
        training_times['rf'] = time.time() - start_time
        models['rf'] = rf
        
        # Validation
        val_pred = rf.predict(val_features_scaled)
        val_acc = accuracy_score(val_labels, val_pred)
        results['rf'] = {'validation_accuracy': val_acc}
        logger.info(f"✅ Random Forest - Training: {training_times['rf']:.2f}s, Val Acc: {val_acc:.4f}")
        
        # Extra Trees
        logger.info("Training Extra Trees...")
        et = ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
        start_time = time.time()
        et.fit(train_features_scaled, train_labels)
        training_times['et'] = time.time() - start_time
        models['et'] = et
        
        # Validation
        val_pred = et.predict(val_features_scaled)
        val_acc = accuracy_score(val_labels, val_pred)
        results['et'] = {'validation_accuracy': val_acc}
        logger.info(f"✅ Extra Trees - Training: {training_times['et']:.2f}s, Val Acc: {val_acc:.4f}")
        
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
            xgb_model.fit(train_features_scaled, train_labels)
            training_times['xgb'] = time.time() - start_time
            models['xgb'] = xgb_model
            
            # Validation
            val_pred = xgb_model.predict(val_features_scaled)
            val_acc = accuracy_score(val_labels, val_pred)
            results['xgb'] = {'validation_accuracy': val_acc}
            logger.info(f"✅ XGBoost - Training: {training_times['xgb']:.2f}s, Val Acc: {val_acc:.4f}")
        
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
        mlp.fit(train_features_scaled, train_labels)
        training_times['mlp'] = time.time() - start_time
        models['mlp'] = mlp
        
        # Validation
        val_pred = mlp.predict(val_features_scaled)
        val_acc = accuracy_score(val_labels, val_pred)
        results['mlp'] = {'validation_accuracy': val_acc}
        logger.info(f"✅ MLP - Training: {training_times['mlp']:.2f}s, Val Acc: {val_acc:.4f}")
        
        # Select best model based on validation accuracy
        best_model_name = max(results.keys(), key=lambda x: results[x]['validation_accuracy'])
        logger.info(f"🏆 Best model on validation: {best_model_name.upper()} (Acc: {results[best_model_name]['validation_accuracy']:.4f})")
        
        return models, scaler, results, training_times, best_model_name
    
    def test_models(self, models, scaler, test_features, test_labels, best_model_name):
        """Test all models on the test set"""
        logger.info("🧪 Testing models on test set...")
        
        # Scale test features
        test_features_scaled = scaler.transform(test_features)
        
        test_results = {}
        
        for name, model in models.items():
            logger.info(f"Testing {name.upper()}...")
            
            # Predictions
            start_time = time.time()
            predictions = model.predict(test_features_scaled)
            prediction_time = time.time() - start_time
            
            # Probabilities
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(test_features_scaled)
            else:
                probabilities = None
            
            # Calculate comprehensive metrics
            metrics = self.calculate_comprehensive_metrics(test_labels, predictions, probabilities)
            metrics['prediction_time'] = prediction_time
            metrics['predictions_per_second'] = len(test_labels) / prediction_time
            metrics['is_best_model'] = (name == best_model_name)
            
            test_results[name] = metrics
            
            logger.info(f"✅ {name.upper()}: Acc={metrics['accuracy']:.4f} F1={metrics['f1_score']:.4f}")
        
        return test_results
    
    def calculate_comprehensive_metrics(self, true_labels, predictions, probabilities):
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = float(accuracy_score(true_labels, predictions))
        metrics['precision'] = float(precision_score(true_labels, predictions, average='weighted'))
        metrics['recall'] = float(recall_score(true_labels, predictions, average='weighted'))
        metrics['f1_score'] = float(f1_score(true_labels, predictions, average='weighted'))
        
        # Per-class metrics
        metrics['precision_per_class'] = precision_score(true_labels, predictions, average=None).tolist()
        metrics['recall_per_class'] = recall_score(true_labels, predictions, average=None).tolist()
        metrics['f1_per_class'] = f1_score(true_labels, predictions, average=None).tolist()
        
        # ROC AUC and Average Precision
        if probabilities is not None and probabilities.shape[1] == 2:
            metrics['roc_auc'] = float(roc_auc_score(true_labels, probabilities[:, 1]))
            metrics['average_precision'] = float(average_precision_score(true_labels, probabilities[:, 1]))
            
            # Store probabilities for ROC curve plotting
            fpr, tpr, _ = roc_curve(true_labels, probabilities[:, 1])
            metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
        else:
            metrics['roc_auc'] = None
            metrics['average_precision'] = None
            metrics['roc_curve'] = None
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Additional metrics from confusion matrix
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            metrics['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            metrics['false_negative_rate'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        
        # Classification report
        metrics['classification_report'] = classification_report(
            true_labels, predictions, 
            target_names=['Authentic', 'Forged'],
            output_dict=True
        )
        
        return metrics
    
    def create_comprehensive_visualizations(self, train_results, test_results, train_labels, val_labels, test_labels, best_model_name, save_dir="./results"):
        """Create comprehensive visualizations for the complete process"""
        logger.info("📊 Creating comprehensive visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Dataset overview
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Complete Process Overview - Dataset Analysis', fontsize=16, fontweight='bold')
        
        # Dataset size distribution
        datasets = ['Train', 'Validation', 'Test']
        sizes = [len(train_labels), len(val_labels), len(test_labels)]
        colors = ['skyblue', 'lightgreen', 'lightcoral']
        
        axes[0, 0].pie(sizes, labels=datasets, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Dataset Split Distribution')
        
        # Class distribution across splits
        train_classes = np.bincount(train_labels)
        val_classes = np.bincount(val_labels)
        test_classes = np.bincount(test_labels)
        
        x = np.arange(2)
        width = 0.25
        
        axes[0, 1].bar(x - width, train_classes, width, label='Train', alpha=0.8)
        axes[0, 1].bar(x, val_classes, width, label='Validation', alpha=0.8)
        axes[0, 1].bar(x + width, test_classes, width, label='Test', alpha=0.8)
        
        axes[0, 1].set_xlabel('Classes')
        axes[0, 1].set_ylabel('Number of Samples')
        axes[0, 1].set_title('Class Distribution Across Splits')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(['Authentic', 'Forged'])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Validation vs Test performance
        models = list(test_results.keys())
        val_accs = [train_results[model]['validation_accuracy'] for model in models]
        test_accs = [test_results[model]['accuracy'] for model in models]
        
        axes[1, 0].scatter(val_accs, test_accs, s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[1, 0].annotate(model.upper(), (val_accs[i], test_accs[i]), 
                               xytext=(5, 5), textcoords='offset points')
        
        # Perfect correlation line
        min_acc = min(min(val_accs), min(test_accs))
        max_acc = max(max(val_accs), max(test_accs))
        axes[1, 0].plot([min_acc, max_acc], [min_acc, max_acc], 'r--', alpha=0.5, label='Perfect Correlation')
        
        axes[1, 0].set_xlabel('Validation Accuracy')
        axes[1, 0].set_ylabel('Test Accuracy')
        axes[1, 0].set_title('Validation vs Test Performance')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Model performance comparison
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        model_names = [m.upper() for m in models]
        
        metric_data = np.array([[test_results[model][metric] for metric in metrics] for model in models])
        
        im = axes[1, 1].imshow(metric_data, cmap='YlOrRd', aspect='auto')
        axes[1, 1].set_xticks(range(len(metrics)))
        axes[1, 1].set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        axes[1, 1].set_yticks(range(len(models)))
        axes[1, 1].set_yticklabels(model_names)
        axes[1, 1].set_title('Model Performance Heatmap')
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(metrics)):
                axes[1, 1].text(j, i, f'{metric_data[i, j]:.3f}', 
                               ha='center', va='center', fontweight='bold')
        
        plt.colorbar(im, ax=axes[1, 1])
        plt.tight_layout()
        plt.savefig(f"{save_dir}/complete_process_overview.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Best model detailed analysis
        best_metrics = test_results[best_model_name]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Best Model Analysis - {best_model_name.upper()}', fontsize=16, fontweight='bold')
        
        # Confusion Matrix
        cm = np.array(best_metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Authentic', 'Forged'],
                   yticklabels=['Authentic', 'Forged'],
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # ROC Curve
        if best_metrics['roc_curve'] is not None:
            fpr = best_metrics['roc_curve']['fpr']
            tpr = best_metrics['roc_curve']['tpr']
            auc = best_metrics['roc_auc']
            
            axes[0, 1].plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {auc:.4f})')
            axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
            axes[0, 1].set_xlim([0.0, 1.0])
            axes[0, 1].set_ylim([0.0, 1.05])
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'ROC Curve\nNot Available', ha='center', va='center', 
                           transform=axes[0, 1].transAxes, fontsize=14)
            axes[0, 1].set_title('ROC Curve')
        
        # Per-class performance
        classes = ['Authentic', 'Forged']
        precision_per_class = best_metrics['precision_per_class']
        recall_per_class = best_metrics['recall_per_class']
        f1_per_class = best_metrics['f1_per_class']
        
        x = np.arange(len(classes))
        width = 0.25
        
        axes[0, 2].bar(x - width, precision_per_class, width, label='Precision', alpha=0.8)
        axes[0, 2].bar(x, recall_per_class, width, label='Recall', alpha=0.8)
        axes[0, 2].bar(x + width, f1_per_class, width, label='F1-Score', alpha=0.8)
        
        axes[0, 2].set_xlabel('Classes')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].set_title('Per-Class Performance')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(classes)
        axes[0, 2].legend()
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Model comparison on test set
        test_metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        
        for idx, metric in enumerate(test_metrics):
            ax = axes[1, idx] if idx < 3 else None
            if ax is None:
                continue
                
            values = [test_results[model][metric] for model in models]
            colors = ['gold' if model == best_model_name else 'skyblue' for model in models]
            
            bars = ax.bar(range(len(models)), values, color=colors, alpha=0.8)
            ax.set_xlabel('Models')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'Test {metric.replace("_", " ").title()} Comparison')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels([m.upper() for m in models], rotation=45)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Performance summary in the last subplot
        if len(test_metrics) >= 3:
            summary_metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Specificity', 'Sensitivity']
            summary_values = [
                best_metrics['accuracy'],
                best_metrics['f1_score'],
                best_metrics['precision'],
                best_metrics['recall'],
                best_metrics.get('specificity', 0),
                best_metrics.get('sensitivity', 0)
            ]
            
            bars = axes[1, 2].barh(summary_metrics, summary_values, alpha=0.8, color='lightgreen')
            axes[1, 2].set_xlabel('Score')
            axes[1, 2].set_title('Performance Summary')
            axes[1, 2].set_xlim(0, 1)
            axes[1, 2].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, summary_values):
                axes[1, 2].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                               f'{value:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/best_model_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"✅ Comprehensive visualizations saved to {save_dir}/")
    
    def save_complete_results(self, models, scaler, train_results, test_results, training_times, best_model_name, data_info):
        """Save comprehensive results from the complete process"""
        logger.info("💾 Saving complete process results...")
        
        # Find best model metrics
        best_metrics = test_results[best_model_name]
        
        # Save best model and scaler
        with open('./models/complete_best_model.pkl', 'wb') as f:
            pickle.dump(models[best_model_name], f)
        
        with open('./models/complete_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save all models
        with open('./models/complete_all_models.pkl', 'wb') as f:
            pickle.dump(models, f)
        
        # Prepare comprehensive results
        comprehensive_results = {
            'process_info': {
                'timestamp': datetime.now().isoformat(),
                'total_process_time': time.time() - self.process_start_time,
                'gpu_used': self.gpu_available,
                'gpu_name': self.gpu_name if self.gpu_available else None,
                'device': str(self.device)
            },
            'dataset_info': data_info,
            'best_model': {
                'name': best_model_name,
                'validation_accuracy': train_results[best_model_name]['validation_accuracy'],
                'test_metrics': best_metrics
            },
            'training_results': {
                'validation_scores': train_results,
                'training_times': training_times
            },
            'test_results': test_results,
            'model_comparison': {
                model: {
                    'validation_accuracy': train_results[model]['validation_accuracy'],
                    'test_accuracy': test_results[model]['accuracy'],
                    'test_f1_score': test_results[model]['f1_score'],
                    'training_time': training_times[model]
                }
                for model in models.keys()
            }
        }
        
        # Save comprehensive results
        with open('./results/complete_process_results.json', 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        # Save model comparison CSV
        comparison_data = []
        for model_name in models.keys():
            comparison_data.append({
                'model': model_name,
                'validation_accuracy': train_results[model_name]['validation_accuracy'],
                'test_accuracy': test_results[model_name]['accuracy'],
                'test_f1_score': test_results[model_name]['f1_score'],
                'test_precision': test_results[model_name]['precision'],
                'test_recall': test_results[model_name]['recall'],
                'test_roc_auc': test_results[model_name]['roc_auc'],
                'training_time_seconds': training_times[model_name],
                'predictions_per_second': test_results[model_name]['predictions_per_second'],
                'is_best_model': model_name == best_model_name
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv('./results/complete_model_comparison.csv', index=False)
        
        # Save configuration
        config = {
            'process_type': 'complete_train_val_test',
            'best_model': best_model_name,
            'best_test_accuracy': best_metrics['accuracy'],
            'dataset_splits': data_info,
            'gpu_available': self.gpu_available,
            'gpu_name': self.gpu_name,
            'device': str(self.device)
        }
        
        with open('./models/complete_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Complete process results saved:")
        logger.info(f"   - Best model: ./models/complete_best_model.pkl")
        logger.info(f"   - All models: ./models/complete_all_models.pkl")
        logger.info(f"   - Scaler: ./models/complete_scaler.pkl")
        logger.info(f"   - Results: ./results/complete_process_results.json")
        logger.info(f"   - Comparison: ./results/complete_model_comparison.csv")
        
        return best_model_name, best_metrics

def main():
    """Main function for complete process"""
    print("=" * 80)
    print("🚀 COMPLETE IMAGE FORGERY DETECTION PROCESS")
    print("🔄 Train/Validation/Test Split Workflow")
    print("=" * 80)
    
    # Initialize processor
    processor = CompleteForgeryProcessor()
    
    # Step 1: Prepare dataset with proper splits
    train_df, val_df, test_df = processor.prepare_dataset()
    
    # Step 2: Load CNN models for feature extraction
    processor.load_cnn_models()
    
    # Step 3: Extract features from all splits
    logger.info("🔧 Extracting features from all dataset splits...")
    
    train_features, train_labels = processor.extract_features_from_dataset(TRAIN_CSV, "Training Set")
    val_features, val_labels = processor.extract_features_from_dataset(VAL_CSV, "Validation Set")
    test_features, test_labels = processor.extract_features_from_dataset(TEST_CSV, "Test Set")
    
    if any(features is None for features in [train_features, val_features, test_features]):
        logger.error("❌ Failed to extract features from one or more datasets")
        return
    
    # Step 4: Train models with validation
    models, scaler, train_results, training_times, best_model_name = processor.train_models(
        train_features, train_labels, val_features, val_labels
    )
    
    # Step 5: Test models on test set
    test_results = processor.test_models(models, scaler, test_features, test_labels, best_model_name)
    
    # Step 6: Create comprehensive visualizations
    data_info = {
        'train_samples': len(train_labels),
        'val_samples': len(val_labels),
        'test_samples': len(test_labels),
        'total_samples': len(train_labels) + len(val_labels) + len(test_labels),
        'feature_count': train_features.shape[1]
    }
    
    processor.create_comprehensive_visualizations(
        train_results, test_results, train_labels, val_labels, test_labels, best_model_name
    )
    
    # Step 7: Save complete results
    final_best_model, final_best_metrics = processor.save_complete_results(
        models, scaler, train_results, test_results, training_times, best_model_name, data_info
    )
    
    # Print final comprehensive summary
    total_time = time.time() - processor.process_start_time
    
    print("\n" + "=" * 80)
    print("🎉 COMPLETE PROCESS FINISHED!")
    print("=" * 80)
    print(f"🎮 Device: {processor.device}")
    print(f"🚀 GPU Used: {'✅ Yes (' + processor.gpu_name + ')' if processor.gpu_available else '❌ No'}")
    print(f"📊 Dataset Information:")
    print(f"   - Training: {data_info['train_samples']} samples")
    print(f"   - Validation: {data_info['val_samples']} samples")
    print(f"   - Test: {data_info['test_samples']} samples")
    print(f"   - Total: {data_info['total_samples']} samples")
    print(f"   - Features: {data_info['feature_count']}")
    print(f"🏆 Best Model: {final_best_model.upper()}")
    print(f"📊 Validation Accuracy: {train_results[final_best_model]['validation_accuracy']:.4f}")
    print(f"📊 Test Results:")
    print(f"   - Accuracy: {final_best_metrics['accuracy']:.4f} ({final_best_metrics['accuracy']*100:.2f}%)")
    print(f"   - F1-Score: {final_best_metrics['f1_score']:.4f}")
    print(f"   - Precision: {final_best_metrics['precision']:.4f}")
    print(f"   - Recall: {final_best_metrics['recall']:.4f}")
    if final_best_metrics['roc_auc']:
        print(f"   - ROC AUC: {final_best_metrics['roc_auc']:.4f}")
    print(f"⚡ Prediction Speed: {final_best_metrics['predictions_per_second']:.1f} predictions/sec")
    print(f"⏱️ Total Process Time: {total_time:.2f} seconds")
    print(f"💾 Results saved to: ./models/ and ./results/")
    print("=" * 80)
    print("✅ Complete process workflow finished successfully!")
    print("🎯 Models ready for deployment and further analysis")
    print("=" * 80)
    
    return final_best_metrics['accuracy']

if __name__ == "__main__":
    accuracy = main()
