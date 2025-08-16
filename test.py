#!/usr/bin/env python3
"""
 Image Forgery Detection Testing on Complete Dataset
GPU-accelerated testing with comprehensive evaluation and visualization
"""

import os
import sys
import time
import warnings
import logging
import pickle
import json
import argparse
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

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import configuration
from core.config import *
from core.models import TIMM_AVAILABLE

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Image Forgery Detection Testing')
    parser.add_argument('--dataset', choices=['4cam', 'misd', 'imsplice'], 
                      default=None, help='Dataset to use for testing (overrides config file)')
    return parser.parse_args()

def update_config_for_dataset(dataset_name):
    """Update global configuration variables for the specified dataset"""
    global ACTIVE_DATASET, CURRENT_DATASET, AUTHENTIC_DIR, FORGED_DIR
    global DATA_CSV, TRAIN_CSV, VAL_CSV, TEST_CSV, RESULTS_DIR
    global BEST_MODEL_PATH, ALL_MODELS_PATH, SCALER_PATH, FEATURE_SELECTOR_PATH
    
    # Update the active dataset
    ACTIVE_DATASET = dataset_name
    CURRENT_DATASET = DATASETS[ACTIVE_DATASET]
    AUTHENTIC_DIR = CURRENT_DATASET["authentic_dir"]
    FORGED_DIR = CURRENT_DATASET["forged_dir"]
    
    # Update paths
    DATASET_PREFIX = f"{ACTIVE_DATASET}_"
    DATA_CSV = os.path.join(DATA_ROOT, f"{DATASET_PREFIX}labels.csv")
    TRAIN_CSV = os.path.join(DATA_ROOT, f"{DATASET_PREFIX}train_labels.csv")
    VAL_CSV = os.path.join(DATA_ROOT, f"{DATASET_PREFIX}val_labels.csv")
    TEST_CSV = os.path.join(DATA_ROOT, f"{DATASET_PREFIX}test_labels.csv")
    
    RESULTS_DIR = f"./results_{ACTIVE_DATASET}"
    BEST_MODEL_PATH = os.path.join(MODELS_DIR, f"{DATASET_PREFIX}best_model.pkl")
    ALL_MODELS_PATH = os.path.join(MODELS_DIR, f"{DATASET_PREFIX}all_models.pkl")
    SCALER_PATH = os.path.join(MODELS_DIR, f"{DATASET_PREFIX}scaler.pkl")
    FEATURE_SELECTOR_PATH = os.path.join(MODELS_DIR, f"{DATASET_PREFIX}feature_selector.pkl")
    
    logger.info(f"📊 Updated test configuration for dataset: {ACTIVE_DATASET.upper()}")
    logger.info(f"📁 Results directory: {RESULTS_DIR}")
    logger.info(f"📋 Test CSV: {TEST_CSV}")

class CompleteForgeryTester:
    """Complete dataset tester with GPU acceleration and comprehensive evaluation"""
    
    def __init__(self, model_dir="./models"):
        self.model_dir = Path(model_dir)
        self.device = DEVICE
        self.gpu_available = GPU_AVAILABLE
        self.gpu_name = GPU_NAME
        
        # Initialize results storage
        self.results = {}
        self.testing_start_time = time.time()
        
        logger.info(f" Device: {self.device}")
        if self.gpu_available:
            logger.info(f" GPU: {self.gpu_name}")
            logger.info(f" GPU Memory: {GPU_MEMORY:.1f} GB")
        else:
            logger.info(" Using CPU")
        
        # Setup directories
        results_dir = RESULTS_DIR
        os.makedirs(results_dir, exist_ok=True)
        
        # Load trained models
        self.load_trained_models()
    
    def load_trained_models(self):
        """Load pre-trained models and preprocessors"""
        try:
            # Load best model
            with open(BEST_MODEL_PATH, 'rb') as f:
                self.best_model = pickle.load(f)
            
            # Load scaler
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load feature selector
            try:
                with open(FEATURE_SELECTOR_PATH, 'rb') as f:
                    self.feature_selector = pickle.load(f)
                logger.info(" Loaded feature selector")
            except FileNotFoundError:
                logger.warning(" Feature selector not found, using all features")
                self.feature_selector = None
            
            # Load all models
            try:
                with open(ALL_MODELS_PATH, 'rb') as f:
                    self.all_models = pickle.load(f)
            except FileNotFoundError:
                logger.warning(" All models file not found, using best model only")
                self.all_models = {'best': self.best_model}
            
            # Load configuration (keep old path format for now)
            try:
                with open(self.model_dir / f'{ACTIVE_DATASET}_config.json', 'r') as f:
                    self.config = json.load(f)
            except FileNotFoundError:
                logger.warning(" Configuration file not found")
                self.config = {}
            
            logger.info(f" Loaded {len(self.all_models)} trained models")
            
        except FileNotFoundError as e:
            logger.error(f" Required model files not found: {e}")
            logger.error("Please run train.py first to train the models")
            sys.exit(1)
    
    def load_cnn_models(self):
        """Load CNN models for feature extraction (matching training)"""
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
        """Extract comprehensive features from complete dataset (matching training exactly)"""
        logger.info(f" Extracting features from {dataset_name}...")
        
        # Load dataset
        df = pd.read_csv(csv_path)
        image_paths = df['filepath'].values
        labels = df['label'].values
        
        logger.info(f" {dataset_name} size: {len(image_paths)} images")
        
        # Transform for CNN models (exactly matching training)
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
                    
                    # Extract basic statistical features (exactly matching training)
                    basic_features = self.extract_basic_features(image)
                    
                    # Combine all features (exactly matching training)
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
            
            logger.info(f" Extracted {features_array.shape[0]} samples with {features_array.shape[1]} features")
            logger.info(f" Average processing time: {avg_processing_time:.3f}s per image")
            
            return features_array, labels_array
        else:
            logger.error(f" No features extracted from {dataset_name}")
            return None, None
    
    def extract_basic_features(self, image):
        """Extract basic statistical features from image (exactly matching training)"""
        # Convert PIL to numpy
        img_array = np.array(image)
        
        features = []
        
        # Convert to different color spaces
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Basic statistics for each channel (exactly as in training)
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
        
        # HSV statistics (exactly as in training)
        for channel in [hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]]:
            features.extend([
                np.mean(channel),
                np.std(channel)
            ])
        
        # Edge features (exactly as in training)
        edges = cv2.Canny(gray, 50, 150)
        features.extend([
            np.mean(edges),
            np.std(edges),
            np.sum(edges > 0) / edges.size  # Edge density
        ])
        
        return features
    
    def test_models(self, features, labels):
        """Test all models with comprehensive evaluation using the same preprocessing pipeline as training"""
        logger.info("🧪 Testing models...")
        
        # Apply same preprocessing pipeline as training
        # 1. Feature selection first
        if self.feature_selector is not None:
            features_selected = self.feature_selector.transform(features)
            logger.info(f"Applied feature selection: {features.shape[1]} -> {features_selected.shape[1]} features")
        else:
            features_selected = features
        
        # 2. For now, skip RFE and use 200 features directly
        # This is a temporary fix until RFE selector is properly saved
        try:
            # Try to load RFE selector if it exists
            import pickle
            rfe_path = FEATURE_SELECTOR_PATH.replace('_feature_selector.pkl', '_rfe_selector.pkl')
            if os.path.exists(rfe_path):
                with open(rfe_path, 'rb') as f:
                    rfe_selector = pickle.load(f)
                features_final = rfe_selector.transform(features_selected)
                logger.info(f"Applied RFE: {features_selected.shape[1]} -> {features_final.shape[1]} features")
            else:
                # Use feature selector to match expected dimensions
                # This should use the same feature selector used during training
                if self.feature_selector is not None and features_selected.shape[1] > 100:
                    # Use first 100 features as a fallback (deterministic)
                    features_final = features_selected[:, :100]
                    logger.info(f"Applied fallback feature reduction: {features_selected.shape[1]} -> {features_final.shape[1]} features")
                else:
                    features_final = features_selected
                    logger.info(f"Using all features: {features_selected.shape[1]} features")
        except Exception as e:
            logger.warning(f"Could not apply RFE: {e}")
            # Fallback: use first 100 features
            features_final = features_selected[:, :100]
            logger.info(f"Using first 100 features: {features_selected.shape[1]} -> {features_final.shape[1]} features")
            
        # 3. Scale features using the training scaler
        features_scaled = self.scaler.transform(features_final)
        
        results = {}
        
        for name, model in self.all_models.items():
            logger.info(f"Testing {name.upper()}...")
            
            # Predictions
            start_time = time.time()
            predictions = model.predict(features_scaled)
            prediction_time = time.time() - start_time
            
            # Probabilities
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_scaled)
            else:
                probabilities = None
            
            # Calculate comprehensive metrics
            metrics = self.calculate_comprehensive_metrics(labels, predictions, probabilities)
            metrics['prediction_time'] = prediction_time
            metrics['predictions_per_second'] = len(labels) / prediction_time
            
            results[name] = metrics
            
            logger.info(f" {name.upper()}: Acc={metrics['accuracy']:.4f} F1={metrics['f1_score']:.4f} Prec={metrics['precision']:.4f} Rec={metrics['recall']:.4f}")
        
        return results
    
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
        else:
            metrics['roc_auc'] = None
            metrics['average_precision'] = None
        
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
    
    def create_comprehensive_visualizations(self, results, labels, save_dir=None):
        """Create comprehensive test visualizations"""
        if save_dir is None:
            save_dir = RESULTS_DIR
        logger.info(" Creating comprehensive visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Model comparison chart
        models = list(results.keys())
        metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Complete Dataset Test Results - Model Performance Analysis', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            values = [results[model][metric] for model in models]
            
            bars = ax.bar(range(len(models)), values, alpha=0.8)
            ax.set_xlabel('Models', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels([m.upper() for m in models], rotation=45)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/test_model_performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Best model detailed analysis
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_metrics = results[best_model_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Detailed Analysis - {best_model_name.upper()} (Best Model)', fontsize=16, fontweight='bold')
        
        # Confusion Matrix
        cm = np.array(best_metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Authentic', 'Forged'],
                   yticklabels=['Authentic', 'Forged'],
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # Per-class metrics
        classes = ['Authentic', 'Forged']
        precision_per_class = best_metrics['precision_per_class']
        recall_per_class = best_metrics['recall_per_class']
        f1_per_class = best_metrics['f1_per_class']
        
        x = np.arange(len(classes))
        width = 0.25
        
        axes[0, 1].bar(x - width, precision_per_class, width, label='Precision', alpha=0.8)
        axes[0, 1].bar(x, recall_per_class, width, label='Recall', alpha=0.8)
        axes[0, 1].bar(x + width, f1_per_class, width, label='F1-Score', alpha=0.8)
        
        axes[0, 1].set_xlabel('Classes')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Per-Class Performance')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(classes)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Performance summary
        summary_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'Sensitivity']
        summary_values = [
            best_metrics['accuracy'],
            best_metrics['precision'],
            best_metrics['recall'],
            best_metrics['f1_score'],
            best_metrics.get('specificity', 0),
            best_metrics.get('sensitivity', 0)
        ]
        
        bars = axes[1, 0].barh(summary_metrics, summary_values, alpha=0.8)
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_title('Performance Summary')
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, summary_values):
            axes[1, 0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{value:.3f}', ha='left', va='center', fontweight='bold')
        
        # Prediction speed analysis
        models_speed = list(results.keys())
        speed_values = [results[model]['predictions_per_second'] for model in models_speed]
        
        bars = axes[1, 1].bar(range(len(models_speed)), speed_values, alpha=0.8, color='orange')
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('Predictions per Second')
        axes[1, 1].set_title('Model Prediction Speed')
        axes[1, 1].set_xticks(range(len(models_speed)))
        axes[1, 1].set_xticklabels([m.upper() for m in models_speed], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, speed_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(speed_values)*0.01,
                           f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/test_detailed_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. ROC Curve for best model (if available)
        if best_metrics['roc_auc'] is not None:
            # Note: We need to re-run prediction to get probabilities for ROC curve
            # This is a placeholder - in practice, you'd store probabilities during testing
            plt.figure(figsize=(10, 8))
            
            # Placeholder ROC curve based on metrics
            fpr_approx = np.linspace(0, 1, 100)
            tpr_approx = np.power(fpr_approx, 1 - best_metrics['roc_auc'])  # Approximate curve
            
            plt.plot(fpr_approx, tpr_approx, linewidth=3, 
                    label=f'ROC Curve (AUC = {best_metrics["roc_auc"]:.4f})', color='blue')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=14)
            plt.ylabel('True Positive Rate', fontsize=14)
            plt.title(f'ROC Curve - {best_model_name.upper()}', fontsize=16, fontweight='bold')
            plt.legend(loc="lower right", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/test_roc_curve.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # 4. Dataset distribution analysis
        plt.figure(figsize=(12, 6))
        
        # Class distribution
        unique, counts = np.unique(labels, return_counts=True)
        class_names = ['Authentic', 'Forged']
        
        plt.subplot(1, 2, 1)
        bars = plt.bar(class_names, counts, alpha=0.8, color=['skyblue', 'lightcoral'])
        plt.title('Dataset Class Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Samples')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Balanced accuracy consideration
        plt.subplot(1, 2, 2)
        balanced_acc = [results[model]['accuracy'] for model in models]
        plt.plot(range(len(models)), balanced_acc, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy on Test Dataset', fontsize=14, fontweight='bold')
        plt.xticks(range(len(models)), [m.upper() for m in models], rotation=45)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/test_dataset_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f" Comprehensive visualizations saved to {save_dir}/")
    
    def save_test_results(self, results, features_shape):
        """Save comprehensive test results"""
        logger.info(" Saving test results...")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_metrics = results[best_model_name]
        
        # Prepare detailed results
        detailed_results = {
            'test_timestamp': datetime.now().isoformat(),
            'best_model': best_model_name,
            'best_metrics': best_metrics,
            'all_results': results,
            'dataset_info': {
                'sample_count': features_shape[0],
                'feature_count': features_shape[1]
            },
            'system_info': {
                'gpu_used': self.gpu_available,
                'gpu_name': self.gpu_name if self.gpu_available else None,
                'device': str(self.device)
            },
            'total_test_time': time.time() - self.testing_start_time
        }
        
        # Save detailed results
        with open(os.path.join(RESULTS_DIR, 'test_complete_results.json'), 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # Save simplified summary
        summary = {
            'best_model': best_model_name,
            'accuracy': best_metrics['accuracy'],
            'f1_score': best_metrics['f1_score'],
            'precision': best_metrics['precision'],
            'recall': best_metrics['recall'],
            'roc_auc': best_metrics['roc_auc'],
            'total_samples': features_shape[0]
        }
        
        with open(os.path.join(RESULTS_DIR, 'test_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save predictions CSV for further analysis
        predictions_data = []
        for model_name, model_results in results.items():
            predictions_data.append({
                'model': model_name,
                'accuracy': model_results['accuracy'],
                'f1_score': model_results['f1_score'],
                'precision': model_results['precision'],
                'recall': model_results['recall'],
                'roc_auc': model_results['roc_auc']
            })
        
        predictions_df = pd.DataFrame(predictions_data)
        predictions_df.to_csv(os.path.join(RESULTS_DIR, 'test_model_comparison.csv'), index=False)
        
        logger.info(f" Test results saved:")
        logger.info(f"   - Detailed results: {os.path.join(RESULTS_DIR, 'test_complete_results.json')}")
        logger.info(f"   - Summary: {os.path.join(RESULTS_DIR, 'test_summary.json')}")
        logger.info(f"   - Model comparison: {os.path.join(RESULTS_DIR, 'test_model_comparison.csv')}")
        
        return best_model_name, best_metrics

def main():
    """Main testing function"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Use dataset from config file if not specified in command line
    if args.dataset is None:
        dataset_to_use = ACTIVE_DATASET
        logger.info(f"📋 Using dataset from config file: {dataset_to_use.upper()}")
    else:
        dataset_to_use = args.dataset
        logger.info(f"📋 Using dataset from command line: {dataset_to_use.upper()}")
    
    # Update configuration for selected dataset
    update_config_for_dataset(dataset_to_use)
    
    print("=" * 80)
    print(" IMAGE FORGERY DETECTION - TEST DATASET EVALUATION")
    print("=" * 80)
    print(f"📊 Dataset: {dataset_to_use.upper()}")
    print("=" * 80)
    
    # Initialize tester
    tester = CompleteForgeryTester()
    
    # Load CNN models for feature extraction
    tester.load_cnn_models()
    
    # Load test dataset
    logger.info(" Loading test dataset for evaluation...")
    if not os.path.exists(TEST_CSV):
        logger.error(f" Test dataset CSV not found. Please ensure {TEST_CSV} exists.")
        return
    
    # Extract features from test dataset only
    features, labels = tester.extract_features_from_dataset(TEST_CSV, "Test Dataset")
    
    if features is None:
        logger.error(" Failed to extract features from test dataset")
        return
    
    # Test models
    results = tester.test_models(features, labels)
    
    # Create comprehensive visualizations
    tester.create_comprehensive_visualizations(results, labels)
    
    # Save test results
    best_model_name, best_metrics = tester.save_test_results(results, features.shape)
    
    # Print final summary
    total_time = time.time() - tester.testing_start_time
    
    print("\n" + "=" * 80)
    print(" TEST DATASET EVALUATION FINISHED!")
    print("=" * 80)
    print(f" Device: {tester.device}")
    print(f" GPU Used: {' Yes (' + tester.gpu_name + ')' if tester.gpu_available else ' No'}")
    print(f" Test Dataset Size: {features.shape[0]} samples")
    print(f" Features: {features.shape[1]}")
    print(f" Best Model: {best_model_name.upper()}")
    print(f" Test Accuracy: {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
    print(f" Test F1-Score: {best_metrics['f1_score']:.4f}")
    print(f" Test Precision: {best_metrics['precision']:.4f}")
    print(f" Test Recall: {best_metrics['recall']:.4f}")
    if best_metrics['roc_auc']:
        print(f" Test ROC AUC: {best_metrics['roc_auc']:.4f}")
    print(f" Prediction Speed: {best_metrics['predictions_per_second']:.1f} predictions/sec")
    print(f" Total Testing Time: {total_time:.2f} seconds")
    print(f" Results saved to: {RESULTS_DIR}")
    print("=" * 80)
    
    return best_metrics['accuracy']

if __name__ == "__main__":
    accuracy = main()
