#!/usr/bin/env python3
"""
 Image Forgery Detection Validation Script
Proper validation on validation set to monitor overfitting
"""

import os
import sys
import json
import time
import warnings
import logging
import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

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
    parser = argparse.ArgumentParser(description='Image Forgery Detection Validation')
    parser.add_argument('--dataset', choices=['4cam', 'misd', 'imsplice'], 
                      default=None, help='Dataset to use for validation (overrides config file)')
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
    
    logger.info(f"📊 Updated validation configuration for dataset: {ACTIVE_DATASET.upper()}")
    logger.info(f"📁 Results directory: {RESULTS_DIR}")
    logger.info(f"📋 Validation CSV: {VAL_CSV}")

class ForgeryValidator:
    """Validation on held-out validation set"""
    
    def __init__(self, model_dir="./models"):
        self.model_dir = Path(model_dir)
        self.device = DEVICE
        self.gpu_available = GPU_AVAILABLE
        self.gpu_name = GPU_NAME
        
        logger.info(f" Device: {self.device}")
        if self.gpu_available:
            logger.info(f" GPU: {self.gpu_name}")
            logger.info(f" GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            logger.info(" Using CPU")
        
        # Load trained models
        self.load_trained_models()
    
    def load_trained_models(self):
        """Load pre-trained models and preprocessors"""
        try:
            # Load all models using dataset-specific paths
            with open(ALL_MODELS_PATH, 'rb') as f:
                self.models = pickle.load(f)
            
            # Load scaler
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load feature selector
            try:
                with open(FEATURE_SELECTOR_PATH, 'rb') as f:
                    self.feature_selector = pickle.load(f)
                logger.info(" Feature selector loaded")
            except FileNotFoundError:
                logger.warning(" Feature selector not found, using all features")
                self.feature_selector = None
            
            # Load RFE selector
            try:
                with open(RFE_SELECTOR_PATH, 'rb') as f:
                    self.rfe_selector = pickle.load(f)
                logger.info(" RFE selector loaded")
            except FileNotFoundError:
                logger.warning(" RFE selector not found, will not apply RFE")
                self.rfe_selector = None
            
            logger.info(f" Loaded {len(self.models)} trained models")
            
        except FileNotFoundError as e:
            logger.error(f" Model files not found: {e}")
            logger.error("Please run train.py first to train the models.")
            sys.exit(1)
    
    def load_cnn_models(self):
        """Load CNN models for feature extraction (matching training)"""
        try:
            import timm
            self.cnn_models = {}
            
            # Load same models as training
            cnn_model_names = ['resnet50', 'efficientnet_b2', 'densenet121']
            
            for model_name in cnn_model_names:
                logger.info(f"Loading {model_name}...")
                model = timm.create_model(model_name, pretrained=True, num_classes=0)  # No classifier
                model = model.to(self.device)
                model.eval()
                self.cnn_models[model_name] = model
            
            logger.info(f" Loaded {len(self.cnn_models)} CNN models")
            
        except Exception as e:
            logger.error(f" Failed to load CNN models: {e}")
            self.cnn_models = {}
    
    def extract_features_from_dataset(self, csv_path, dataset_name="Dataset"):
        """Extract comprehensive features from dataset (matching training exactly)"""
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
            for idx, (image_path, label) in enumerate(tqdm(zip(image_paths, labels), 
                                                          desc=f"Processing {dataset_name}", 
                                                          total=len(image_paths))):
                start_time = time.time()
                
                try:
                    # Load and preprocess image
                    image = Image.open(image_path).convert('RGB')
                    
                    # Extract CNN features if available
                    cnn_features = []
                    if self.cnn_models:
                        image_tensor = transform(image).unsqueeze(0).to(self.device)
                        
                        for model_name, model in self.cnn_models.items():
                            with torch.no_grad():
                                features = model(image_tensor)
                                features = features.view(features.size(0), -1)  # Flatten
                                cnn_features.append(features.cpu().numpy().flatten())
                    
                    # Extract basic statistical features
                    basic_features = self.extract_basic_features(image)
                    
                    # Combine all features
                    if cnn_features:
                        combined_features = np.concatenate(cnn_features + [basic_features])
                    else:
                        combined_features = basic_features
                    
                    all_features.append(combined_features)
                    valid_labels.append(label)
                    
                    processing_times.append(time.time() - start_time)
                    
                except Exception as e:
                    logger.warning(f"Failed to process {image_path}: {e}")
                    continue
        
        if all_features:
            features_array = np.array(all_features)
            labels_array = np.array(valid_labels)
            
            logger.info(f" Extracted {len(features_array)} samples with {features_array.shape[1]} features")
            logger.info(f" Average processing time: {np.mean(processing_times):.3f}s per image")
            
            return features_array, labels_array
        else:
            logger.error(" No valid features extracted")
            return None, None
    
    def extract_basic_features(self, image):
        """Extract basic statistical features from image (exactly matching training)"""
        import cv2
        
        # Convert PIL to numpy
        img_array = np.array(image)
        
        features = []
        
        # Convert to different color spaces
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Basic statistics for each channel (exactly matching training)
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
        
        # HSV statistics (exactly matching training)
        for channel in [hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]]:
            features.extend([
                np.mean(channel),
                np.std(channel)
            ])
        
        # Edge features (exactly matching training)
        edges = cv2.Canny(gray, 50, 150)
        features.extend([
            np.mean(edges),
            np.std(edges),
            np.sum(edges > 0) / edges.size  # Edge density
        ])
        
        return features
    
    def validate_models(self, features, labels):
        """Validate all trained models on validation set with threshold optimization"""
        logger.info(" Validating models on validation set...")
        
        # Apply feature selection pipeline exactly as in training
        if self.feature_selector is not None:
            features_selected = self.feature_selector.transform(features)
            logger.info(f"Applied feature selection: {features.shape[1]} -> {features_selected.shape[1]} features")
        else:
            features_selected = features
        
        # Apply RFE if available (this reduces from 200 to 100 features)
        if self.rfe_selector is not None:
            features_final = self.rfe_selector.transform(features_selected)
            logger.info(f"Applied RFE: {features_selected.shape[1]} -> {features_final.shape[1]} features")
        else:
            features_final = features_selected
            
        # Scale features using training scaler
        features_scaled = self.scaler.transform(features_final)
        
        results = {}
        optimized_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Validating {model_name.upper()}...")
            
            start_time = time.time()
            
            # Make standard predictions
            y_pred = model.predict(features_scaled)
            
            # Get prediction probabilities if available
            try:
                y_pred_proba = model.predict_proba(features_scaled)
                y_pred_proba_class1 = y_pred_proba[:, 1]
            except:
                y_pred_proba = None
                y_pred_proba_class1 = None
            
            # Calculate standard metrics
            metrics = self.calculate_metrics(labels, y_pred, y_pred_proba_class1)
            metrics['validation_time'] = time.time() - start_time
            metrics['predictions_per_second'] = len(features) / metrics['validation_time']
            
            results[model_name] = metrics
            
            # Optimize threshold if probabilities available
            if y_pred_proba_class1 is not None:
                best_threshold, best_metrics = self.optimize_threshold(labels, y_pred_proba_class1)
                if best_metrics is not None:
                    # Copy timing information to optimized metrics
                    best_metrics['validation_time'] = metrics['validation_time']
                    best_metrics['predictions_per_second'] = metrics['predictions_per_second']
                    best_metrics['optimized_threshold'] = best_threshold
                    
                    optimized_results[model_name] = {
                        'best_threshold': best_threshold,
                        'optimized_metrics': best_metrics,
                        'standard_metrics': metrics
                    }
                    
                    logger.info(f" {model_name.upper()}: Standard Acc={metrics['accuracy']:.4f} | "
                               f"Optimized Acc={best_metrics['accuracy']:.4f} (threshold={best_threshold:.3f})")
                else:
                    logger.info(f" {model_name.upper()}: Acc={metrics['accuracy']:.4f} F1={metrics['f1_score']:.4f} "
                               f"Prec={metrics['precision']:.4f} Rec={metrics['recall']:.4f}")
            else:
                logger.info(f" {model_name.upper()}: Acc={metrics['accuracy']:.4f} F1={metrics['f1_score']:.4f} "
                           f"Prec={metrics['precision']:.4f} Rec={metrics['recall']:.4f}")
        
        # Use optimized results if available
        if optimized_results:
            logger.info("\n Using threshold-optimized results for final evaluation")
            final_results = {}
            for model_name, opt_result in optimized_results.items():
                final_results[model_name] = opt_result['optimized_metrics']
        else:
            final_results = results
        
        return final_results
    
    def optimize_threshold(self, true_labels, probabilities):
        """Find optimal threshold for binary classification"""
        from sklearn.metrics import f1_score, accuracy_score
        
        try:
            best_f1 = 0
            best_threshold = 0.5
            best_metrics = None
            
            # Test different thresholds
            thresholds = np.arange(0.1, 0.9, 0.02)
            
            for threshold in thresholds:
                y_pred_thresh = (probabilities >= threshold).astype(int)
                
                # Calculate metrics for this threshold
                f1 = f1_score(true_labels, y_pred_thresh, average='binary')
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_metrics = self.calculate_metrics(true_labels, y_pred_thresh, probabilities)
            
            return best_threshold, best_metrics
        except Exception as e:
            logger.warning(f"Threshold optimization failed: {e}")
            return 0.5, None
    
    def calculate_metrics(self, true_labels, predictions, probabilities):
        """Calculate comprehensive validation metrics"""
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, average='binary'),
            'recall': recall_score(true_labels, predictions, average='binary'),
            'f1_score': f1_score(true_labels, predictions, average='binary'),
            'confusion_matrix': confusion_matrix(true_labels, predictions).tolist()
        }
        
        # ROC AUC if probabilities available
        if probabilities is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(true_labels, probabilities)
            except:
                metrics['roc_auc'] = None
        else:
            metrics['roc_auc'] = None
        
        return metrics
    
    def save_validation_results(self, results, features_shape):
        """Save validation results"""
        logger.info(" Saving validation results...")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_metrics = results[best_model_name]
        
        # Prepare summary
        validation_summary = {
            'validation_time': datetime.now().isoformat(),
            'dataset_info': {
                'active_dataset': ACTIVE_DATASET,
                'dataset_name': CURRENT_DATASET['name'],
                'dataset_description': CURRENT_DATASET['description'],
                'validation_samples': features_shape[0],
                'feature_count': features_shape[1],
                'validation_csv': VAL_CSV
            },
            'best_model': best_model_name,
            'best_metrics': best_metrics,
            'all_results': results,
            'gpu_info': {
                'gpu_available': self.gpu_available,
                'gpu_name': self.gpu_name if self.gpu_available else None,
                'device': str(self.device)
            }
        }
        
        # Save detailed results in dataset-specific directory
        validation_results_path = os.path.join(RESULTS_DIR, 'validation_results.json')
        with open(validation_results_path, 'w') as f:
            json.dump(validation_summary, f, indent=2, default=str)
        
        # Save model comparison CSV
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name.upper(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'ROC AUC': f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "N/A",
                'Val Time (s)': f"{metrics['validation_time']:.2f}",
                'Pred/sec': f"{metrics['predictions_per_second']:.1f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        validation_comparison_path = os.path.join(RESULTS_DIR, 'validation_model_comparison.csv')
        comparison_df.to_csv(validation_comparison_path, index=False)
        
        logger.info(f" Validation results saved:")
        logger.info(f"   - Detailed results: {validation_results_path}")
        logger.info(f"   - Model comparison: {validation_comparison_path}")
        
        return best_model_name, best_metrics

def main():
    """Main validation function"""
    
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
    print(" IMAGE FORGERY DETECTION - VALIDATION SET EVALUATION")
    print("=" * 80)
    print(f"📊 Dataset: {dataset_to_use.upper()}")
    print("=" * 80)
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initialize validator
    validator = ForgeryValidator()
    
    # Load CNN models for feature extraction
    validator.load_cnn_models()
    
    # Load validation dataset
    logger.info(" Loading validation dataset...")
    if not os.path.exists(VAL_CSV):
        logger.error(f" Validation dataset CSV not found. Please ensure {VAL_CSV} exists.")
        return
    
    # Extract features from validation dataset
    features, labels = validator.extract_features_from_dataset(VAL_CSV, "Validation Dataset")
    
    if features is None:
        logger.error(" Failed to extract features from validation dataset")
        return
    
    # Validate models
    results = validator.validate_models(features, labels)
    
    # Save validation results
    best_model_name, best_metrics = validator.save_validation_results(results, features.shape)
    
    # Print final summary
    print("\n" + "=" * 80)
    print(" VALIDATION SET EVALUATION FINISHED!")
    print("=" * 80)
    print(f" Device: {validator.device}")
    print(f" GPU Used: {' Yes (' + validator.gpu_name + ')' if validator.gpu_available else ' No'}")
    print(f" Validation Dataset Size: {features.shape[0]} samples")
    print(f" Features: {features.shape[1]}")
    print(f" Best Model: {best_model_name.upper()}")
    print(f" Validation Accuracy: {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
    print(f" Validation F1-Score: {best_metrics['f1_score']:.4f}")
    print(f" Validation Precision: {best_metrics['precision']:.4f}")
    print(f" Validation Recall: {best_metrics['recall']:.4f}")
    if best_metrics['roc_auc']:
        print(f" Validation ROC AUC: {best_metrics['roc_auc']:.4f}")
    print(f" Prediction Speed: {best_metrics['predictions_per_second']:.1f} predictions/sec")
    print(f" Results saved to: {RESULTS_DIR}")
    print("=" * 80)
    
    return best_metrics['accuracy']

if __name__ == "__main__":
    accuracy = main()
