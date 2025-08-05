#!/usr/bin/env python3
"""
 Complete Dataset Test and Analysis
GPU-accelerated comprehensive evaluation with detailed metrics and analysis
"""

import os
import sys
import time
import json
import pickle
import logging
import warnings
import argparse
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_dataset_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import configuration
from core.config import *

class ComprehensiveDatasetTester:
    """Complete dataset tester with GPU acceleration and detailed analysis"""
    
    def __init__(self):
        self.device = DEVICE
        self.gpu_available = GPU_AVAILABLE
        self.gpu_name = GPU_NAME
        
        # Model components
        self.model = None
        self.scaler = None
        self.loaded_models = {}
        self.config = None
        self.all_models = {}
        
        # Results storage
        self.test_results = []
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
        self.feature_extraction_times = []
        
        logger.info(f"Device: {self.device}")
        if self.gpu_available:
            logger.info(f"GPU: {self.gpu_name}")
            # Optimize GPU memory
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        
    def load_trained_models(self):
        """Load all trained models and components"""
        try:
            # Load best model
            model_path = './models/optimized_best_model.pkl'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"[OK] Loaded best model from {model_path}")
            else:
                logger.error(f"[ERROR] Best model not found at {model_path}")
                return False
            
            # Load scaler
            scaler_path = './models/optimized_scaler.pkl'
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"[OK] Loaded scaler from {scaler_path}")
            else:
                logger.error(f"[ERROR] Scaler not found at {scaler_path}")
                return False
            
            # Load all models for ensemble analysis
            all_models_path = './models/optimized_all_models.pkl'
            if os.path.exists(all_models_path):
                with open(all_models_path, 'rb') as f:
                    self.all_models = pickle.load(f)
                logger.info(f"[OK] Loaded all models for ensemble analysis")
            
            # Load configuration
            config_path = './models/optimized_config.json'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"[OK] Loaded configuration")
            
            # Load CNN models for feature extraction
            self.load_cnn_models()
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error loading models: {e}")
            return False
    
    def load_cnn_models(self):
        """Load pre-trained CNN models for feature extraction"""
        try:
            import torchvision.models as models
            
            self.loaded_models = {}
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
            logger.warning(f"[WARNING] Error loading CNN models: {e}")
            return False
    
    def load_complete_dataset(self):
        """Load and prepare the complete dataset"""
        logger.info(" Loading complete dataset...")
        
        # Check if CSV files exist
        csv_files = [TRAIN_CSV, VAL_CSV, TEST_CSV]
        datasets = []
        
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                datasets.append(df)
                logger.info(f"[OK] Loaded {len(df)} samples from {os.path.basename(csv_file)}")
            else:
                logger.warning(f"[WARNING] {csv_file} not found, creating from directories...")
                # Create dataset from directories if CSV doesn't exist
                self.create_dataset_from_directories()
                break
        
        if datasets:
            # Combine all datasets
            complete_dataset = pd.concat(datasets, ignore_index=True)
        else:
            # Load from directories
            complete_dataset = self.create_dataset_from_directories()
        
        logger.info(f" Complete dataset: {len(complete_dataset)} samples")
        
        # Verify file paths exist
        valid_samples = []
        for idx, row in complete_dataset.iterrows():
            if os.path.exists(row['filepath']):
                valid_samples.append(row)
            else:
                logger.warning(f"[WARNING] File not found: {row['filepath']}")
        
        if valid_samples:
            complete_dataset = pd.DataFrame(valid_samples)
            logger.info(f"[OK] Valid samples: {len(complete_dataset)}")
        
        return complete_dataset
    
    def create_dataset_from_directories(self):
        """Create dataset from directory structure if CSV files don't exist"""
        logger.info("📁 Creating dataset from directories...")
        
        data = []
        
        # Process authentic images
        if os.path.exists(AUTHENTIC_DIR):
            auth_files = [f for f in os.listdir(AUTHENTIC_DIR) if f.endswith('.tif')]
            for filename in auth_files:
                filepath = os.path.join(AUTHENTIC_DIR, filename)
                data.append({'filepath': filepath, 'label': 0, 'filename': filename})
            logger.info(f"[OK] Found {len(auth_files)} authentic images")
        
        # Process forged images  
        if os.path.exists(FORGED_DIR):
            forged_files = [f for f in os.listdir(FORGED_DIR) if f.endswith('.tif')]
            for filename in forged_files:
                filepath = os.path.join(FORGED_DIR, filename)
                data.append({'filepath': filepath, 'label': 1, 'filename': filename})
            logger.info(f"[OK] Found {len(forged_files)} forged images")
        
        return pd.DataFrame(data)
    
    def extract_features_batch(self, image_paths, batch_size=8):
        """Extract features from multiple images in batches for GPU efficiency"""
        all_features = []
        
        # Transform for CNN models
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f" Extracting features from {len(image_paths)} images...")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
                batch_paths = image_paths[i:i+batch_size]
                batch_features = []
                
                for image_path in batch_paths:
                    start_time = time.time()
                    
                    try:
                        # Load and preprocess image
                        image = Image.open(image_path).convert('RGB')
                        image_tensor = transform(image).unsqueeze(0).to(self.device)
                        
                        # Extract CNN features if available
                        cnn_features = []
                        if self.loaded_models:
                            for model_name, model in self.loaded_models.items():
                                try:
                                    features = model(image_tensor)
                                    features = features.view(features.size(0), -1)
                                    cnn_features.append(features.cpu().numpy().flatten())
                                except Exception as e:
                                    logger.warning(f"[WARNING] Error with {model_name}: {e}")
                        
                        # Extract basic features
                        basic_features = self.extract_basic_features(image)
                        
                        # Combine all features
                        combined_features = []
                        if cnn_features:
                            combined_features.extend(np.concatenate(cnn_features))
                        combined_features.extend(basic_features)
                        
                        batch_features.append(combined_features)
                        
                        # Track extraction time
                        extraction_time = time.time() - start_time
                        self.feature_extraction_times.append(extraction_time)
                        
                    except Exception as e:
                        logger.error(f"[ERROR] Error processing {image_path}: {e}")
                        # Add zero features for failed images
                        if all_features:
                            feature_dim = len(all_features[0])
                        else:
                            feature_dim = 4517  # Use config feature count
                        batch_features.append([0.0] * feature_dim)
                
                all_features.extend(batch_features)
                
                # Clear GPU cache periodically
                if self.gpu_available and (i + batch_size) % (batch_size * 4) == 0:
                    torch.cuda.empty_cache()
        
        if all_features:
            return np.array(all_features)
        else:
            logger.error("[ERROR] No features extracted")
            return None
    
    def extract_basic_features(self, image):
        """Extract basic statistical features from image - matching training format"""
        # Convert PIL to numpy
        img_array = np.array(image)
        
        features = []
        
        # Convert to different color spaces
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Basic statistics for each channel (match training exactly)
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
        
        # HSV statistics (match training exactly)
        for channel in [hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]]:
            features.extend([
                np.mean(channel),
                np.std(channel)
            ])
        
        # Edge features (match training exactly)
        edges = cv2.Canny(gray, 50, 150)
        features.extend([
            np.mean(edges),
            np.std(edges),
            np.sum(edges > 0) / edges.size  # Edge density
        ])
        
        return features
    
    def test_complete_dataset(self, dataset):
        """Test the model on the complete dataset"""
        logger.info(" Testing model on complete dataset...")
        
        image_paths = dataset['filepath'].values
        true_labels = dataset['label'].values
        
        # Extract features in batches for efficiency
        features = self.extract_features_batch(image_paths, batch_size=16 if self.gpu_available else 8)
        
        if features is None:
            logger.error("[ERROR] Feature extraction failed")
            return None
        
        # Scale features
        if self.scaler:
            features_scaled = self.scaler.transform(features)
        else:
            logger.warning("[WARNING] No scaler available, using raw features")
            features_scaled = features
        
        # Make predictions
        logger.info(" Making predictions...")
        start_time = time.time()
        
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        prediction_time = time.time() - start_time
        
        # Store results
        self.true_labels = true_labels
        self.predictions = predictions
        self.probabilities = probabilities
        
        # Create detailed results
        for i, (path, true_label, pred_label, prob) in enumerate(zip(
            image_paths, true_labels, predictions, probabilities
        )):
            self.test_results.append({
                'filepath': path,
                'filename': os.path.basename(path),
                'true_label': int(true_label),
                'predicted_label': int(pred_label),
                'authentic_prob': float(prob[0]),
                'forged_prob': float(prob[1]),
                'correct': int(true_label == pred_label),
                'feature_extraction_time': self.feature_extraction_times[i] if i < len(self.feature_extraction_times) else 0.0
            })
        
        logger.info(f"[OK] Predictions completed in {prediction_time:.2f} seconds")
        
        return self.calculate_comprehensive_metrics()
    
    def calculate_comprehensive_metrics(self):
        """Calculate comprehensive performance metrics"""
        logger.info(" Calculating comprehensive metrics...")
        
        # Basic metrics
        accuracy = accuracy_score(self.true_labels, self.predictions)
        precision = precision_score(self.true_labels, self.predictions, average='weighted')
        recall = recall_score(self.true_labels, self.predictions, average='weighted')
        f1 = f1_score(self.true_labels, self.predictions, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(self.true_labels, self.predictions, average=None)
        recall_per_class = recall_score(self.true_labels, self.predictions, average=None)
        f1_per_class = f1_score(self.true_labels, self.predictions, average=None)
        
        # ROC AUC and other metrics
        roc_auc = roc_auc_score(self.true_labels, self.probabilities[:, 1])
        avg_precision = average_precision_score(self.true_labels, self.probabilities[:, 1])
        
        # Confusion matrix
        cm = confusion_matrix(self.true_labels, self.predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Performance timing
        avg_feature_time = np.mean(self.feature_extraction_times) if self.feature_extraction_times else 0
        total_samples = len(self.test_results)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'positive_predictive_value': ppv,
            'negative_predictive_value': npv,
            'per_class_precision': precision_per_class.tolist(),
            'per_class_recall': recall_per_class.tolist(),
            'per_class_f1': f1_per_class.tolist(),
            'total_samples': total_samples,
            'avg_feature_extraction_time': avg_feature_time,
            'gpu_used': self.gpu_available,
            'gpu_name': self.gpu_name if self.gpu_available else 'None'
        }
        
        return metrics
    
    def display_results(self, metrics):
        """Display comprehensive test results"""
        print("\n" + "="*80)
        print(" COMPLETE DATASET TEST RESULTS")
        print("="*80)
        
        print(f" Device: {self.device}")
        if self.gpu_available:
            print(f" GPU: {self.gpu_name}")
        
        print(f" Total Samples: {metrics['total_samples']}")
        print(f"⏱️ Avg Feature Extraction Time: {metrics['avg_feature_extraction_time']:.4f}s per image")
        
        print("\n🎯 OVERALL PERFORMANCE METRICS")
        print("-" * 50)
        print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"F1 Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"Avg Precision: {metrics['average_precision']:.4f}")
        
        print("\n📈 PER-CLASS METRICS")
        print("-" * 50)
        class_names = ['Authentic (0)', 'Forged (1)']
        for i, class_name in enumerate(class_names):
            print(f"{class_name}:")
            print(f"  Precision: {metrics['per_class_precision'][i]:.4f} ({metrics['per_class_precision'][i]*100:.2f}%)")
            print(f"  Recall:    {metrics['per_class_recall'][i]:.4f} ({metrics['per_class_recall'][i]*100:.2f}%)")
            print(f"  F1 Score:  {metrics['per_class_f1'][i]:.4f} ({metrics['per_class_f1'][i]*100:.2f}%)")
        
        print("\n🔲 CONFUSION MATRIX")
        print("-" * 50)
        cm = metrics['confusion_matrix']
        print(f"              Predicted")
        print(f"              Authentic  Forged")
        print(f"True Authentic    {cm[0][0]:3d}      {cm[0][1]:3d}")
        print(f"     Forged       {cm[1][0]:3d}      {cm[1][1]:3d}")
        
        print("\n DETAILED ANALYSIS")
        print("-" * 50)
        print(f"True Negatives (TN):  {metrics['true_negatives']:3d} (Correctly identified authentic)")
        print(f"False Positives (FP): {metrics['false_positives']:3d} (Authentic wrongly classified as forged)")
        print(f"False Negatives (FN): {metrics['false_negatives']:3d} (Forged wrongly classified as authentic)")
        print(f"True Positives (TP):  {metrics['true_positives']:3d} (Correctly identified forged)")
        
        print("\n🧮 ADDITIONAL METRICS")
        print("-" * 50)
        print(f"Sensitivity (TPR): {metrics['sensitivity']:.4f} ({metrics['sensitivity']*100:.2f}%) - Ability to detect forged images")
        print(f"Specificity (TNR): {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%) - Ability to detect authentic images")
        print(f"Positive Pred Val: {metrics['positive_predictive_value']:.4f} ({metrics['positive_predictive_value']*100:.2f}%) - When predicting forged, how often correct")
        print(f"Negative Pred Val: {metrics['negative_predictive_value']:.4f} ({metrics['negative_predictive_value']*100:.2f}%) - When predicting authentic, how often correct")
        
        # Performance assessment
        print(f"\n{'🎉' if metrics['accuracy'] >= 0.85 else '[WARNING]'} PERFORMANCE ASSESSMENT")
        print("-" * 50)
        if metrics['accuracy'] >= 0.85:
            print(f"[OK] EXCELLENT: Accuracy {metrics['accuracy']*100:.2f}% meets the 85% threshold!")
            print("🎯 Model performance is satisfactory for production use.")
        else:
            print(f"[ERROR] NEEDS IMPROVEMENT: Accuracy {metrics['accuracy']*100:.2f}% is below 85% threshold.")
            print(" Model requires optimization and improvements.")
    
    def plot_visualizations(self, metrics):
        """Create comprehensive visualizations"""
        logger.info(" Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(2, 3, 1)
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Authentic', 'Forged'],
                   yticklabels=['Authentic', 'Forged'],
                   ax=ax1)
        ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. ROC Curve
        ax2 = plt.subplot(2, 3, 2)
        fpr, tpr, _ = roc_curve(self.true_labels, self.probabilities[:, 1])
        ax2.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        ax3 = plt.subplot(2, 3, 3)
        precision_curve, recall_curve, _ = precision_recall_curve(self.true_labels, self.probabilities[:, 1])
        ax3.plot(recall_curve, precision_curve, linewidth=2, 
                label=f'PR Curve (AP = {metrics["average_precision"]:.3f})')
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Metrics Comparison
        ax4 = plt.subplot(2, 3, 4)
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        metrics_values = [
            metrics['accuracy'], metrics['precision'], 
            metrics['recall'], metrics['f1_score'], metrics['roc_auc']
        ]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars = ax4.bar(metrics_names, metrics_values, color=colors, alpha=0.8)
        ax4.set_ylim([0, 1])
        ax4.set_title('Performance Metrics Overview', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.axhline(y=0.85, color='red', linestyle='--', alpha=0.7, label='85% Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Class Distribution
        ax5 = plt.subplot(2, 3, 5)
        class_counts = [np.sum(self.true_labels == 0), np.sum(self.true_labels == 1)]
        class_names = ['Authentic', 'Forged']
        colors = ['#2ca02c', '#d62728']
        wedges, texts, autotexts = ax5.pie(class_counts, labels=class_names, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax5.set_title('Dataset Class Distribution', fontsize=14, fontweight='bold')
        
        # 6. Performance Summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Create performance summary text
        summary_text = f"""
PERFORMANCE SUMMARY
{'='*30}

Total Samples: {metrics['total_samples']}
GPU Used: {'[OK] Yes' if metrics['gpu_used'] else '[ERROR] No'}
GPU: {metrics['gpu_name']}

ACCURACY: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
{'[OK] ABOVE 85% THRESHOLD' if metrics['accuracy'] >= 0.85 else '[ERROR] BELOW 85% THRESHOLD'}

Key Metrics:
• Precision: {metrics['precision']:.4f}
• Recall: {metrics['recall']:.4f}  
• F1 Score: {metrics['f1_score']:.4f}
• ROC AUC: {metrics['roc_auc']:.4f}

Classification Details:
• True Positives: {metrics['true_positives']}
• True Negatives: {metrics['true_negatives']}
• False Positives: {metrics['false_positives']}
• False Negatives: {metrics['false_negatives']}

Avg Feature Time: {metrics['avg_feature_extraction_time']:.4f}s
"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('./models/complete_dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("[OK] Visualizations saved to ./models/complete_dataset_analysis.png")
    
    def save_results(self, metrics):
        """Save detailed results to files"""
        logger.info("💾 Saving results...")
        
        # Save test results CSV
        results_df = pd.DataFrame(self.test_results)
        results_csv_path = './models/complete_dataset_test_results.csv'
        results_df.to_csv(results_csv_path, index=False)
        logger.info(f"[OK] Test results saved to {results_csv_path}")
        
        # Save metrics JSON
        metrics_json_path = './models/complete_dataset_metrics.json'
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"[OK] Metrics saved to {metrics_json_path}")
        
        # Save detailed classification report
        report = classification_report(
            self.true_labels, self.predictions,
            target_names=['Authentic', 'Forged'],
            output_dict=True
        )
        
        report_json_path = './models/complete_dataset_classification_report.json'
        with open(report_json_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"[OK] Classification report saved to {report_json_path}")
        
        return results_csv_path, metrics_json_path, report_json_path

def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description='Complete Dataset Test and Analysis')
    parser.add_argument('--visualize', '-v', action='store_true', help='Show visualizations')
    parser.add_argument('--save-plots', '-s', action='store_true', help='Save plot images')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("="*80)
    print(" COMPLETE DATASET TEST AND ANALYSIS")
    print("="*80)
    
    # Initialize tester
    tester = ComprehensiveDatasetTester()
    
    # Load trained models
    logger.info(" Loading trained models...")
    if not tester.load_trained_models():
        print("[ERROR] Failed to load models. Please train first with train_optimized.py")
        return False
    
    # Load complete dataset
    logger.info(" Loading complete dataset...")
    dataset = tester.load_complete_dataset()
    if dataset is None or len(dataset) == 0:
        print("[ERROR] Failed to load dataset")
        return False
    
    # Test the model
    logger.info(" Testing model on complete dataset...")
    start_time = time.time()
    metrics = tester.test_complete_dataset(dataset)
    total_time = time.time() - start_time
    
    if metrics is None:
        print("[ERROR] Testing failed")
        return False
    
    # Display results
    tester.display_results(metrics)
    
    # Save results
    tester.save_results(metrics)
    
    # Create visualizations
    if args.visualize or args.save_plots:
        tester.plot_visualizations(metrics)
    
    # Final summary
    print(f"\n⏱️ Total testing time: {total_time:.2f} seconds")
    print(f"💾 Results saved to ./models/")
    
    # Return success status
    return metrics['accuracy'] >= 0.85

if __name__ == "__main__":
    success = main()
    print("\n" + "="*80)
    if success:
        print("🎉 TEST COMPLETED SUCCESSFULLY - Model meets 85% accuracy threshold!")
    else:
        print("[WARNING] TEST COMPLETED - Model needs improvement to reach 85% accuracy threshold.")
    print("="*80)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
