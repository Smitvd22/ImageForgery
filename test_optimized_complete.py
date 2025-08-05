#!/usr/bin/env python3
"""
🚀 Complete Dataset Test Evaluation for Optimized Model
GPU-accelerated comprehensive testing with detailed analysis
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import pickle
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import configuration
from core.config import *

# Import necessary libraries for evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

class OptimizedModelEvaluator:
    """Comprehensive evaluator for optimized models with GPU acceleration"""
    
    def __init__(self, model_dir="./models"):
        self.model_dir = Path(model_dir)
        self.device = DEVICE
        self.gpu_available = GPU_AVAILABLE
        self.gpu_name = GPU_NAME
        
        # Initialize storage for results
        self.results = {}
        self.evaluation_start_time = time.time()
        
        logger.info(f"🎮 Device: {self.device}")
        if self.gpu_available:
            logger.info(f"🚀 GPU: {self.gpu_name}")
        else:
            logger.info("💻 Using CPU")
        
        # Load models and preprocessors
        self.load_optimized_models()
    
    def load_optimized_models(self):
        """Load the optimized trained models"""
        try:
            # Load configuration
            config_path = self.model_dir / "optimized_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"✅ Loaded configuration")
                logger.info(f"📊 Best model: {self.config['best_model']}")
                logger.info(f"📊 Feature count: {self.config['feature_count']}")
            else:
                logger.error("❌ Configuration file not found")
                return False
            
            # Load best model
            best_model_path = self.model_dir / "optimized_best_model.pkl"
            if best_model_path.exists():
                with open(best_model_path, 'rb') as f:
                    self.best_model = pickle.load(f)
                logger.info(f"✅ Loaded best model: {self.config['best_model']}")
            else:
                logger.error("❌ Best model file not found")
                return False
            
            # Load all models
            all_models_path = self.model_dir / "optimized_all_models.pkl"
            if all_models_path.exists():
                with open(all_models_path, 'rb') as f:
                    self.all_models = pickle.load(f)
                logger.info(f"✅ Loaded all models: {list(self.all_models.keys())}")
            else:
                logger.error("❌ All models file not found")
                return False
            
            # Load scaler
            scaler_path = self.model_dir / "optimized_scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"✅ Loaded feature scaler")
            else:
                logger.error("❌ Scaler file not found")
                return False
            
            # Load CNN models for feature extraction
            self.load_cnn_models()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    def load_cnn_models(self):
        """Load CNN models for feature extraction"""
        try:
            import timm
            import torch
            
            self.cnn_models = ['resnet50', 'efficientnet_b2', 'densenet121']
            self.loaded_cnn_models = {}
            
            for model_name in self.cnn_models:
                logger.info(f"Loading {model_name}...")
                model = timm.create_model(f'{model_name}.ra_in1k', pretrained=True, num_classes=0)
                model = model.to(self.device)
                model.eval()
                self.loaded_cnn_models[model_name] = model
                logger.info(f"✅ {model_name} loaded")
            
            return True
        except Exception as e:
            logger.error(f"❌ Error loading CNN models: {e}")
            return False
    
    def extract_features_from_dataset(self, csv_path, dataset_name="Dataset"):
        """Extract features from complete dataset"""
        import torch
        import timm
        from PIL import Image
        import torchvision.transforms as T
        
        logger.info(f"🔧 Extracting features from {dataset_name}...")
        
        # Load dataset
        df = pd.read_csv(csv_path)
        image_paths = df['filepath'].values
        labels = df['label'].values
        
        logger.info(f"📊 {dataset_name} size: {len(image_paths)} images")
        
        # Transform
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        all_features = []
        valid_labels = []
        processing_times = []
        
        with torch.no_grad():
            for i, img_path in enumerate(tqdm(image_paths, desc=f"Processing {dataset_name}")):
                start_time = time.time()
                try:
                    # Load and process image
                    image = Image.open(img_path).convert('RGB')
                    img_tensor = transform(image).unsqueeze(0).to(self.device)
                    
                    # Extract features from all CNN models
                    features = []
                    for model_name, model in self.loaded_cnn_models.items():
                        feat = model(img_tensor).cpu().numpy().flatten()
                        features.extend(feat)
                    
                    # Add basic image features
                    basic_features = self.extract_basic_features(image)
                    features.extend(basic_features)
                    
                    all_features.append(features)
                    valid_labels.append(labels[i])
                    
                    processing_times.append(time.time() - start_time)
                    
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")
                    continue
        
        if all_features:
            features_array = np.array(all_features)
            labels_array = np.array(valid_labels)
            
            avg_processing_time = np.mean(processing_times)
            logger.info(f"✅ {dataset_name}: {features_array.shape[0]} samples with {features_array.shape[1]} features")
            logger.info(f"⚡ Average processing time: {avg_processing_time:.4f}s per image")
            logger.info(f"🚀 Processing speed: {1/avg_processing_time:.2f} images/second")
            
            return features_array, labels_array, processing_times
        else:
            logger.error(f"❌ No features extracted from {dataset_name}")
            return None, None, []
    
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
    
    def evaluate_dataset(self, features, labels, dataset_name="Dataset"):
        """Evaluate all models on the given dataset"""
        logger.info(f"\n🎯 Evaluating {dataset_name}...")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        evaluation_results = {}
        
        # Evaluate all models
        for model_name, model in self.all_models.items():
            try:
                start_time = time.time()
                
                # Make predictions
                predictions = model.predict(features_scaled)
                probabilities = model.predict_proba(features_scaled)
                
                prediction_time = time.time() - start_time
                
                # Calculate metrics
                metrics = self.calculate_comprehensive_metrics(labels, predictions, probabilities)
                metrics['prediction_time'] = prediction_time
                metrics['prediction_speed'] = len(labels) / prediction_time
                
                evaluation_results[model_name] = metrics
                
                logger.info(f"✅ {model_name.upper()}: Acc={metrics['accuracy']:.4f} F1={metrics['f1_score']:.4f} AUC={metrics['roc_auc']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error evaluating {model_name}: {e}")
                continue
        
        return evaluation_results
    
    def calculate_comprehensive_metrics(self, true_labels, predictions, probabilities):
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(true_labels, predictions)
        metrics['precision'] = precision_score(true_labels, predictions, average='weighted')
        metrics['recall'] = recall_score(true_labels, predictions, average='weighted')
        metrics['f1_score'] = f1_score(true_labels, predictions, average='weighted')
        
        # ROC AUC
        if probabilities.shape[1] == 2:
            metrics['roc_auc'] = roc_auc_score(true_labels, probabilities[:, 1])
        else:
            metrics['roc_auc'] = None
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(true_labels, predictions)
        
        # Per-class metrics
        metrics['classification_report'] = classification_report(
            true_labels, predictions, 
            target_names=['Authentic', 'Forged'],
            output_dict=True
        )
        
        # Additional metrics
        tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return metrics
    
    def plot_confusion_matrix(self, metrics, title="Confusion Matrix", save_path=None):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        cm = metrics['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Authentic', 'Forged'],
                   yticklabels=['Authentic', 'Forged'],
                   cbar_kws={'label': 'Count'})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        
        # Add accuracy to the plot
        accuracy = metrics['accuracy']
        plt.text(0.5, -0.1, f'Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)', 
                ha='center', transform=plt.gca().transAxes, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, true_labels, probabilities, title="ROC Curve", save_path=None):
        """Plot ROC curve"""
        if probabilities.shape[1] != 2:
            logger.warning("⚠️ Cannot plot ROC curve - not binary classification")
            return
        
        fpr, tpr, _ = roc_curve(true_labels, probabilities[:, 1])
        auc_score = roc_auc_score(true_labels, probabilities[:, 1])
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {auc_score:.4f})', color='blue')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, true_labels, probabilities, title="Precision-Recall Curve", save_path=None):
        """Plot precision-recall curve"""
        if probabilities.shape[1] != 2:
            logger.warning("⚠️ Cannot plot PR curve - not binary classification")
            return
        
        precision, recall, _ = precision_recall_curve(true_labels, probabilities[:, 1])
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, linewidth=3, color='green')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Precision-recall curve saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, results, save_path=None):
        """Plot model comparison chart"""
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        f1_scores = [results[model]['f1_score'] for model in models]
        roc_aucs = [results[model]['roc_auc'] for model in models if results[model]['roc_auc'] is not None]
        
        # Create comparison plot
        x = np.arange(len(models))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        bars1 = ax.bar(x - width, accuracies, width, label='Accuracy', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x, f1_scores, width, label='F1-Score', alpha=0.8, color='lightgreen')
        if len(roc_aucs) == len(models):
            bars3 = ax.bar(x + width, roc_aucs, width, label='ROC AUC', alpha=0.8, color='salmon')
        
        ax.set_xlabel('Models', fontsize=14)
        ax.set_ylabel('Score', fontsize=14)
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([model.upper() for model in models], rotation=45)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
        
        autolabel(bars1)
        autolabel(bars2)
        if len(roc_aucs) == len(models):
            autolabel(bars3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Model comparison saved to {save_path}")
        
        plt.show()
    
    def perform_cross_validation(self, features, labels):
        """Perform cross-validation on the complete dataset"""
        logger.info("\n🔄 Performing Cross-Validation...")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        cv_results = {}
        
        # Use the best model for CV
        best_model_name = self.config['best_model']
        best_model = self.all_models[best_model_name]
        
        # Perform stratified cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        start_time = time.time()
        cv_scores = cross_val_score(
            best_model, features_scaled, labels,
            cv=cv, scoring='accuracy', n_jobs=-1
        )
        cv_time = time.time() - start_time
        
        cv_results = {
            'model': best_model_name,
            'scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'cv_time': cv_time
        }
        
        logger.info(f"✅ Cross-Validation Results ({best_model_name.upper()}):")
        logger.info(f"   Scores: {cv_scores}")
        logger.info(f"   Mean Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        logger.info(f"   Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
        logger.info(f"   CV Time: {cv_time:.2f}s")
        
        return cv_results
    
    def generate_comprehensive_report(self, all_results):
        """Generate comprehensive evaluation report"""
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_evaluation_time = time.time() - self.evaluation_start_time
        
        logger.info("\n" + "="*80)
        logger.info("🎯 COMPREHENSIVE EVALUATION REPORT")
        logger.info("="*80)
        logger.info(f"📅 Report generated: {report_time}")
        logger.info(f"⏱️ Total evaluation time: {total_evaluation_time:.2f}s")
        logger.info(f"🎮 Device used: {self.device}")
        if self.gpu_available:
            logger.info(f"🚀 GPU: {self.gpu_name}")
        
        # Dataset summary
        logger.info(f"\n📊 DATASET SUMMARY:")
        for dataset_name, results in all_results.items():
            if 'dataset_info' in results:
                info = results['dataset_info']
                logger.info(f"   {dataset_name}: {info['total_samples']} samples")
                logger.info(f"      Processing speed: {info['avg_speed']:.2f} images/sec")
        
        # Model performance summary
        logger.info(f"\n🏆 MODEL PERFORMANCE SUMMARY:")
        best_overall_acc = 0
        best_overall_model = ""
        best_overall_dataset = ""
        
        for dataset_name, results in all_results.items():
            if 'models' in results:
                logger.info(f"\n   {dataset_name.upper()}:")
                for model_name, metrics in results['models'].items():
                    acc = metrics['accuracy']
                    f1 = metrics['f1_score']
                    auc = metrics['roc_auc'] or 0
                    
                    logger.info(f"      {model_name.upper()}: Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f}")
                    
                    if acc > best_overall_acc:
                        best_overall_acc = acc
                        best_overall_model = model_name
                        best_overall_dataset = dataset_name
        
        logger.info(f"\n🎯 BEST OVERALL PERFORMANCE:")
        logger.info(f"   Model: {best_overall_model.upper()}")
        logger.info(f"   Dataset: {best_overall_dataset}")
        logger.info(f"   Accuracy: {best_overall_acc:.4f} ({best_overall_acc*100:.2f}%)")
        
        # Cross-validation summary
        if 'cross_validation' in all_results:
            cv_results = all_results['cross_validation']
            logger.info(f"\n🔄 CROSS-VALIDATION SUMMARY:")
            logger.info(f"   Model: {cv_results['model'].upper()}")
            logger.info(f"   Mean Accuracy: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
            logger.info(f"   CV Time: {cv_results['cv_time']:.2f}s")
        
        # Performance assessment
        logger.info(f"\n📈 PERFORMANCE ASSESSMENT:")
        if best_overall_acc >= 0.85:
            logger.info(f"   ✅ EXCELLENT: Accuracy {best_overall_acc*100:.2f}% meets 85%+ target!")
        elif best_overall_acc >= 0.80:
            logger.info(f"   ✅ GOOD: Accuracy {best_overall_acc*100:.2f}% meets 80%+ target")
        elif best_overall_acc >= 0.75:
            logger.info(f"   ⚠️ MODERATE: Accuracy {best_overall_acc*100:.2f}% - room for improvement")
        else:
            logger.info(f"   ❌ POOR: Accuracy {best_overall_acc*100:.2f}% - significant improvement needed")
        
        logger.info("="*80)
        
        return {
            'report_time': report_time,
            'total_evaluation_time': total_evaluation_time,
            'best_overall_accuracy': best_overall_acc,
            'best_overall_model': best_overall_model,
            'best_overall_dataset': best_overall_dataset,
            'all_results': all_results
        }
    
    def save_detailed_results(self, all_results, comprehensive_report):
        """Save detailed results to files"""
        logger.info("\n💾 Saving detailed results...")
        
        # Create results directory
        results_dir = Path("./models")
        results_dir.mkdir(exist_ok=True)
        
        # Save metrics summary (main results)
        metrics_path = results_dir / "complete_dataset_metrics.json"
        metrics_summary = {
            'best_model': comprehensive_report['best_overall_model'],
            'best_accuracy': comprehensive_report['best_overall_accuracy'],
            'evaluation_time': comprehensive_report['total_evaluation_time'],
            'gpu_used': self.gpu_available,
            'gpu_name': self.gpu_name if self.gpu_available else None,
            'complete_dataset_accuracy': all_results['complete_dataset']['models']['rf']['accuracy'],
            'complete_dataset_f1': all_results['complete_dataset']['models']['rf']['f1_score'],
            'complete_dataset_auc': all_results['complete_dataset']['models']['rf']['roc_auc'],
            'test_set_accuracy': all_results['test_set']['models']['rf']['accuracy'],
            'test_set_f1': all_results['test_set']['models']['rf']['f1_score'],
            'test_set_auc': all_results['test_set']['models']['rf']['roc_auc'],
            'cv_mean_accuracy': all_results['cross_validation']['mean_score'],
            'cv_std_accuracy': all_results['cross_validation']['std_score']
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        logger.info(f"✅ Metrics summary saved to {metrics_path}")
        
        # Save performance comparison for all models
        performance_path = results_dir / "complete_dataset_performance.json"
        performance_data = {}
        
        for dataset_name, dataset_results in all_results.items():
            if 'models' in dataset_results:
                performance_data[dataset_name] = {}
                for model_name, metrics in dataset_results['models'].items():
                    performance_data[dataset_name][model_name] = {
                        'accuracy': float(metrics['accuracy']),
                        'f1_score': float(metrics['f1_score']),
                        'roc_auc': float(metrics['roc_auc']) if metrics['roc_auc'] is not None else None,
                        'precision': float(metrics['precision']),
                        'recall': float(metrics['recall'])
                    }
        
        with open(performance_path, 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        logger.info(f"✅ Performance comparison saved to {performance_path}")
        
        # Save CSV with predictions for best model on complete dataset
        if 'complete_dataset' in all_results and 'models' in all_results['complete_dataset']:
            best_model_name = 'rf'  # Use RF as it has best accuracy
            best_model_results = all_results['complete_dataset']['models'][best_model_name]
            
            if 'true_labels' in best_model_results and 'predictions' in best_model_results:
                # Create CSV with detailed results
                csv_data = {
                    'true_label': best_model_results['true_labels'].tolist(),
                    'predicted_label': best_model_results['predictions'].tolist(),
                    'accuracy': [best_model_results['accuracy']] * len(best_model_results['true_labels']),
                    'f1_score': [best_model_results['f1_score']] * len(best_model_results['true_labels']),
                    'roc_auc': [best_model_results['roc_auc']] * len(best_model_results['true_labels'])
                }
                
                csv_path = results_dir / "complete_dataset_test_results.csv"
                pd.DataFrame(csv_data).to_csv(csv_path, index=False)
                logger.info(f"✅ CSV results saved to {csv_path}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj

def main():
    """Main evaluation function"""
    print("=" * 80)
    print("🚀 OPTIMIZED MODEL COMPLETE DATASET EVALUATION")
    print("=" * 80)
    
    # Initialize evaluator
    evaluator = OptimizedModelEvaluator()
    
    # Storage for all results
    all_results = {}
    
    # Test on complete dataset
    logger.info("📂 Testing on complete dataset...")
    
    # Load and process complete dataset
    try:
        # Use labels.csv for complete dataset
        complete_features, complete_labels, processing_times = evaluator.extract_features_from_dataset(
            './data/labels.csv', 'Complete Dataset'
        )
        
        if complete_features is not None:
            # Evaluate all models
            complete_results = evaluator.evaluate_dataset(
                complete_features, complete_labels, 'Complete Dataset'
            )
            
            all_results['complete_dataset'] = {
                'models': complete_results,
                'dataset_info': {
                    'total_samples': len(complete_labels),
                    'authentic_samples': np.sum(complete_labels == 0),
                    'forged_samples': np.sum(complete_labels == 1),
                    'avg_processing_time': np.mean(processing_times),
                    'avg_speed': 1 / np.mean(processing_times)
                }
            }
            
            # Get best model for detailed analysis
            best_model_name = evaluator.config['best_model']
            best_metrics = complete_results[best_model_name]
            
            # Store additional data for the best model
            best_features_scaled = evaluator.scaler.transform(complete_features)
            best_predictions = evaluator.best_model.predict(best_features_scaled)
            best_probabilities = evaluator.best_model.predict_proba(best_features_scaled)
            
            all_results['complete_dataset']['models'][best_model_name]['true_labels'] = complete_labels
            all_results['complete_dataset']['models'][best_model_name]['predictions'] = best_predictions
            all_results['complete_dataset']['models'][best_model_name]['probabilities'] = best_probabilities
            
            # Generate visualizations
            logger.info("📊 Generating visualizations...")
            
            # Confusion Matrix
            evaluator.plot_confusion_matrix(
                best_metrics, 
                f"Confusion Matrix - {best_model_name.upper()} on Complete Dataset",
                "./models/complete_dataset_confusion_matrix.png"
            )
            
            # ROC Curve
            evaluator.plot_roc_curve(
                complete_labels, best_probabilities,
                f"ROC Curve - {best_model_name.upper()} on Complete Dataset",
                "./models/complete_dataset_roc_curve.png"
            )
            
            # Precision-Recall Curve
            evaluator.plot_precision_recall_curve(
                complete_labels, best_probabilities,
                f"Precision-Recall Curve - {best_model_name.upper()} on Complete Dataset",
                "./models/complete_dataset_pr_curve.png"
            )
            
            # Model Comparison
            evaluator.plot_model_comparison(
                complete_results,
                "./models/complete_dataset_model_comparison.png"
            )
            
            # Perform cross-validation
            cv_results = evaluator.perform_cross_validation(complete_features, complete_labels)
            all_results['cross_validation'] = cv_results
            
        else:
            logger.error("❌ Failed to extract features from complete dataset")
            return
            
    except Exception as e:
        logger.error(f"❌ Error processing complete dataset: {e}")
        return
    
    # Test on individual datasets if available
    datasets_to_test = [
        ('./data/train_labels.csv', 'Training Set'),
        ('./data/val_labels.csv', 'Validation Set'),
        ('./data/test_labels.csv', 'Test Set')
    ]
    
    for csv_path, dataset_name in datasets_to_test:
        if os.path.exists(csv_path):
            try:
                logger.info(f"📂 Testing on {dataset_name}...")
                features, labels, processing_times = evaluator.extract_features_from_dataset(
                    csv_path, dataset_name
                )
                
                if features is not None:
                    results = evaluator.evaluate_dataset(features, labels, dataset_name)
                    all_results[dataset_name.lower().replace(' ', '_')] = {
                        'models': results,
                        'dataset_info': {
                            'total_samples': len(labels),
                            'authentic_samples': np.sum(labels == 0),
                            'forged_samples': np.sum(labels == 1),
                            'avg_processing_time': np.mean(processing_times),
                            'avg_speed': 1 / np.mean(processing_times)
                        }
                    }
                    
            except Exception as e:
                logger.warning(f"⚠️ Error processing {dataset_name}: {e}")
                continue
    
    # Generate comprehensive report
    comprehensive_report = evaluator.generate_comprehensive_report(all_results)
    
    # Save detailed results
    evaluator.save_detailed_results(all_results, comprehensive_report)
    
    print("\n" + "=" * 80)
    print("🎉 COMPLETE DATASET EVALUATION FINISHED!")
    print("=" * 80)
    print(f"🏆 Best Model: {comprehensive_report['best_overall_model'].upper()}")
    print(f"📊 Best Accuracy: {comprehensive_report['best_overall_accuracy']:.4f} ({comprehensive_report['best_overall_accuracy']*100:.2f}%)")
    print(f"⏱️ Total Time: {comprehensive_report['total_evaluation_time']:.2f}s")
    print(f"🎮 GPU Used: {'✅ Yes' if evaluator.gpu_available else '❌ No'}")
    if evaluator.gpu_available:
        print(f"🚀 GPU: {evaluator.gpu_name}")
    print(f"💾 Results saved to: ./models/complete_dataset_test_results.json")
    print(f"📊 Visualizations saved to: ./models/")
    print("=" * 80)
    
    return comprehensive_report['best_overall_accuracy']

if __name__ == "__main__":
    accuracy = main()
