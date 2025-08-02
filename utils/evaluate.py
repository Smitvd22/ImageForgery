#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script for Image Forgery Detection
Provides detailed metrics, visualizations, and performance analysis
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tqdm import tqdm

# Import project modules
from config import *
from models import ImprovedMultiModelExtractor, EnhancedFeatureStatistics
from dataset import ForgeryDataset, get_data_loaders
from classifier import XGBoostClassifier
from core.preprocessing import preprocess_image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluator for the image forgery detection system
    """
    
    def __init__(self, model_dir="./models"):
        self.model_dir = Path(model_dir)
        self.device = DEVICE
        self.results = {}
        
        # Load trained models
        self.load_models()
        
    def load_models(self):
        """Load all trained models"""
        try:
            # Load XGBoost classifier
            xgb_path = self.model_dir / "improved_xgb_forgery_detector.pkl"
            if xgb_path.exists():
                self.classifier = joblib.load(xgb_path)
                logger.info("✅ XGBoost classifier loaded successfully")
            else:
                logger.error(f"❌ XGBoost model not found at {xgb_path}")
                self.classifier = None
            
            # Load feature scaler
            scaler_path = self.model_dir / "improved_feature_scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("✅ Feature scaler loaded successfully")
            else:
                logger.error(f"❌ Feature scaler not found at {scaler_path}")
                self.scaler = None
            
            # Load feature extractor
            extractor_path = self.model_dir / "improved_feature_extractor.pth"
            if extractor_path.exists():
                self.feature_extractor = ImprovedMultiModelExtractor(
                    use_huggingface=USE_HUGGINGFACE,
                    dropout_rate=0.3
                )
                state_dict = torch.load(extractor_path, map_location=self.device)
                self.feature_extractor.load_state_dict(state_dict)
                self.feature_extractor.to(self.device)
                self.feature_extractor.eval()
                logger.info("✅ Feature extractor loaded successfully")
            else:
                logger.warning(f"⚠️ Feature extractor not found at {extractor_path}")
                self.feature_extractor = None
                
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            self.classifier = None
            self.scaler = None
            self.feature_extractor = None

    def extract_features_from_loader(self, data_loader, description="Extracting features"):
        """Extract features from a data loader"""
        if self.feature_extractor is None:
            logger.error("❌ Feature extractor not available")
            return None, None
            
        features = []
        labels = []
        
        logger.info(f"{description}...")
        self.feature_extractor.eval()
        
        with torch.no_grad():
            for batch_idx, (images, batch_labels) in enumerate(tqdm(data_loader, desc=description)):
                images = images.to(self.device)
                
                # Extract features
                batch_features = self.feature_extractor(images)
                
                # Move to CPU and convert to numpy
                features.append(batch_features.cpu().numpy())
                labels.append(batch_labels.numpy())
        
        # Concatenate all features
        features = np.vstack(features)
        labels = np.concatenate(labels)
        
        return features, labels

    def evaluate_on_test_set(self):
        """Evaluate the model on the test set"""
        logger.info("\n" + "="*60)
        logger.info("EVALUATING ON TEST SET")
        logger.info("="*60)
        
        if self.classifier is None or self.scaler is None:
            logger.error("❌ Models not loaded properly")
            return None
        
        try:
            # Load test data
            test_dataset = ForgeryDataset(
                csv_path=TEST_CSV,
                apply_augmentation=False,
                image_size=IMAGE_SIZE
            )
            
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY
            )
            
            logger.info(f"Test dataset size: {len(test_dataset)}")
            
            # Extract features if feature extractor is available
            if self.feature_extractor is not None:
                test_features, test_labels = self.extract_features_from_loader(
                    test_loader, "Extracting test features"
                )
                
                # Scale features
                test_features_scaled = self.scaler.transform(test_features)
            else:
                # Use simple feature extraction as fallback
                logger.warning("⚠️ Using simple feature extraction")
                test_features_scaled, test_labels = self.extract_simple_features_from_dataset(test_dataset)
            
            # Make predictions
            logger.info("Making predictions...")
            predictions = self.classifier.predict(test_features_scaled)
            probabilities = self.classifier.predict_proba(test_features_scaled)
            
            # Calculate metrics
            metrics = self.calculate_metrics(test_labels, predictions, probabilities)
            
            # Store results
            self.results['test_evaluation'] = {
                'metrics': metrics,
                'predictions': predictions,
                'probabilities': probabilities,
                'true_labels': test_labels
            }
            
            # Display results
            self.display_evaluation_results(metrics, "Test Set")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Test set evaluation failed: {e}")
            return None

    def extract_simple_features_from_dataset(self, dataset):
        """Extract simple features from dataset (fallback method)"""
        features = []
        labels = []
        
        logger.info("Extracting simple features...")
        
        for i in tqdm(range(len(dataset)), desc="Processing images"):
            try:
                data_item = dataset[i]
                if isinstance(data_item, tuple):
                    image_tensor, label = data_item
                else:
                    continue
                
                # Convert tensor to numpy if needed
                if isinstance(image_tensor, torch.Tensor):
                    image_array = image_tensor.numpy()
                else:
                    image_array = image_tensor
                
                # Extract basic features
                simple_features = self.extract_basic_features(image_array)
                features.append(simple_features)
                labels.append(label)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to process item {i}: {e}")
                continue
        
        if len(features) == 0:
            logger.error("❌ No features extracted")
            return None, None
        
        features = np.vstack(features)
        labels = np.array(labels)
        
        return features, labels

    def extract_basic_features(self, image_array):
        """Extract basic statistical features from image"""
        # Ensure proper shape and range
        if len(image_array.shape) == 3 and image_array.shape[0] <= 3:
            image_array = np.transpose(image_array, (1, 2, 0))
        
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        
        features = []
        
        # Color channel statistics
        if len(image_array.shape) == 3:
            for c in range(min(3, image_array.shape[2])):
                channel = image_array[:, :, c].flatten()
                features.extend([
                    np.mean(channel), np.std(channel), np.median(channel),
                    np.percentile(channel, 25), np.percentile(channel, 75)
                ])
        
        # Grayscale features
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        features.extend([
            np.mean(gray), np.std(gray), np.var(gray),
            np.min(gray), np.max(gray)
        ])
        
        # Pad to 140 features
        while len(features) < 140:
            features.append(0.0)
        
        return np.array(features[:140]).reshape(1, -1)

    def calculate_metrics(self, true_labels, predictions, probabilities):
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic metrics
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
        
        # Classification report
        metrics['classification_report'] = classification_report(
            true_labels, predictions, 
            target_names=['Authentic', 'Forged'],
            output_dict=True
        )
        
        return metrics

    def display_evaluation_results(self, metrics, dataset_name):
        """Display evaluation results"""
        logger.info(f"\n{dataset_name} Evaluation Results:")
        logger.info("-" * 40)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1_score']:.4f}")
        
        if metrics['roc_auc'] is not None:
            logger.info(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"{metrics['confusion_matrix']}")

    def plot_confusion_matrix(self, metrics, title="Confusion Matrix"):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        cm = metrics['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Authentic', 'Forged'],
                   yticklabels=['Authentic', 'Forged'])
        
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, true_labels, probabilities, title="ROC Curve"):
        """Plot ROC curve"""
        if probabilities.shape[1] != 2:
            logger.warning("⚠️ Cannot plot ROC curve - not binary classification")
            return
        
        fpr, tpr, _ = roc_curve(true_labels, probabilities[:, 1])
        auc_score = roc_auc_score(true_labels, probabilities[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_precision_recall_curve(self, true_labels, probabilities, title="Precision-Recall Curve"):
        """Plot precision-recall curve"""
        if probabilities.shape[1] != 2:
            logger.warning("⚠️ Cannot plot PR curve - not binary classification")
            return
        
        precision, recall, _ = precision_recall_curve(true_labels, probabilities[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def perform_cross_validation(self):
        """Perform cross-validation evaluation"""
        logger.info("\n" + "="*60)
        logger.info("CROSS-VALIDATION EVALUATION")
        logger.info("="*60)
        
        if self.classifier is None:
            logger.error("❌ Classifier not loaded")
            return None
        
        try:
            # Load full dataset for cross-validation
            full_dataset = ForgeryDataset(
                csv_path=DATA_CSV,
                apply_augmentation=False,
                image_size=IMAGE_SIZE
            )
            
            # Extract features and labels
            if self.feature_extractor is not None:
                full_loader = torch.utils.data.DataLoader(
                    full_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    pin_memory=PIN_MEMORY
                )
                
                features, labels = self.extract_features_from_loader(
                    full_loader, "Extracting features for CV"
                )
                features = self.scaler.transform(features)
            else:
                features, labels = self.extract_simple_features_from_dataset(full_dataset)
                features = self.scaler.transform(features)
            
            # Perform stratified cross-validation
            cv = StratifiedKFold(
                n_splits=CV_CONFIG['cv_folds'],
                shuffle=CV_CONFIG['shuffle'],
                random_state=CV_CONFIG['random_state']
            )
            
            # Get a fresh copy of the classifier for CV
            cv_classifier = XGBoostClassifier(**XGB_PARAMS)
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                cv_classifier.get_cv_model(),
                features, labels,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1
            )
            
            # Store results
            self.results['cross_validation'] = {
                'scores': cv_scores,
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std()
            }
            
            # Display results
            logger.info(f"Cross-Validation Results ({CV_CONFIG['cv_folds']}-fold):")
            logger.info(f"Scores: {cv_scores}")
            logger.info(f"Mean Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            logger.info(f"Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
            
            return cv_scores
            
        except Exception as e:
            logger.error(f"❌ Cross-validation failed: {e}")
            return None

    def analyze_feature_importance(self):
        """Analyze and visualize feature importance"""
        logger.info("\n" + "="*60)
        logger.info("FEATURE IMPORTANCE ANALYSIS")
        logger.info("="*60)
        
        if self.classifier is None or not hasattr(self.classifier, 'feature_importance_'):
            logger.error("❌ Feature importance not available")
            return None
        
        try:
            importance = self.classifier.feature_importance_
            
            # Get top N important features
            top_n = 20
            indices = np.argsort(importance)[::-1][:top_n]
            
            # Display top features
            logger.info(f"Top {top_n} Most Important Features:")
            for i, idx in enumerate(indices):
                logger.info(f"{i+1:2d}. Feature {idx:3d}: {importance[idx]:.4f}")
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            plt.bar(range(top_n), importance[indices])
            plt.title(f'Top {top_n} Feature Importance')
            plt.xlabel('Feature Rank')
            plt.ylabel('Importance')
            plt.xticks(range(top_n), [f'F{idx}' for idx in indices], rotation=45)
            plt.tight_layout()
            plt.show()
            
            self.results['feature_importance'] = {
                'importance': importance,
                'top_features': indices
            }
            
            return importance
            
        except Exception as e:
            logger.error(f"❌ Feature importance analysis failed: {e}")
            return None

    def generate_comprehensive_report(self):
        """Generate a comprehensive evaluation report"""
        logger.info("\n" + "📊"*20)
        logger.info("COMPREHENSIVE EVALUATION REPORT")
        logger.info("📊"*20)
        
        # Summary statistics
        if 'test_evaluation' in self.results:
            test_metrics = self.results['test_evaluation']['metrics']
            logger.info(f"\n🎯 Test Set Performance:")
            logger.info(f"   Accuracy:  {test_metrics['accuracy']:.1%}")
            logger.info(f"   Precision: {test_metrics['precision']:.1%}")
            logger.info(f"   Recall:    {test_metrics['recall']:.1%}")
            logger.info(f"   F1 Score:  {test_metrics['f1_score']:.1%}")
            if test_metrics['roc_auc']:
                logger.info(f"   ROC AUC:   {test_metrics['roc_auc']:.3f}")
        
        if 'cross_validation' in self.results:
            cv_results = self.results['cross_validation']
            logger.info(f"\n🔄 Cross-Validation Performance:")
            logger.info(f"   Mean Accuracy: {cv_results['mean_score']:.1%} ± {cv_results['std_score']:.3f}")
        
        # Model characteristics
        logger.info(f"\n🔧 Model Configuration:")
        logger.info(f"   Architecture: Multi-backbone (ResNet++, U-Net, U-Net R)")
        logger.info(f"   Classifier: XGBoost")
        logger.info(f"   Image Size: {IMAGE_SIZE}")
        logger.info(f"   Device: {self.device}")
        
        # Performance summary
        logger.info(f"\n✅ Evaluation Complete!")
        
        return self.results

    def save_results(self, output_path="evaluation_results.csv"):
        """Save evaluation results to file"""
        try:
            if 'test_evaluation' in self.results:
                test_data = self.results['test_evaluation']
                
                # Create results DataFrame
                results_df = pd.DataFrame({
                    'true_label': test_data['true_labels'],
                    'predicted_label': test_data['predictions'],
                    'probability_authentic': test_data['probabilities'][:, 0],
                    'probability_forged': test_data['probabilities'][:, 1]
                })
                
                results_df.to_csv(output_path, index=False)
                logger.info(f"✅ Results saved to {output_path}")
                
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")


def main():
    """Main evaluation function"""
    print("Image Forgery Detection - Model Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Run comprehensive evaluation
    try:
        # Test set evaluation
        test_metrics = evaluator.evaluate_on_test_set()
        
        # Cross-validation
        cv_scores = evaluator.perform_cross_validation()
        
        # Feature importance analysis
        feature_importance = evaluator.analyze_feature_importance()
        
        # Generate comprehensive report
        results = evaluator.generate_comprehensive_report()
        
        # Save results
        evaluator.save_results()
        
        # Generate visualizations if test evaluation was successful
        if 'test_evaluation' in evaluator.results:
            test_data = evaluator.results['test_evaluation']
            
            # Plot confusion matrix
            evaluator.plot_confusion_matrix(test_data['metrics'], "Test Set Confusion Matrix")
            
            # Plot ROC curve
            evaluator.plot_roc_curve(
                test_data['true_labels'],
                test_data['probabilities'],
                "Test Set ROC Curve"
            )
            
            # Plot precision-recall curve
            evaluator.plot_precision_recall_curve(
                test_data['true_labels'],
                test_data['probabilities'],
                "Test Set Precision-Recall Curve"
            )
        
        logger.info("\n🎉 Evaluation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
