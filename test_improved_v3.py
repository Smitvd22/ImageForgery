#!/usr/bin/env python
"""
Complete Dataset Testing for Improved Image Forgery Detection v3.0
Tests the improved model on the complete dataset with detailed analysis
"""

import os
import numpy as np
import pandas as pd
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    balanced_accuracy_score
)
from sklearn.preprocessing import StandardScaler
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('improved_v3_dataset_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def safe_divide(a, b):
    """Safe division to avoid division by zero"""
    return a / b if b != 0 else 0.0

def extract_cnn_features(image_path, model, transform, device):
    """Extract CNN features from image"""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = model(img_tensor)
                return features.cpu().numpy().flatten()
    except Exception as e:
        logger.warning(f"CNN feature extraction failed for {image_path}: {e}")
        return None

def extract_basic_features(image_path):
    """Extract basic statistical features from image"""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)
            
            # Basic statistics for each channel
            features = []
            for channel in range(3):  # R, G, B
                channel_data = img_array[:, :, channel].flatten()
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.median(channel_data),
                    np.min(channel_data),
                    np.max(channel_data)
                ])
            
            return np.array(features)
    except Exception as e:
        logger.warning(f"Basic feature extraction failed for {image_path}: {e}")
        return np.zeros(15)  # Return zeros for failed extraction

def extract_consistent_features(image_paths, models_dict, transform, device):
    """Extract features consistently with training approach"""
    all_features = []
    
    logger.info("Extracting consistent CNN features...")
    for image_path in tqdm(image_paths, desc="Processing images"):
        # Extract CNN features from all three models
        cnn_features = []
        for model_name, model in models_dict.items():
            features = extract_cnn_features(image_path, model, transform, device)
            if features is not None:
                cnn_features.append(features)
            else:
                # Use zeros if extraction fails
                cnn_features.append(np.zeros(1000))  # Assuming 1000 features per model
        
        # Extract basic features
        basic_features = extract_basic_features(image_path)
        
        # Combine all features
        combined_features = np.concatenate([np.concatenate(cnn_features), basic_features])
        all_features.append(combined_features)
    
    return np.array(all_features)

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Authentic', 'Forged'],
                yticklabels=['Authentic', 'Forged'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return cm

def plot_roc_curve(y_true, y_scores, title="ROC Curve"):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    return roc_auc

def main():
    print("=" * 80)
    print("COMPLETE DATASET TESTING - IMPROVED MODEL v3.0")
    print("Testing improved model on complete dataset with detailed analysis")
    print("=" * 80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
    
    # Load all labels (complete dataset)
    labels_path = os.path.join("data", "labels.csv")
    if not os.path.exists(labels_path):
        logger.error("Complete labels file not found!")
        return
    
    df = pd.read_csv(labels_path)
    logger.info(f"Complete dataset: {len(df)} images")
    logger.info(f"Authentic: {len(df[df['label'] == 0])}, Forged: {len(df[df['label'] == 1])}")
    
    # Load improved models
    model_path = "models/improved_v3_best_model.pkl"
    scaler_path = "models/improved_v3_scaler.pkl"
    config_path = "models/improved_v3_config.json"
    
    if not all(os.path.exists(p) for p in [model_path, scaler_path, config_path]):
        logger.error("Improved model files not found!")
        return
    
    logger.info("Loading improved model components...")
    with open(model_path, 'rb') as f:
        best_model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    logger.info(f"Best model type: {model_config.get('best_model', 'Unknown')}")
    
    # Load pre-trained CNN models
    logger.info("Loading pre-trained models...")
    models_dict = {}
    
    # ResNet50
    resnet = models.resnet50(pretrained=True)
    resnet.fc = torch.nn.Identity()  # Remove classifier
    resnet.eval()
    resnet.to(device)
    models_dict['resnet50'] = resnet
    logger.info("[OK] Loaded resnet50")
    
    # EfficientNet-B2
    efficientnet = models.efficientnet_b2(pretrained=True)
    efficientnet.classifier = torch.nn.Identity()
    efficientnet.eval()
    efficientnet.to(device)
    models_dict['efficientnet_b2'] = efficientnet
    logger.info("[OK] Loaded efficientnet_b2")
    
    # DenseNet121
    densenet = models.densenet121(pretrained=True)
    densenet.classifier = torch.nn.Identity()
    densenet.eval()
    densenet.to(device)
    models_dict['densenet121'] = densenet
    logger.info("[OK] Loaded densenet121")
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Extract features for complete dataset
    logger.info("Extracting features for complete dataset...")
    image_paths = []
    labels = []
    
    for _, row in df.iterrows():
        image_path = os.path.join("data", row['file_path'])
        if os.path.exists(image_path):
            image_paths.append(image_path)
            labels.append(row['label'])
        else:
            logger.warning(f"Image not found: {image_path}")
    
    logger.info(f"Processing {len(image_paths)} images...")
    
    # Extract features
    features = extract_consistent_features(image_paths, models_dict, transform, device)
    logger.info(f"Extracted features shape: {features.shape}")
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make predictions
    logger.info("Making predictions...")
    y_true = np.array(labels)
    y_pred = best_model.predict(features_scaled)
    y_scores = best_model.predict_proba(features_scaled)[:, 1]  # Probability of forged class
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Calculate sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = safe_divide(tp, tp + fn)  # True Positive Rate
    specificity = safe_divide(tn, tn + fp)  # True Negative Rate
    
    print("\n" + "=" * 60)
    print("IMPROVED MODEL v3.0 - COMPLETE DATASET RESULTS")
    print("=" * 60)
    print(f"Total Images Tested: {len(y_true)}")
    print(f"Authentic Images: {len(y_true[y_true == 0])}")
    print(f"Forged Images: {len(y_true[y_true == 1])}")
    print("-" * 60)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("-" * 60)
    print(f"Sensitivity (Forged Detection): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    print(f"Specificity (Authentic Detection): {specificity:.4f} ({specificity*100:.2f}%)")
    print("-" * 60)
    
    # Check if accuracy meets threshold
    threshold = 0.85
    if accuracy >= threshold:
        print(f"✅ SUCCESS: Accuracy {accuracy*100:.2f}% meets threshold of {threshold*100:.0f}%")
        status = "PASSED"
    else:
        print(f"❌ FAILED: Accuracy {accuracy*100:.2f}% below threshold of {threshold*100:.0f}%")
        status = "FAILED"
    
    print("=" * 60)
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Authentic', 'Forged']))
    
    # Confusion Matrix
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    cm = plot_confusion_matrix(y_true, y_pred, "Improved Model v3.0 - Confusion Matrix")
    
    # ROC Curve
    plt.subplot(1, 3, 2)
    roc_auc = plot_roc_curve(y_true, y_scores, "Improved Model v3.0 - ROC Curve")
    
    # Performance metrics bar plot
    plt.subplot(1, 3, 3)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Sensitivity', 'Specificity']
    values = [accuracy, precision, recall, f1, sensitivity, specificity]
    colors = ['green' if v >= 0.85 else 'orange' if v >= 0.75 else 'red' for v in values]
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.axhline(y=0.85, color='red', linestyle='--', alpha=0.7, label='Target (85%)')
    plt.ylim(0, 1)
    plt.title('Performance Metrics')
    plt.xticks(rotation=45)
    plt.legend()
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('improved_v3_complete_dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed results
    results = {
        'model_type': model_config.get('best_model', 'Unknown'),
        'total_images': len(y_true),
        'authentic_images': int(len(y_true[y_true == 0])),
        'forged_images': int(len(y_true[y_true == 1])),
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'threshold_met': accuracy >= threshold,
        'status': status
    }
    
    with open('improved_v3_complete_dataset_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Results saved to improved_v3_complete_dataset_results.json")
    logger.info("Analysis plot saved as improved_v3_complete_dataset_analysis.png")
    
    print(f"\n📊 Complete analysis saved!")
    print(f"📈 Results file: improved_v3_complete_dataset_results.json")
    print(f"🎯 Visualization: improved_v3_complete_dataset_analysis.png")
    
    return accuracy >= threshold

if __name__ == "__main__":
    main()
