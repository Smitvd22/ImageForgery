#!/usr/bin/env python3
"""
📊 Model Evaluation Utilities
Simple evaluation functions for the Image Forgery Detection System
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import json
from pathlib import Path

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate comprehensive metrics for binary classification
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
    
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary'),
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """Plot and optionally save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Authentic', 'Forged'],
                yticklabels=['Authentic', 'Forged'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved: {save_path}")
    
    plt.show()
    return cm

def plot_roc_curve(y_true, y_prob, title="ROC Curve", save_path=None):
    """Plot and optionally save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curve saved: {save_path}")
    
    plt.show()

def plot_precision_recall_curve(y_true, y_prob, title="Precision-Recall Curve", save_path=None):
    """Plot and optionally save Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ PR curve saved: {save_path}")
    
    plt.show()

def save_metrics(metrics, filepath):
    """Save metrics to JSON file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Metrics saved: {filepath}")

def print_classification_report(y_true, y_pred, target_names=None):
    """Print detailed classification report"""
    if target_names is None:
        target_names = ['Authentic', 'Forged']
    
    print("\n📊 Classification Report:")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=target_names))

def evaluate_model_comprehensive(y_true, y_pred, y_prob=None, 
                                save_dir=None, model_name="model"):
    """
    Comprehensive model evaluation with all metrics and plots
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        y_prob: Prediction probabilities
        save_dir: Directory to save results
        model_name: Name for saving files
    """
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # Print results
    print(f"\n🎯 {model_name.title()} Evaluation Results:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Classification report
    print_classification_report(y_true, y_pred)
    
    # Save metrics
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_metrics(metrics, save_dir / f"{model_name}_metrics.json")
    
    # Generate plots
    if save_dir:
        plot_confusion_matrix(y_true, y_pred, 
                            title=f"{model_name.title()} - Confusion Matrix",
                            save_path=save_dir / f"{model_name}_confusion_matrix.png")
        
        if y_prob is not None:
            plot_roc_curve(y_true, y_prob,
                          title=f"{model_name.title()} - ROC Curve", 
                          save_path=save_dir / f"{model_name}_roc_curve.png")
            
            plot_precision_recall_curve(y_true, y_prob,
                                      title=f"{model_name.title()} - PR Curve",
                                      save_path=save_dir / f"{model_name}_pr_curve.png")
    else:
        # Just show plots without saving
        plot_confusion_matrix(y_true, y_pred, title=f"{model_name.title()} - Confusion Matrix")
        if y_prob is not None:
            plot_roc_curve(y_true, y_prob, title=f"{model_name.title()} - ROC Curve")
            plot_precision_recall_curve(y_true, y_prob, title=f"{model_name.title()} - PR Curve")
    
    return metrics
