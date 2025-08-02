#!/usr/bin/env python3
"""
Calculate performance metrics from test results CSV
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics_from_csv(csv_path):
    """Calculate all performance metrics from test results CSV"""
    print("📊 Calculating Performance Metrics from Test Results")
    print("=" * 60)
    
    # Load the test results
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} test samples from {csv_path}")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None
    
    # Extract true labels and predictions
    true_labels = df['true_label'].values
    predicted_labels = df['predicted_label'].values
    forged_probabilities = df['forged_prob'].values
    
    # Calculate basic metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    
    # Calculate per-class metrics
    precision_per_class = precision_score(true_labels, predicted_labels, average=None)
    recall_per_class = recall_score(true_labels, predicted_labels, average=None)
    f1_per_class = f1_score(true_labels, predicted_labels, average=None)
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(true_labels, forged_probabilities)
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Display results
    print("\\n🎯 OVERALL PERFORMANCE METRICS")
    print("-" * 40)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"ROC AUC:   {roc_auc:.4f}")
    
    print("\\n📈 PER-CLASS METRICS")
    print("-" * 40)
    class_names = ['Authentic (0)', 'Forged (1)']
    for i, class_name in enumerate(class_names):
        print(f"{class_name}:")
        print(f"  Precision: {precision_per_class[i]:.4f} ({precision_per_class[i]*100:.2f}%)")
        print(f"  Recall:    {recall_per_class[i]:.4f} ({recall_per_class[i]*100:.2f}%)")
        print(f"  F1 Score:  {f1_per_class[i]:.4f} ({f1_per_class[i]*100:.2f}%)")
    
    print("\\n🔲 CONFUSION MATRIX")
    print("-" * 40)
    print("Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Authentic  Forged")
    print(f"True Authentic    {cm[0,0]:3d}      {cm[0,1]:3d}")
    print(f"     Forged       {cm[1,0]:3d}      {cm[1,1]:3d}")
    
    # Calculate additional metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    
    print("\\n📊 DETAILED CONFUSION MATRIX ANALYSIS")
    print("-" * 40)
    print(f"True Negatives (TN):  {tn:3d} (Correctly identified authentic)")
    print(f"False Positives (FP): {fp:3d} (Authentic wrongly classified as forged)")
    print(f"False Negatives (FN): {fn:3d} (Forged wrongly classified as authentic)")
    print(f"True Positives (TP):  {tp:3d} (Correctly identified forged)")
    
    # Calculate sensitivity, specificity, etc.
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for positive class
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for negative class
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision for positive class
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
    
    print("\\n🧮 ADDITIONAL METRICS")
    print("-" * 40)
    print(f"Sensitivity (TPR): {sensitivity:.4f} ({sensitivity*100:.2f}%) - Ability to detect forged images")
    print(f"Specificity (TNR): {specificity:.4f} ({specificity*100:.2f}%) - Ability to detect authentic images")
    print(f"Positive Pred Val: {ppv:.4f} ({ppv*100:.2f}%) - When predicting forged, how often correct")
    print(f"Negative Pred Val: {npv:.4f} ({npv*100:.2f}%) - When predicting authentic, how often correct")
    
    # Generate classification report
    print("\\n📋 DETAILED CLASSIFICATION REPORT")
    print("-" * 40)
    print(classification_report(true_labels, predicted_labels, 
                              target_names=['Authentic', 'Forged'], 
                              digits=4))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Authentic', 'Forged'],
               yticklabels=['Authentic', 'Forged'])
    plt.title('Confusion Matrix - Image Forgery Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Return metrics dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'positive_predictive_value': ppv,
        'negative_predictive_value': npv,
        'per_class_precision': precision_per_class,
        'per_class_recall': recall_per_class,
        'per_class_f1': f1_per_class
    }

if __name__ == "__main__":
    # Calculate metrics from the test results CSV
    csv_path = "./models/improved_test_results.csv"
    metrics = calculate_metrics_from_csv(csv_path)
    
    if metrics:
        print("\\n✅ Metrics calculation completed successfully!")
    else:
        print("\\n❌ Failed to calculate metrics!")
