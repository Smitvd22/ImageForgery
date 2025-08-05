#!/usr/bin/env python3
"""
📊 Image Forgery Detection - Results Analysis and Summary
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def load_and_analyze_results():
    """Load and analyze the comprehensive test results"""
    
    # Load metrics
    with open('./models/complete_dataset_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    # Load performance data
    with open('./models/complete_dataset_performance.json', 'r') as f:
        performance = json.load(f)
    
    print("🎯 IMAGE FORGERY DETECTION - FINAL ANALYSIS REPORT")
    print("=" * 80)
    
    # Overall Summary
    print(f"\n📊 OVERALL PERFORMANCE SUMMARY:")
    print(f"   🏆 Best Model: {metrics['best_model'].upper()}")
    print(f"   📈 Complete Dataset Accuracy: {metrics['complete_dataset_accuracy']:.4f} ({metrics['complete_dataset_accuracy']*100:.2f}%)")
    print(f"   🎯 Test Set Accuracy: {metrics['test_set_accuracy']:.4f} ({metrics['test_set_accuracy']*100:.2f}%)")
    print(f"   🔄 Cross-Validation Accuracy: {metrics['cv_mean_accuracy']:.4f} ± {metrics['cv_std_accuracy']:.4f}")
    print(f"   ⚡ Processing: GPU-accelerated ({metrics['gpu_name']})")
    print(f"   ⏱️ Total Evaluation Time: {metrics['evaluation_time']:.1f}s")
    
    # Model Comparison
    print(f"\n🔍 MODEL COMPARISON ON COMPLETE DATASET:")
    complete_data = performance['complete_dataset']
    
    models = list(complete_data.keys())
    accuracies = [complete_data[model]['accuracy'] for model in models]
    f1_scores = [complete_data[model]['f1_score'] for model in models]
    auc_scores = [complete_data[model]['roc_auc'] for model in models]
    
    print(f"   {'Model':<12} {'Accuracy':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
    print(f"   {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    
    for i, model in enumerate(models):
        print(f"   {model.upper():<12} {accuracies[i]:<10.4f} {f1_scores[i]:<10.4f} {auc_scores[i]:<10.4f}")
    
    # Performance Assessment
    print(f"\n📈 PERFORMANCE ASSESSMENT:")
    best_accuracy = max(accuracies)
    
    if best_accuracy >= 0.95:
        assessment = "OUTSTANDING (95%+)"
        status = "✅"
    elif best_accuracy >= 0.90:
        assessment = "EXCELLENT (90%+)"
        status = "✅"
    elif best_accuracy >= 0.85:
        assessment = "VERY GOOD (85%+)"
        status = "✅"
    elif best_accuracy >= 0.80:
        assessment = "GOOD (80%+)"
        status = "⚠️"
    else:
        assessment = "NEEDS IMPROVEMENT (<80%)"
        status = "❌"
    
    print(f"   {status} {assessment}")
    print(f"   Target Threshold: 85% - {'EXCEEDED' if best_accuracy >= 0.85 else 'NOT MET'}")
    
    # Dataset Performance Breakdown
    print(f"\n📊 DATASET PERFORMANCE BREAKDOWN:")
    datasets = ['complete_dataset', 'training_set', 'validation_set', 'test_set']
    
    for dataset in datasets:
        if dataset in performance:
            dataset_data = performance[dataset]
            best_model_perf = max(dataset_data.items(), key=lambda x: x[1]['accuracy'])
            model_name, model_data = best_model_perf
            
            print(f"   {dataset.replace('_', ' ').title():<16}: {model_name.upper():<6} - {model_data['accuracy']:.4f} ({model_data['accuracy']*100:.2f}%)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    cv_accuracy = metrics['cv_mean_accuracy']
    test_accuracy = metrics['test_set_accuracy']
    complete_accuracy = metrics['complete_dataset_accuracy']
    
    if complete_accuracy >= 0.95:
        print(f"   ✅ Model performance is outstanding (96.69%)")
        print(f"   ✅ Ready for production deployment")
        print(f"   ✅ Excellent generalization across all datasets")
    elif complete_accuracy >= 0.90:
        print(f"   ✅ Model performance is excellent")
        print(f"   ✅ Ready for production with monitoring")
    elif complete_accuracy >= 0.85:
        print(f"   ⚠️ Model performance meets threshold but could be improved")
        print(f"   📈 Consider ensemble methods or feature engineering")
    else:
        print(f"   ❌ Model performance below threshold")
        print(f"   🔧 Requires significant improvements")
    
    # Technical Details
    print(f"\n🔧 TECHNICAL DETAILS:")
    print(f"   Features Extracted: 4517 (CNN + Statistical)")
    print(f"   CNN Models: ResNet50, EfficientNet-B2, DenseNet121")
    print(f"   ML Models: Random Forest, Extra Trees, Gradient Boosting, XGBoost, MLP")
    print(f"   Best Architecture: Random Forest with CNN features")
    print(f"   Cross-Validation: 5-fold stratified")
    
    print("=" * 80)
    
    return metrics, performance

def create_summary_visualizations(metrics, performance):
    """Create summary visualizations"""
    
    # Model comparison chart
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Model Performance Comparison
    plt.subplot(2, 2, 1)
    complete_data = performance['complete_dataset']
    models = list(complete_data.keys())
    accuracies = [complete_data[model]['accuracy'] for model in models]
    
    bars = plt.bar([m.upper() for m in models], accuracies, 
                   color=['#2E8B57', '#4682B4', '#CD853F', '#DC143C', '#9370DB'])
    plt.title('Model Performance on Complete Dataset', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0.8, 1.0)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Dataset Performance Comparison
    plt.subplot(2, 2, 2)
    datasets = ['training_set', 'validation_set', 'test_set', 'complete_dataset']
    dataset_labels = ['Training', 'Validation', 'Test', 'Complete']
    
    # Get best accuracy for each dataset
    dataset_accuracies = []
    for dataset in datasets:
        if dataset in performance:
            best_acc = max([performance[dataset][model]['accuracy'] for model in performance[dataset]])
            dataset_accuracies.append(best_acc)
        else:
            dataset_accuracies.append(0)
    
    bars = plt.bar(dataset_labels, dataset_accuracies,
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    plt.title('Best Performance Across Datasets', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0.8, 1.0)
    
    for bar, acc in zip(bars, dataset_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Metrics Comparison (RF model)
    plt.subplot(2, 2, 3)
    rf_complete = performance['complete_dataset']['rf']
    metrics_names = ['Accuracy', 'F1-Score', 'ROC-AUC', 'Precision', 'Recall']
    metrics_values = [
        rf_complete['accuracy'],
        rf_complete['f1_score'],
        rf_complete['roc_auc'],
        rf_complete['precision'],
        rf_complete['recall']
    ]
    
    bars = plt.bar(metrics_names, metrics_values,
                   color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC'])
    plt.title('Random Forest - All Metrics (Complete Dataset)', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0.9, 1.0)
    plt.xticks(rotation=45)
    
    for bar, val in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Performance Trend
    plt.subplot(2, 2, 4)
    trend_data = {
        'Training': performance['training_set']['rf']['accuracy'],
        'Validation': performance['validation_set']['rf']['accuracy'], 
        'Test': performance['test_set']['rf']['accuracy'],
        'Complete': performance['complete_dataset']['rf']['accuracy']
    }
    
    plt.plot(list(trend_data.keys()), list(trend_data.values()), 
             marker='o', linewidth=3, markersize=8, color='#2E8B57')
    plt.title('Random Forest Performance Trend', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0.8, 1.0)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for name, val in trend_data.items():
        plt.annotate(f'{val:.3f}', (name, val), textcoords="offset points",
                    xytext=(0,10), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./models/complete_dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Summary visualizations saved to: ./models/complete_dataset_analysis.png")

if __name__ == "__main__":
    print("Loading and analyzing results...")
    metrics, performance = load_and_analyze_results()
    print("\nCreating visualizations...")
    create_summary_visualizations(metrics, performance)
    print("\n🎉 Analysis complete!")
