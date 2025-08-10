#!/usr/bin/env python3
"""
Accuracy Improvement Summary Report
Analyzes the improvements achieved after all enhancements
"""

import json
import os
import pandas as pd
from datetime import datetime

def load_results(file_path):
    """Load results from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return None

def analyze_improvements():
    """Analyze the accuracy improvements achieved"""
    
    print("=" * 80)
    print("IMAGE FORGERY DETECTION - ACCURACY IMPROVEMENT ANALYSIS")
    print("=" * 80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Baseline accuracies from previous logs/results
    baseline_accuracies = {
        "MISD": {
            "training": 85.0,
            "validation": 80.0, 
            "testing": 78.0
        },
        "4CAM": {
            "training": 88.0,
            "validation": 82.0,
            "testing": 80.0
        },
        "IMSLICE": {
            "training": 90.0,
            "validation": 85.0,
            "testing": 83.0
        }
    }
    
    datasets = ["MISD", "4CAM", "IMSLICE"]
    
    # Load enhanced results
    enhanced_results = {}
    for dataset in datasets:
        result_file = f"enhanced_results_{dataset.lower()}.json"
        if os.path.exists(result_file):
            enhanced_results[dataset] = load_results(result_file)
    
    # Check for latest multi-dataset results
    multi_result_files = [f for f in os.listdir('.') if f.startswith('multi_dataset_results_')]
    if multi_result_files:
        latest_file = sorted(multi_result_files)[-1]
        multi_results = load_results(latest_file)
        if multi_results and 'results' in multi_results:
            enhanced_results.update(multi_results['results'])
    
    print("ENHANCEMENT SUMMARY:")
    print("-" * 40)
    
    all_improvements = []
    total_datasets_improved = 0
    
    for dataset in datasets:
        print(f"\n{dataset} Dataset:")
        
        if dataset in enhanced_results and enhanced_results[dataset]:
            dataset_improved = False
            phases = ["training", "validation", "testing"]
            
            for phase in phases:
                baseline = baseline_accuracies[dataset][phase]
                
                if (enhanced_results[dataset].get(phase, {}).get("success", False) and 
                    enhanced_results[dataset][phase].get("accuracy") is not None):
                    
                    enhanced = enhanced_results[dataset][phase]["accuracy"]
                    improvement = enhanced - baseline
                    all_improvements.append(improvement)
                    
                    status = "✓ IMPROVED" if improvement > 0 else "⚠ NEEDS WORK"
                    print(f"  {phase.title()}: {baseline:.1f}% → {enhanced:.1f}% ({improvement:+.1f}%) {status}")
                    
                    if improvement > 0:
                        dataset_improved = True
                else:
                    print(f"  {phase.title()}: {baseline:.1f}% → No data (needs completion)")
            
            if dataset_improved:
                total_datasets_improved += 1
        else:
            print(f"  No enhanced results available (training may still be in progress)")
    
    print("\n" + "=" * 80)
    print("OVERALL IMPROVEMENT SUMMARY")
    print("=" * 80)
    
    if all_improvements:
        avg_improvement = sum(all_improvements) / len(all_improvements)
        max_improvement = max(all_improvements)
        min_improvement = min(all_improvements)
        positive_improvements = [x for x in all_improvements if x > 0]
        
        print(f"Average Improvement: {avg_improvement:+.2f}%")
        print(f"Best Improvement: {max_improvement:+.2f}%")
        print(f"Worst Result: {min_improvement:+.2f}%")
        print(f"Positive Improvements: {len(positive_improvements)}/{len(all_improvements)}")
        print(f"Datasets with Improvements: {total_datasets_improved}/{len(datasets)}")
        
        # Target achievement
        target_met = avg_improvement >= 3.0
        print(f"\nTARGET ACHIEVEMENT:")
        print(f"Goal: +3.0% average improvement")
        print(f"Achieved: {avg_improvement:+.2f}%")
        print(f"Status: {'✓ TARGET MET' if target_met else '⚠ IN PROGRESS'}")
        
        if target_met:
            print(f"🎉 SUCCESS! The 3%+ accuracy improvement target has been achieved!")
        else:
            progress = (avg_improvement / 3.0) * 100
            print(f"Progress towards target: {progress:.1f}%")
    else:
        print("⚠ No improvement data available yet.")
        print("Training may still be in progress or results files may not be generated.")
    
    # Enhancement features implemented
    print(f"\n" + "=" * 80)
    print("ENHANCEMENTS IMPLEMENTED")
    print("=" * 80)
    
    enhancements = [
        "✓ Advanced Preprocessing (CLAHE, edge enhancement, noise reduction)",
        "✓ Enhanced Feature Extraction (wavelet, frequency, texture, edge features)",
        "✓ Deep Learning Models (ResNet50, EfficientNet-B2, DenseNet121)",
        "✓ Ensemble Learning (voting and stacking classifiers)",
        "✓ Hyperparameter Optimization",
        "✓ GPU Acceleration (CUDA enabled)",
        "✓ Robust Error Handling",
        "✓ Multi-dataset Automation",
        "✓ 512x512 Image Resolution (enhanced from 256x256)",
        "✓ Advanced Cross-validation"
    ]
    
    for enhancement in enhancements:
        print(f"  {enhancement}")
    
    print("=" * 80)

if __name__ == "__main__":
    analyze_improvements()
