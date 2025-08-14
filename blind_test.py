#!/usr/bin/env python3
"""
Ultimate test: Copy images with neutral names to test if model uses image content or filenames
"""
import os
import shutil
import tempfile
from pathlib import Path
import random
import pandas as pd

# Import the active dataset from config
import sys
sys.path.append('.')
from core.config import ACTIVE_DATASET
dataset = ACTIVE_DATASET

def create_filename_blind_test():
    """Create a test where filenames don't reveal the class"""
    
    print("="*80)
    print("FILENAME-BLIND TEST")
    print("="*80)
    
    # Load dataset
    csv_path = "./data/"+dataset+"_labels.csv"
    df = pd.read_csv(csv_path)
    
    # Create temporary directory for renamed files
    test_dir = Path("./temp_blind_test")
    test_dir.mkdir(exist_ok=True)
    
    # Sample some files from each class (random each time)
    authentic_samples = df[df['label'] == 0].sample(n=5)
    forged_samples = df[df['label'] == 1].sample(n=5)
    
    test_cases = []
    
    print("Creating test files with neutral names...")
    
    # Copy authentic files with neutral names
    for i, (_, row) in enumerate(authentic_samples.iterrows()):
        if os.path.exists(row['filepath']):
            original_ext = Path(row['filepath']).suffix
            new_name = f"image_{i+1:03d}{original_ext}"
            new_path = test_dir / new_name
            shutil.copy2(row['filepath'], new_path)
            test_cases.append({
                'new_name': new_name,
                'new_path': str(new_path),
                'original_name': row['filename'],
                'true_label': 0,
                'true_class': 'Authentic'
            })
            print(f"  {row['filename']} -> {new_name} (Authentic)")
    
    # Copy forged files with neutral names
    for i, (_, row) in enumerate(forged_samples.iterrows()):
        if os.path.exists(row['filepath']):
            original_ext = Path(row['filepath']).suffix
            new_name = f"image_{i+6:03d}{original_ext}"
            new_path = test_dir / new_name
            shutil.copy2(row['filepath'], new_path)
            test_cases.append({
                'new_name': new_name,
                'new_path': str(new_path),
                'original_name': row['filename'],
                'true_label': 1,
                'true_class': 'Forged'
            })
            print(f"  {row['filename']} -> {new_name} (Forged)")
    
    return test_cases, test_dir

def run_blind_predictions(test_cases):
    """Run predictions on renamed files"""
    
    print(f"\n{'='*80}")
    print("RUNNING BLIND PREDICTIONS")
    print("="*80)
    
    # Import the predictor
    import sys
    sys.path.append('.')
    from predict_optimized import OptimizedPredictor
    
    # Initialize predictor
    predictor = OptimizedPredictor(dataset)
    
    # Load models
    if not predictor.load_models():
        print("❌ Failed to load models")
        return
    
    print("\nPredicting on renamed files (no filename cues):")
    print("-" * 80)
    
    correct_predictions = 0
    total_predictions = 0
    
    for case in test_cases:
        try:
            prediction, probability = predictor.predict(case['new_path'])
            
            if prediction is not None:
                predicted_class = 'Forged' if prediction == 1 else 'Authentic'
                is_correct = prediction == case['true_label']
                
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                status_icon = "✅" if is_correct else "❌"
                confidence = probability if probability is not None else 0.5
                
                print(f"{status_icon} {case['new_name']:<15} | True: {case['true_class']:<9} | "
                      f"Pred: {predicted_class:<9} | Conf: {confidence:.3f}")
                print(f"   Original: {case['original_name']}")
                print()
            else:
                print(f"❌ Failed to predict {case['new_name']}")
                
        except Exception as e:
            print(f"❌ Error predicting {case['new_name']}: {e}")
    
    # Calculate accuracy
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print("="*80)
        print(f"BLIND TEST RESULTS:")
        print(f"Correct predictions: {correct_predictions}/{total_predictions}")
        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print("="*80)
        
        # Interpretation
        if accuracy >= 0.8:
            print("✅ EXCELLENT: Model is truly analyzing image content!")
            print("   High accuracy despite removed filename cues suggests the model")
            print("   is learning meaningful visual features for forgery detection.")
        elif accuracy >= 0.6:
            print("⚠️ MODERATE: Model shows some real learning but may have issues")
            print("   Moderate accuracy suggests the model has learned some image")
            print("   features but may not be as robust as expected.")
        else:
            print("❌ POOR: Model appears to rely heavily on filename patterns!")
            print("   Low accuracy when filename cues are removed suggests the model")
            print("   was primarily using filename patterns rather than image content.")
        
        return accuracy
    else:
        print("❌ No successful predictions made")
        return 0.0

def cleanup_test_files(test_dir):
    """Clean up temporary test files"""
    try:
        shutil.rmtree(test_dir)
        print(f"\n🧹 Cleaned up test directory: {test_dir}")
    except Exception as e:
        print(f"⚠️ Could not clean up {test_dir}: {e}")

def main():
    """Main function to run the filename-blind test"""
    
    print("🔍 TESTING IF MODEL USES IMAGE CONTENT OR FILENAME PATTERNS")
    print("This test copies images with neutral names to remove filename cues.")
    print()
    
    try:
        # Create test files
        test_cases, test_dir = create_filename_blind_test()
        
        if not test_cases:
            print("❌ No test cases created")
            return
        
        # Run predictions
        accuracy = run_blind_predictions(test_cases)
        
        # Check if accuracy is None
        if accuracy is None:
            accuracy = 0.0
        
        # Final conclusion
        print("\n" + "="*80)
        print("FINAL CONCLUSION")
        print("="*80)
        
        if accuracy >= 0.8:
            print("🎉 CONCLUSION: Your model is LEGITIMATELY LEARNING!")
            print()
            print("✅ The model successfully identifies forged images even when")
            print("   filename patterns are removed, proving it analyzes actual")
            print("   image content using CNN features.")
            print()
            print("✅ The consistent results you observed are due to the model")
            print("   correctly learning visual patterns that distinguish")
            print("   authentic from forged images.")
            
        elif accuracy >= 0.6:
            print("🤔 CONCLUSION: Model shows MIXED BEHAVIOR")
            print()
            print("⚠️ The model has learned some image features but performance")
            print("   drops when filename cues are removed, suggesting it may")
            print("   be using a combination of image content and filename patterns.")
            
        else:
            print("🚨 CONCLUSION: Model is primarily using FILENAME PATTERNS!")
            print()
            print("❌ Poor performance when filename cues are removed strongly")
            print("   suggests the model learned to associate filename patterns")
            print("   with labels rather than analyzing image content.")
            print()
            print("💡 To fix this, you should:")
            print("   1. Randomize filenames before training")
            print("   2. Ensure train/val/test splits don't correlate with filenames")
            print("   3. Use cross-dataset validation")
    
    finally:
        # Clean up
        if 'test_dir' in locals():
            cleanup_test_files(test_dir)

if __name__ == "__main__":
    main()
