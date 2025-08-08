#!/usr/bin/env python3
"""
 Complete Image Forgery Detection Process
Runs training, validation, and testing in sequence with proper evaluation
"""

import os
import sys
import time
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_name, description):
    """Run a Python script and capture its output"""
    print(f"\n{'='*80}")
    print(f" {description}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.getcwd())
        
        execution_time = time.time() - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        # Check for errors
        if result.returncode != 0:
            print(f" ERROR in {script_name}:")
            if result.stderr:
                print(result.stderr)
            return False, execution_time
        else:
            print(f" {script_name} completed successfully in {execution_time:.2f} seconds")
            return True, execution_time
            
    except Exception as e:
        execution_time = time.time() - start_time
        print(f" Exception running {script_name}: {e}")
        return False, execution_time

def check_prerequisites():
    """Check if all required files exist"""
    required_files = [
        'train.py',
        'validate.py', 
        'test.py',
        'data/train_labels.csv',
        'data/val_labels.csv',
        'data/test_labels.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f" Missing required files: {missing_files}")
        return False
    
    logger.info(" All required files found")
    return True

def generate_summary_report(train_time, val_time, test_time):
    """Generate a comprehensive summary report"""
    
    # Load results from each stage
    results_summary = {
        'process_info': {
            'start_time': datetime.now().isoformat(),
            'total_time': train_time + val_time + test_time,
            'train_time': train_time,
            'validation_time': val_time,
            'test_time': test_time
        }
    }
    
    # Load training results
    try:
        with open('./results/train_complete_results.json', 'r') as f:
            train_results = json.load(f)
        results_summary['training'] = {
            'best_model': train_results.get('best_model', 'Unknown'),
            'best_accuracy': train_results.get('best_metrics', {}).get('accuracy', 0),
            'training_samples': train_results.get('sample_count', 0),
            'features': train_results.get('feature_count', 0)
        }
    except:
        results_summary['training'] = {'status': 'failed_to_load'}
    
    # Load validation results
    try:
        with open('./results/validation_results.json', 'r') as f:
            val_results = json.load(f)
        results_summary['validation'] = {
            'best_model': val_results.get('best_model', 'Unknown'),
            'best_accuracy': val_results.get('best_metrics', {}).get('accuracy', 0),
            'validation_samples': val_results.get('dataset_info', {}).get('validation_samples', 0),
            'overfitting_detected': val_results.get('best_metrics', {}).get('accuracy', 0) < 0.7
        }
    except:
        results_summary['validation'] = {'status': 'failed_to_load'}
    
    # Load test results
    try:
        with open('./results/test_complete_results.json', 'r') as f:
            test_results = json.load(f)
        results_summary['testing'] = {
            'best_model': test_results.get('best_model', 'Unknown'),
            'best_accuracy': test_results.get('best_metrics', {}).get('accuracy', 0),
            'test_samples': test_results.get('dataset_info', {}).get('test_samples', 0)
        }
    except:
        results_summary['testing'] = {'status': 'failed_to_load'}
    
    # Save summary
    with open('./results/complete_process_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    return results_summary

def print_final_summary(summary):
    """Print a comprehensive final summary"""
    print(f"\n{'='*80}")
    print(" COMPLETE PROCESS FINISHED!")
    print(f"{'='*80}")
    
    # Process timing
    process_info = summary.get('process_info', {})
    print(f" Total Process Time: {process_info.get('total_time', 0):.2f} seconds")
    print(f"   - Training: {process_info.get('train_time', 0):.2f}s")
    print(f"   - Validation: {process_info.get('validation_time', 0):.2f}s")
    print(f"   - Testing: {process_info.get('test_time', 0):.2f}s")
    
    # Dataset info
    train_info = summary.get('training', {})
    val_info = summary.get('validation', {})
    test_info = summary.get('testing', {})
    
    print(f"\n Dataset Summary:")
    print(f"   - Training samples: {train_info.get('training_samples', 'N/A')}")
    print(f"   - Validation samples: {val_info.get('validation_samples', 'N/A')}")
    print(f"   - Test samples: {test_info.get('test_samples', 'N/A')}")
    print(f"   - Features: {train_info.get('features', 'N/A')}")
    
    # Performance summary
    print(f"\n Performance Summary:")
    print(f"   - Training Accuracy: {train_info.get('best_accuracy', 0):.4f} ({train_info.get('best_accuracy', 0)*100:.2f}%)")
    print(f"   - Validation Accuracy: {val_info.get('best_accuracy', 0):.4f} ({val_info.get('best_accuracy', 0)*100:.2f}%)")
    print(f"   - Test Accuracy: {test_info.get('best_accuracy', 0):.4f} ({test_info.get('best_accuracy', 0)*100:.2f}%)")
    
    # Overfitting warning
    train_acc = train_info.get('best_accuracy', 0)
    val_acc = val_info.get('best_accuracy', 0)
    
    if train_acc > 0.95 and val_acc < 0.7:
        print(f"\n OVERFITTING DETECTED!")
        print(f"   Training accuracy ({train_acc:.2%}) >> Validation accuracy ({val_acc:.2%})")
        print(f"   Consider:")
        print(f"   - Reducing model complexity")
        print(f"   - Adding more training data")
        print(f"   - Increasing regularization")
        print(f"   - Using data augmentation")
    
    print(f"\n All results saved to: ./results/")
    print(f"{'='*80}")

def main():
    """Main function to run the complete process"""
    print(" IMAGE FORGERY DETECTION - COMPLETE PROCESS")
    print("Running: Train  Validate  Test")
    print(f"{'='*80}")
    
    # Check prerequisites
    if not check_prerequisites():
        return False
    
    # Ensure results directory exists
    os.makedirs('./results', exist_ok=True)
    
    total_start_time = time.time()
    
    # Step 1: Training
    logger.info("Starting training phase...")
    train_success, train_time = run_script('train.py', 'TRAINING PHASE')
    
    if not train_success:
        logger.error(" Training failed. Stopping process.")
        return False
    
    # Step 2: Validation
    logger.info("Starting validation phase...")
    val_success, val_time = run_script('validate.py', 'VALIDATION PHASE')
    
    if not val_success:
        logger.error(" Validation failed. Stopping process.")
        return False
    
    # Step 3: Testing
    logger.info("Starting testing phase...")
    test_success, test_time = run_script('test.py', 'TESTING PHASE')
    
    if not test_success:
        logger.error(" Testing failed. Stopping process.")
        return False
    
    # Generate summary report
    logger.info("Generating summary report...")
    summary = generate_summary_report(train_time, val_time, test_time)
    
    # Print final summary
    print_final_summary(summary)
    
    total_time = time.time() - total_start_time
    logger.info(f" Complete process finished in {total_time:.2f} seconds")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
