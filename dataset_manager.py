#!/usr/bin/env python3
"""
Dataset Switching Utility
Easily switch between different datasets and manage their training/testing
"""

import os
import sys
import shutil
from pathlib import Path

def switch_dataset(target_dataset):
    """Switch to a different dataset by modifying the config file"""
    
    config_file = "core/config.py"
    if not os.path.exists(config_file):
        print(f"Error: Config file not found: {config_file}")
        return False
    
    # Read current config
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Available datasets
    valid_datasets = ["4cam", "misd"]
    
    if target_dataset not in valid_datasets:
        print(f"Error: Invalid dataset '{target_dataset}'. Valid options: {valid_datasets}")
        return False
    
    # Replace the ACTIVE_DATASET line
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('ACTIVE_DATASET = '):
            lines[i] = f'ACTIVE_DATASET = "{target_dataset}"  # Options: "4cam", "misd"'
            break
    
    # Write back to file
    new_content = '\n'.join(lines)
    with open(config_file, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Successfully switched to {target_dataset.upper()} dataset")
    print(f"📁 Configuration updated in {config_file}")
    
    return True

def show_current_dataset():
    """Show current dataset configuration"""
    try:
        # Import after potential config change
        sys.path.append('.')
        from core.config import ACTIVE_DATASET, print_dataset_info
        print_dataset_info()
        return ACTIVE_DATASET
    except Exception as e:
        print(f"Error reading dataset configuration: {e}")
        return None

def list_available_datasets():
    """List all available datasets"""
    try:
        sys.path.append('.')
        from core.config import DATASETS
        
        print("="*60)
        print("AVAILABLE DATASETS")
        print("="*60)
        
        for key, dataset in DATASETS.items():
            status = "✅ Available" if (os.path.exists(dataset["authentic_dir"]) and 
                                     os.path.exists(dataset["forged_dir"])) else "❌ Not Found"
            
            print(f"\n{key.upper()} Dataset:")
            print(f"  Name: {dataset['name']}")
            print(f"  Description: {dataset['description']}")
            print(f"  Authentic Dir: {dataset['authentic_dir']}")
            print(f"  Forged Dir: {dataset['forged_dir']}")
            print(f"  Results Dir: {dataset['results_dir']}")
            print(f"  Status: {status}")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error listing datasets: {e}")

def run_pipeline(phases=None):
    """Run the training/validation/testing pipeline"""
    if phases is None:
        phases = ["train", "validate", "test"]
    
    python_cmd = sys.executable
    
    for phase in phases:
        script = f"{phase}.py"
        if not os.path.exists(script):
            print(f"❌ Script not found: {script}")
            continue
            
        print(f"\n{'='*60}")
        print(f"RUNNING {phase.upper()} PHASE")
        print(f"{'='*60}")
        
        result = os.system(f'"{python_cmd}" {script}')
        
        if result != 0:
            print(f"❌ {phase.upper()} phase failed with exit code {result}")
            return False
        else:
            print(f"✅ {phase.upper()} phase completed successfully")
    
    return True

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset Management Utility")
    parser.add_argument("--switch", choices=["4cam", "misd"], 
                       help="Switch to specified dataset")
    parser.add_argument("--show", action="store_true",
                       help="Show current dataset configuration")
    parser.add_argument("--list", action="store_true",
                       help="List all available datasets")
    parser.add_argument("--run", nargs="*", 
                       choices=["train", "validate", "test"],
                       help="Run specified phases (default: all)")
    parser.add_argument("--full-test", action="store_true",
                       help="Test both datasets (switch and run all phases for each)")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Handle different commands
    if args.switch:
        success = switch_dataset(args.switch)
        if success:
            show_current_dataset()
    
    elif args.show:
        show_current_dataset()
    
    elif args.list:
        list_available_datasets()
    
    elif args.run is not None:
        phases = args.run if args.run else ["train", "validate", "test"]
        current = show_current_dataset()
        if current:
            print(f"\n🚀 Running pipeline for {current.upper()} dataset...")
            run_pipeline(phases)
    
    elif args.full_test:
        print("🔄 Running full test on both datasets...")
        
        datasets = ["4cam", "misd"]
        for dataset in datasets:
            print(f"\n{'='*80}")
            print(f"TESTING {dataset.upper()} DATASET")
            print(f"{'='*80}")
            
            # Switch dataset
            if switch_dataset(dataset):
                # Run full pipeline
                success = run_pipeline(["train", "validate", "test"])
                if not success:
                    print(f"❌ Pipeline failed for {dataset.upper()} dataset")
                else:
                    print(f"✅ Pipeline completed successfully for {dataset.upper()} dataset")
            else:
                print(f"❌ Failed to switch to {dataset.upper()} dataset")

if __name__ == "__main__":
    main()
