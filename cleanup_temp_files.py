#!/usr/bin/env python3
"""
Cleanup Script for Temporary Files
Removes temporary files created during the accuracy improvement process
"""

import os
import glob
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_temporary_files():
    """Remove temporary files created during the improvement process"""
    
    logger.info("Starting cleanup of temporary files...")
    
    # List of temporary files and patterns to remove
    temp_files_patterns = [
        # Temporary run scripts
        "run_all_datasets.py",  # Original problematic version
        "run_all_datasets_fixed.py",  # Keep only if needed for future runs
        "monitor_progress.py",  # Progress monitoring script
        
        # Temporary configuration files
        "core/config_fixed.py",
        
        # Log files from automation runs
        "multi_dataset_run.log",
        "*multi_dataset_results_*.json",  # Keep final results but remove temp ones
        
        # Python cache files
        "**/__pycache__",
        "**/*.pyc",
        
        # Any other temporary files
        "*.tmp",
        "*_temp.py",
        "*_backup.py"
    ]
    
    removed_files = []
    kept_files = []
    
    for pattern in temp_files_patterns:
        files = glob.glob(pattern, recursive=True)
        for file_path in files:
            try:
                if os.path.isfile(file_path):
                    # Special handling for certain files
                    if "run_all_datasets_fixed.py" in file_path:
                        # Keep this if user wants to run multi-dataset training again
                        kept_files.append(file_path)
                        logger.info(f"Keeping useful automation script: {file_path}")
                        continue
                    
                    if "multi_dataset_results_" in file_path:
                        # Keep the latest results file
                        if "20250809" in file_path:  # Today's results
                            kept_files.append(file_path)
                            logger.info(f"Keeping latest results: {file_path}")
                            continue
                    
                    os.remove(file_path)
                    removed_files.append(file_path)
                    logger.info(f"Removed: {file_path}")
                    
                elif os.path.isdir(file_path):
                    # Remove cache directories
                    if "__pycache__" in file_path:
                        import shutil
                        shutil.rmtree(file_path)
                        removed_files.append(file_path)
                        logger.info(f"Removed directory: {file_path}")
                        
            except Exception as e:
                logger.warning(f"Could not remove {file_path}: {e}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("CLEANUP SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Files removed: {len(removed_files)}")
    for file in removed_files:
        logger.info(f"  - {file}")
    
    logger.info(f"\nImportant files kept: {len(kept_files)}")
    for file in kept_files:
        logger.info(f"  - {file}")
    
    logger.info("\nCore project files (untouched):")
    core_files = [
        "train.py", "validate.py", "test.py", "predict_optimized.py",
        "core/config.py", "core/advanced_preprocessing.py", 
        "core/enhanced_features.py", "core/advanced_ensemble.py",
        "core/simple_preprocessing.py"
    ]
    
    for file in core_files:
        if os.path.exists(file):
            logger.info(f"  ✓ {file}")
        else:
            logger.warning(f"  ✗ {file} (missing)")
    
    logger.info("=" * 60)
    logger.info("Cleanup completed successfully!")
    
def main():
    cleanup_temporary_files()

if __name__ == "__main__":
    main()
