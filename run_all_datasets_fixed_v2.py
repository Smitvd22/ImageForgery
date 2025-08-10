#!/usr/bin/env python3
"""
Enhanced Multi-Dataset Training Automation Script
Runs training, validation, and testing on all three datasets
"""

import subprocess
import sys
import os
import json
import logging
import shutil
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_dataset_run.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MultiDatasetRunner:
    """Orchestrates training, validation, and testing across all datasets"""
    
    def __init__(self):
        self.datasets = ["MISD", "4CAM", "IMSLICE"]
        self.results = {}
    
    def set_dataset(self, dataset_name):
        """Update config.py to use the specified dataset"""
        logger.info(f"Setting dataset to {dataset_name}")
        
        # Use the config function instead of overwriting the file
        try:
            # Import and use the set_active_dataset function
            sys.path.append('.')
            from core.config import set_active_dataset
            result = set_active_dataset(dataset_name)
            logger.info(result)
        except Exception as e:
            logger.error(f"Failed to set dataset: {e}")
            # Fallback: copy the complete config and update only the current dataset
            shutil.copy("core/config_minimal.py", "core/config.py")
            
            # Read the config file and update only the CURRENT_DATASET line
            with open("core/config.py", 'r') as f:
                content = f.read()
            
            # Replace CURRENT_DATASET line
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('CURRENT_DATASET = '):
                    lines[i] = f'CURRENT_DATASET = "{dataset_name}"'
                    break
            
            with open("core/config.py", 'w') as f:
                f.write('\n'.join(lines))
        
    def run_command(self, command, description, timeout=7200):  # 2 hour timeout
        """Run a Python command with timeout and error handling"""
        logger.info(f"Starting: {description}")
        start_time = datetime.now()
        
        try:
            result = subprocess.run(
                [sys.executable, command],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd(),
                encoding='utf-8',
                errors='replace'
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if result.returncode == 0:
                logger.info(f"✅ {description} completed successfully in {duration:.1f}s")
                return True, result.stdout, result.stderr
            else:
                error_output = result.stderr if result.stderr else result.stdout
                logger.error(f"❌ {description} failed: {error_output}")
                return False, result.stdout, result.stderr
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏱️ {description} timed out after {timeout}s")
            return False, "", "Timeout"
        except Exception as e:
            logger.error(f"💥 {description} crashed: {str(e)}")
            return False, "", str(e)
    
    def run_dataset_pipeline(self, dataset):
        """Run complete pipeline for a single dataset"""
        logger.info("=" * 60)
        logger.info(f"RUNNING ENHANCED PIPELINE FOR {dataset}")
        logger.info("=" * 60)
        
        # Set the dataset
        self.set_dataset(dataset)
        
        dataset_results = {
            "dataset": dataset,
            "start_time": datetime.now().isoformat(),
            "training": {"success": False, "duration": 0},
            "validation": {"success": False, "duration": 0},
            "testing": {"success": False, "duration": 0}
        }
        
        # Run Training
        start_time = datetime.now()
        success, stdout, stderr = self.run_command("train.py", f"Enhanced Training on {dataset}")
        end_time = datetime.now()
        
        dataset_results["training"] = {
            "success": success,
            "duration": (end_time - start_time).total_seconds(),
            "stdout": stdout[:1000] if stdout else "",  # Limit output size
            "stderr": stderr[:1000] if stderr else ""
        }
        
        if not success:
            logger.error(f"Training failed for {dataset}")
        
        # Run Validation
        start_time = datetime.now()
        success, stdout, stderr = self.run_command("validate.py", f"Enhanced Validation on {dataset}")
        end_time = datetime.now()
        
        dataset_results["validation"] = {
            "success": success,
            "duration": (end_time - start_time).total_seconds(),
            "stdout": stdout[:1000] if stdout else "",
            "stderr": stderr[:1000] if stderr else ""
        }
        
        if not success:
            logger.error(f"Validation failed for {dataset}")
        
        # Run Testing
        start_time = datetime.now()
        success, stdout, stderr = self.run_command("test.py", f"Enhanced Testing on {dataset}")
        end_time = datetime.now()
        
        dataset_results["testing"] = {
            "success": success,
            "duration": (end_time - start_time).total_seconds(),
            "stdout": stdout[:1000] if stdout else "",
            "stderr": stderr[:1000] if stderr else ""
        }
        
        if not success:
            logger.error(f"Testing failed for {dataset}")
        
        dataset_results["end_time"] = datetime.now().isoformat()
        logger.info(f"{dataset} pipeline completed")
        
        return dataset_results
    
    def run_all_datasets(self):
        """Run the complete pipeline for all datasets"""
        logger.info("STARTING ENHANCED MULTI-DATASET TRAINING")
        logger.info(f"Start Time: {datetime.now()}")
        logger.info("=" * 60)
        
        overall_start = datetime.now()
        
        for dataset in self.datasets:
            try:
                dataset_results = self.run_dataset_pipeline(dataset)
                self.results[dataset] = dataset_results
                
                # Save intermediate results
                self.save_results()
                
            except Exception as e:
                logger.error(f"Pipeline failed for {dataset}: {str(e)}")
                self.results[dataset] = {
                    "dataset": dataset,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        overall_end = datetime.now()
        
        # Final summary
        logger.info("=" * 60)
        logger.info("MULTI-DATASET PIPELINE COMPLETE")
        logger.info(f"Total Duration: {(overall_end - overall_start).total_seconds():.1f}s")
        logger.info("=" * 60)
        
        # Print summary
        self.print_summary()
        
        return self.results
    
    def save_results(self):
        """Save current results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_dataset_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"📊 Results saved to {filename}")
    
    def print_summary(self):
        """Print a summary of all results"""
        logger.info("📋 PIPELINE SUMMARY:")
        logger.info("-" * 40)
        
        for dataset, results in self.results.items():
            if "error" in results:
                logger.info(f"{dataset}: ❌ FAILED - {results['error']}")
                continue
                
            training_status = "✅" if results["training"]["success"] else "❌"
            validation_status = "✅" if results["validation"]["success"] else "❌"
            testing_status = "✅" if results["testing"]["success"] else "❌"
            
            logger.info(f"{dataset}:")
            logger.info(f"  Training: {training_status} ({results['training']['duration']:.1f}s)")
            logger.info(f"  Validation: {validation_status} ({results['validation']['duration']:.1f}s)")
            logger.info(f"  Testing: {testing_status} ({results['testing']['duration']:.1f}s)")

def main():
    """Main execution function"""
    try:
        runner = MultiDatasetRunner()
        results = runner.run_all_datasets()
        
        # Run analysis if available
        if os.path.exists("analyze_improvements.py"):
            logger.info("🔍 Running improvement analysis...")
            runner.run_command("analyze_improvements.py", "Final Analysis")
        
        # Run cleanup if available
        if os.path.exists("cleanup_temp_files.py"):
            logger.info("🧹 Running cleanup...")
            runner.run_command("cleanup_temp_files.py", "Cleanup")
        
        logger.info("🎉 All operations completed!")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
