#!/usr/bin/env python3
"""
Setup and Test Script for Image Forgery Detection System
Validates environment and runs basic tests
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    logger.info("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        logger.error(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Need Python 3.8+")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'torch', 'torchvision', 'timm', 'sklearn', 'xgboost', 
        'numpy', 'pandas', 'cv2', 'PIL', 'tqdm', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                import PIL
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            logger.info(f"✅ {package} is installed")
        except ImportError:
            logger.warning(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Install missing packages with: pip install -r requirements.txt")
        return False
    else:
        logger.info("✅ All dependencies are installed")
        return True

def check_data_structure():
    """Check if data directory structure is correct"""
    logger.info("Checking data structure...")
    
    required_paths = [
        'data/4cam_auth',
        'data/4cam_splc', 
        'data/train_labels.csv',
        'data/val_labels.csv',
        'data/test_labels.csv'
    ]
    
    for path in required_paths:
        if os.path.exists(path):
            logger.info(f"✅ {path} exists")
        else:
            logger.warning(f"❌ {path} is missing")
    
    # Count images
    auth_count = len(list(Path('data/4cam_auth').glob('*.tif'))) if os.path.exists('data/4cam_auth') else 0
    splc_count = len(list(Path('data/4cam_splc').glob('*.tif'))) if os.path.exists('data/4cam_splc') else 0
    
    logger.info(f"Authentic images: {auth_count}")
    logger.info(f"Forged images: {splc_count}")
    
    return auth_count > 0 and splc_count > 0

def check_gpu_availability():
    """Check if GPU is available"""
    logger.info("Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            logger.info("⚠️ No GPU available, will use CPU")
            return False
    except Exception as e:
        logger.error(f"❌ Error checking GPU: {e}")
        return False

def test_core_imports():
    """Test if core modules can be imported"""
    logger.info("Testing core module imports...")
    
    try:
        import core.config as config
        logger.info("✅ core.config imported successfully")
        
        from core.dataset import get_data_loaders
        logger.info("✅ core.dataset imported successfully")
        
        import core.models
        logger.info("✅ core.models imported successfully")
        
        import core.classifier
        logger.info("✅ core.classifier imported successfully")
        
        import utils.metrics
        logger.info("✅ utils.metrics imported successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import error: {e}")
        return False

def run_basic_test():
    """Run a basic functionality test"""
    logger.info("Running basic functionality test...")
    
    try:
        # Test image loading with a real image from dataset
        import cv2
        import numpy as np
        from pathlib import Path
        
        # Try to find a sample image
        auth_dir = Path("data/4cam_auth")
        if auth_dir.exists():
            sample_images = list(auth_dir.glob("*.tif"))
            if sample_images:
                sample_path = str(sample_images[0])
                logger.info(f"Testing with sample image: {sample_path}")
                
                # Test preprocessing
                from core.preprocessing import preprocess_image
                processed = preprocess_image(sample_path)
                
                logger.info("✅ Basic image processing test passed")
                return True
            else:
                logger.warning("No sample images found in data/4cam_auth")
                return False
        else:
            logger.warning("Data directory not found")
            return False
        
    except Exception as e:
        logger.error(f"❌ Basic test failed: {e}")
        return False

def main():
    """Main setup and test function"""
    logger.info("🚀 Image Forgery Detection System - Setup & Test")
    logger.info("=" * 60)
    
    tests = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Data Structure", check_data_structure),
        ("GPU Availability", check_gpu_availability),
        ("Core Imports", test_core_imports),
        ("Basic Functionality", run_basic_test)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Testing: {test_name}")
        results[test_name] = test_func()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 SETUP TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:20s}: {status}")
        if result:
            passed += 1
    
    logger.info("=" * 60)
    logger.info(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready.")
        logger.info("\nNext steps:")
        logger.info("1. Run training: python train.py")
        logger.info("2. Make predictions: python predict.py <image_path>")
    else:
        logger.warning("⚠️ Some tests failed. Please fix the issues before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
