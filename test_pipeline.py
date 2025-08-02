#!/usr/bin/env python3
"""
Simple Pipeline Test Script
Tests the complete pipeline on multiple images to ensure everything works
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Add current directory to path
sys.path.append('.')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_prediction_pipeline():
    """Test the prediction pipeline on multiple images"""
    print("🧪 Testing Complete Pipeline")
    print("=" * 60)
    
    # Import prediction function
    from predict import predict_image
    
    # Test authentic images
    auth_dir = Path("data/4cam_auth")
    auth_files = list(auth_dir.glob("*.tif"))[:3]  # Test first 3 authentic images
    
    print("\n📊 Testing Authentic Images:")
    auth_correct = 0
    for img_path in auth_files:
        try:
            result = predict_image(str(img_path))
            if result:
                is_correct = result['label'] == 'Authentic'
                if is_correct:
                    auth_correct += 1
                print(f"✓ {img_path.name}: {result['label']} ({result['confidence']:.1%}) {'✅' if is_correct else '❌'}")
            else:
                print(f"❌ {img_path.name}: Failed to predict")
        except Exception as e:
            print(f"❌ {img_path.name}: Error - {e}")
    
    # Test forged images
    splc_dir = Path("data/4cam_splc")
    splc_files = list(splc_dir.glob("*.tif"))[:3]  # Test first 3 forged images
    
    print("\n📊 Testing Forged Images:")
    splc_correct = 0
    for img_path in splc_files:
        try:
            result = predict_image(str(img_path))
            if result:
                is_correct = result['label'] == 'Forged'
                if is_correct:
                    splc_correct += 1
                print(f"✓ {img_path.name}: {result['label']} ({result['confidence']:.1%}) {'✅' if is_correct else '❌'}")
            else:
                print(f"❌ {img_path.name}: Failed to predict")
        except Exception as e:
            print(f"❌ {img_path.name}: Error - {e}")
    
    # Summary
    total_correct = auth_correct + splc_correct
    total_tested = len(auth_files) + len(splc_files)
    accuracy = total_correct / total_tested if total_tested > 0 else 0
    
    print("\n" + "=" * 60)
    print("📊 PIPELINE TEST SUMMARY")
    print("=" * 60)
    print(f"Authentic Images: {auth_correct}/{len(auth_files)} correct")
    print(f"Forged Images: {splc_correct}/{len(splc_files)} correct")
    print(f"Overall Accuracy: {accuracy:.1%} ({total_correct}/{total_tested})")
    
    if accuracy >= 0.6:  # 60% threshold for basic functionality
        print("✅ Pipeline test PASSED")
        return True
    else:
        print("❌ Pipeline test FAILED")
        return False

def test_model_loading():
    """Test that all models can be loaded"""
    print("\n🔧 Testing Model Loading")
    print("-" * 40)
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory not found")
        return False
    
    model_files = list(models_dir.glob("*.pkl"))
    if len(model_files) == 0:
        print("❌ No model files found")
        return False
    
    print(f"✅ Found {len(model_files)} model files")
    
    # Try loading the best model
    import joblib
    try:
        best_model = joblib.load("models/best_model.pkl")
        preprocessors = joblib.load("models/preprocessors.pkl")
        print("✅ Best model and preprocessors loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def test_data_availability():
    """Test that data is available"""
    print("\n📁 Testing Data Availability")
    print("-" * 40)
    
    # Check directories
    auth_dir = Path("data/4cam_auth")
    splc_dir = Path("data/4cam_splc")
    
    if not auth_dir.exists():
        print("❌ Authentic images directory not found")
        return False
    
    if not splc_dir.exists():
        print("❌ Forged images directory not found")
        return False
    
    auth_count = len(list(auth_dir.glob("*.tif")))
    splc_count = len(list(splc_dir.glob("*.tif")))
    
    print(f"✅ Authentic images: {auth_count}")
    print(f"✅ Forged images: {splc_count}")
    
    if auth_count > 0 and splc_count > 0:
        print("✅ Data availability test PASSED")
        return True
    else:
        print("❌ Data availability test FAILED")
        return False

def main():
    """Main test function"""
    print("🎯 Image Forgery Detection - Complete Pipeline Test")
    print("=" * 80)
    
    # Run all tests
    tests = [
        ("Data Availability", test_data_availability),
        ("Model Loading", test_model_loading),
        ("Prediction Pipeline", test_prediction_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎯 FINAL TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
        if result:
            passed += 1
    
    success_rate = passed / len(results) if results else 0
    print("-" * 80)
    print(f"Tests Passed: {passed}/{len(results)} ({success_rate:.1%})")
    
    if success_rate >= 1.0:
        print("🎉 ALL TESTS PASSED! Pipeline is fully functional.")
    elif success_rate >= 0.75:
        print("✅ Most tests passed. Pipeline is mostly functional.")
    else:
        print("⚠️ Several tests failed. Please check the issues above.")
    
    return success_rate >= 0.75

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
