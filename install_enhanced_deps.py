#!/usr/bin/env python3
"""
Install Missing Dependencies for Enhanced Image Forgery Detection
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    """Install all required packages for enhanced features"""
    print("🚀 Installing enhanced dependencies for 3%+ accuracy improvement...")
    
    # Required packages for enhanced features
    packages = [
        "scikit-image>=0.19.0",
        "pywavelets>=1.3.0",
        "opencv-python>=4.5.0",
        "scipy>=1.8.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "pillow>=8.3.0",
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "timm>=0.6.0",
        "xgboost>=1.6.0"
    ]
    
    print(f"📦 Installing {len(packages)} packages...")
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"✅ Successfully installed: {success_count}/{len(packages)} packages")
    
    if success_count == len(packages):
        print("🎉 All dependencies installed successfully!")
        print("💡 You can now run training with enhanced features for 3%+ accuracy improvement")
    else:
        print("⚠️ Some packages failed to install. The system will use fallback methods.")
    
    # Test imports
    print("\n🧪 Testing imports...")
    test_imports()

def test_imports():
    """Test if key packages can be imported"""
    test_packages = [
        ("skimage", "scikit-image"),
        ("pywt", "PyWavelets"),
        ("cv2", "OpenCV"),
        ("torch", "PyTorch"),
        ("timm", "TIMM"),
        ("xgboost", "XGBoost")
    ]
    
    for module, name in test_packages:
        try:
            __import__(module)
            print(f"✅ {name} import successful")
        except ImportError:
            print(f"❌ {name} import failed")

if __name__ == "__main__":
    main()
