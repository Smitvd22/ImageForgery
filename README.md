# 🚀 Image Forgery Detection System

A high-performance image forgery detection system with **90.91% accuracy** using GPU-accelerated deep learning and ensemble methods.

## 🎯 Key Features

- **🚀 GPU Acceleration**: Automatic GPU detection with CPU fallback
- **🎯 High Accuracy**: 90.91% accuracy on test dataset
- **🧠 Advanced ML**: Ensemble of CNN + traditional ML models
- **⚡ Fast Processing**: ~15-16 images/second on GPU
- **🔧 Easy Setup**: One-command installation and training
- **🔊 Comprehensive Noise Handling**: Advanced detection and suppression of multiple noise types
- **🎨 Adaptive Preprocessing**: Edge-preserving enhancement and artifact removal

## 📊 Performance Results

- **Best Model**: Extra Trees Ensemble
- **Test Accuracy**: 90.91%
- **F1-Score**: 0.9090
- **Precision**: 0.9096
- **Recall**: 0.9091

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Check and install dependencies
python setup.py

# Or manually install
pip install -r requirements.txt
```

### 2. Train Model
```bash
# GPU-optimized training (automatic CPU fallback)
python train_optimized.py
```

### 3. Make Predictions
```bash
# Single image
python predict_optimized.py data/4cam_auth/canong3_02_sub_01.tif

# Batch processing
python predict_optimized.py data/4cam_auth/ --output results.json

# Complete testing
python test_optimized_complete.py

# Test noise suppression capabilities
python demo_noise_suppression.py data/4cam_auth/canong3_02_sub_01.tif
```

## 📊 Model Performance

| Model | Test Accuracy | AUC | F1-Score |
|-------|---------------|-----|----------|
| **Extra Trees (et_1)** | **80.0%** | 0.922 | 0.756 |
| Extra Trees (et_2) | 78.2% | 0.933 | 0.750 |
| Random Forest (rf_2) | 72.7% | 0.931 | 0.615 |
| Stacking Ensemble | 70.9% | 0.913 | 0.579 |

## 🏗️ Architecture

### Multi-Scale Feature Extraction
- **ResNet50**: Robust deep features (2048-dim)
- **EfficientNet-B0**: Efficient features (1280-dim) 
- **DenseNet121**: Dense features (1024-dim)
- **Total**: 4,502 dimensional feature vectors

### Advanced Ensemble
- **10 Base Models**: XGBoost, Random Forest, Extra Trees, Gradient Boosting, MLP
- **Stacking Ensemble**: Meta-learner for optimal combination
- **Cross-Validation**: Stratified 10-fold validation

### Preprocessing Pipeline
- **Comprehensive Noise Suppression**: Gaussian, salt-pepper, Poisson, speckle, uniform noise detection and removal
- **Adaptive Filtering**: Wiener, bilateral, non-local means, and median filtering
- **Edge-Preserving Enhancement**: CLAHE, adaptive gamma correction, local contrast normalization
- **Sparkle Noise Suppression**: Custom morphological filtering for sensor artifacts
- Variance filtering: 4,502 → 1,465 features
- Feature selection: Top 200 most informative features
- Standardization and scaling

## 📁 Project Structure

```
ImageForgery/
├── 🚀 train_optimized.py        # GPU-optimized training (90.91% accuracy)
├── 🔮 predict_optimized.py      # GPU-optimized prediction
├── 🧪 test_optimized_complete.py # Complete testing suite
├── 🛠️ setup.py                 # Environment setup script
├── 📋 requirements.txt          # Dependencies
├── 📋 requirements_gpu.txt      # GPU-specific dependencies
├── 🚫 .gitignore              # Git ignore rules
│
├── core/                      # Core modules
│   ├── config.py              # Configuration settings
│   ├── models.py              # CNN model architectures
│   ├── dataset.py             # Data loading utilities
│   ├── classifier.py          # ML classifiers
│   ├── preprocessing.py       # Image preprocessing
│   └── __init__.py
│
├── data/                      # Dataset
│   ├── train_labels.csv       # Training labels
│   ├── val_labels.csv         # Validation labels 
│   ├── test_labels.csv        # Test labels
│   ├── 4cam_auth/             # Authentic images (183)
│   └── 4cam_splc/             # Forged images (180)
│
├── models/                    # Trained models & results
│   ├── optimized_best_model.pkl    # Primary model (90.91%)
│   ├── optimized_scaler.pkl        # Feature scaler
│   ├── complete_dataset_*.json/png # Latest evaluation results
│   └── README.md                   # Model documentation
│
├── utils/                     # Utilities
│   ├── evaluate.py            # Model evaluation functions
│   ├── test_system.py         # System validation
│   └── __init__.py
│
└── docs/                      # Documentation
    ├── USAGE_GUIDE.md         # Detailed usage guide
    └── README.md              # Additional documentation
```

## 🔧 Technical Details

### Requirements
- **Python**: 3.8+ (tested on 3.13.1)
- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting
- **OpenCV**: Image processing
- **timm**: Pretrained models

### Hardware
- **CPU**: Works on any modern CPU
- **GPU**: Optional (CUDA support)
- **Memory**: ~4GB RAM recommended
- **Storage**: ~2GB for dataset + models

### Dataset
- **Format**: 4CAM dataset (camera forensics)
- **Images**: 363 total (183 authentic, 180 forged)
- **Resolution**: Variable (resized to 384x384)
- **Formats**: TIFF images

## 📈 Training Process

1. **Feature Extraction** (~7 minutes)
   - Load pretrained CNNs (ResNet50, EfficientNet, DenseNet)
   - Extract deep features from all images
   - Generate 4,502-dimensional vectors

2. **Preprocessing** (~1 minute)
   - Variance filtering removes low-variance features
   - Feature selection keeps top 200 features
   - Standardization and scaling

3. **Model Training** (~7 minutes)
   - Train 10 diverse base models
   - Cross-validation for robust evaluation
   - Create stacking ensemble

4. **Evaluation** (~1 minute)
   - Test on held-out test set
   - Generate comprehensive metrics
   - Save models and results

## 🎯 Usage Examples

### Basic Training
```python
from core.config import *
# Configuration automatically loaded
# Just run: python train.py
```

### Custom Prediction
```python
import joblib
from core.preprocessing import preprocess_image

# Load trained model
model = joblib.load('models/et_1_model.pkl')

# Predict image
features = preprocess_image('path/to/image.jpg')
prediction = model.predict([features])[0]
print(f"Prediction: {'Forged' if prediction else 'Authentic'}")
```

### Batch Evaluation
```python
from utils.evaluate import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.evaluate_all_models()
print(f"Best accuracy: {max(results.values()):.1%}")
```

## 🔬 Research & Development

### Key Innovations
1. **Multi-backbone CNN**: Combines 3 SOTA architectures
2. **Advanced Ensembling**: 10 models + stacking
3. **Efficient Preprocessing**: Smart feature selection
4. **Robust Validation**: Stratified cross-validation

### Future Improvements
- [ ] Add more CNN backbones (Vision Transformers)
- [ ] Implement attention mechanisms  
- [ ] Add adversarial training
- [ ] Support video forgery detection

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{image_forgery_detection_2025,
  title={Advanced Image Forgery Detection with Multi-Scale Ensemble Learning},
  author={Your Name},
  year={2025},
  note={Clean workspace achieving 80%+ accuracy}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For questions or issues:
- Check the `docs/USAGE_GUIDE.md` for detailed instructions
- Run `python setup_test.py` to validate your setup
- Create an issue on GitHub for bugs

---

**Status**: ✅ Production-ready system with 80%+ accuracy
