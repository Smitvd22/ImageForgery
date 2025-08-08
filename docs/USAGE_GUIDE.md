# Image Forgery Detection - Complete Usage Guide

This comprehensive guide demonstrates how to use the multi-dataset image forgery detection system with support for both 4CAM and MISD datasets.

## 🎯 Project Overview

This project implements a state-of-the-art image forgery detection system with the following key components:

### ✅ Implemented Features

1. **Multi-Dataset Support** - 4CAM (TIF) and MISD (JPG/BMP/PNG) datasets
2. **Dataset Management** - Easy switching between datasets
3. **Preprocessing Pipeline** - Advanced noise suppression and enhancement
4. **Multi-backbone Architecture** - ResNet50, EfficientNet-B0, DenseNet121
5. **Feature Fusion** - 4,502-dimensional feature vectors
6. **Ensemble Learning** - Multiple ML models with stacking
7. **Separate Results** - Dataset-specific result directories
8. **Complete Utilities** - Training, testing, validation, and prediction

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Management

```bash
# List available datasets
python dataset_manager.py list

# Switch to MISD dataset
python dataset_manager.py switch misd

# Switch to 4CAM dataset
python dataset_manager.py switch 4cam
```

### 3. Train Models

```bash
# Train on current active dataset
python train.py

# Or run complete pipeline for specific dataset
python dataset_manager.py run-pipeline misd
```

### 4. Test System

```bash
# Test current active dataset
python test.py

# Validate model performance
python validate.py
```

### 5. Make Predictions

```bash
# Single image prediction
python predict_optimized.py --image "data/4cam_auth/canong3_02_sub_01.tif"

# Batch prediction
python predict_optimized.py --directory "data/4cam_auth/"

# Specify dataset explicitly
python predict_optimized.py --image "data/Dataset/Au/img001.jpg" --dataset misd
```

## 📁 Dataset Support

### 4CAM Dataset
- **Format**: TIFF images
- **Authentic**: 183 images in `data/4cam_auth/`
- **Forged**: 180 images in `data/4cam_splc/`
- **Results**: Stored in `results_4cam/`

### MISD Dataset
- **Formats**: JPG, BMP, PNG images
- **Authentic**: 1,239 images in `data/Dataset/Au/`
- **Forged**: 606 images in `data/Dataset/Sp/`
- **Results**: Stored in `results_misd/`

## 🔧 Advanced Usage

### Dataset Manager

The `dataset_manager.py` utility provides comprehensive dataset management:

```bash
# List all available datasets and their status
python dataset_manager.py list

# Switch active dataset
python dataset_manager.py switch 4cam
python dataset_manager.py switch misd

# Run complete pipeline (train + validate + test)
python dataset_manager.py run-pipeline 4cam
python dataset_manager.py run-pipeline misd
```

### Configuration

The system uses a unified configuration in `core/config.py`:

- `ACTIVE_DATASET`: Currently active dataset ('4cam' or 'misd')
- `DATASETS`: Dataset-specific configurations
- `RESULTS_DIR`: Automatically set based on active dataset

### Model Training

Train models for specific datasets:

```bash
# Train on 4CAM dataset
python dataset_manager.py switch 4cam
python train.py

# Train on MISD dataset
python dataset_manager.py switch misd
python train.py
```

Training process:
1. Feature extraction using CNN backbones
2. Preprocessing and feature selection
3. Training multiple ML models
4. Model validation and selection
5. Results saved to dataset-specific directory

### Model Testing

Comprehensive testing with detailed metrics:

```bash
# Test current active dataset
python test.py

# The test script will:
# - Load the appropriate model for active dataset
# - Evaluate on test set
# - Generate confusion matrix
# - Save results to results_[dataset]/
```

### Model Validation

Cross-validation for robust model evaluation:

```bash
# Validate current active dataset
python validate.py

# The validation script will:
# - Perform stratified k-fold cross-validation
# - Compare multiple models
# - Generate performance plots
# - Save validation results
```

### Prediction System

The prediction system supports both datasets automatically:

```bash
# Single image prediction
python predict_optimized.py --image "path/to/image.jpg"

# Batch prediction on directory
python predict_optimized.py --directory "path/to/images/"

# Force specific dataset
python predict_optimized.py --image "image.jpg" --dataset 4cam

# Get confidence scores
python predict_optimized.py --image "image.jpg" --confidence
```

## 📊 Results and Outputs

### Result Directories

Results are automatically organized by dataset:

- `results_4cam/`: All 4CAM dataset results
- `results_misd/`: All MISD dataset results

### Generated Files

Each result directory contains:

- `model_comparison.csv`: Performance comparison of all models
- `confusion_matrix_best_model.png`: Confusion matrix visualization
- `roc_curve.png`: ROC curve analysis
- `feature_importance.png`: Feature importance plots
- `[dataset]_results.json`: Detailed results in JSON format

### Model Files

Trained models are saved with dataset prefixes:

- `models/4cam_best_model.pkl`: Best 4CAM model
- `models/4cam_scaler.pkl`: 4CAM feature scaler
- `models/4cam_feature_selector.pkl`: 4CAM feature selector
- `models/misd_best_model.pkl`: Best MISD model
- `models/misd_scaler.pkl`: MISD feature scaler
- `models/misd_feature_selector.pkl`: MISD feature selector

## 🔍 Troubleshooting

### Common Issues

1. **Dataset not found**
   ```bash
   # Verify dataset structure
   python dataset_manager.py list
   ```

2. **Model not trained**
   ```bash
   # Train model for current dataset
   python train.py
   ```

3. **Wrong dataset active**
   ```bash
   # Check current dataset
   python dataset_manager.py list
   
   # Switch if needed
   python dataset_manager.py switch [dataset_name]
   ```

### Performance Tips

1. **GPU Acceleration**: The system automatically detects and uses GPU if available
2. **Batch Processing**: Use directory prediction for multiple images
3. **Memory Management**: Large batches are processed in chunks

### Debugging

Enable verbose output by modifying `core/config.py`:

```python
VERBOSE = True
DEBUG = True
```

## 📈 Model Performance

### Expected Accuracy

- **4CAM Dataset**: ~80-90% accuracy
- **MISD Dataset**: ~85-95% accuracy

### Feature Information

- **Total Features**: 4,502 (before selection)
- **Selected Features**: 200 (after variance filtering and selection)
- **CNN Backbones**: ResNet50 (2048), EfficientNet-B0 (1280), DenseNet121 (1024)

### Model Types

The system trains multiple models:
- Extra Trees Classifier
- Random Forest
- XGBoost
- Gradient Boosting
- MLP Classifier
- Stacking Ensemble

## 🎯 Best Practices

1. **Always switch dataset before training**: Use `dataset_manager.py switch [dataset]`
2. **Check active dataset**: Use `dataset_manager.py list`
3. **Use appropriate image formats**: TIF for 4CAM, JPG/BMP/PNG for MISD
4. **Monitor results**: Check dataset-specific result directories
5. **Validate regularly**: Run validation after training new models

## 📞 Support

For issues or questions:

1. Check this usage guide
2. Verify dataset and model files exist
3. Ensure correct dataset is active
4. Check error messages in terminal output

---

**System Status**: ✅ Production-ready with multi-dataset support
