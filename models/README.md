# 📁 Models Directory

This directory contains the trained models, configurations, and evaluation results for the Image Forgery Detection System.

## 🎯 Active Models (Primary)

### Current Production Models
- `optimized_best_model.pkl` - **Primary model** (90.91% accuracy)
- `optimized_scaler.pkl` - Feature scaler for optimized model
- `optimized_config.json` - Configuration for optimized model
- `optimized_results.json` - Performance metrics for optimized model
- `optimized_all_models.pkl` - All trained models ensemble

### Fallback Models
- `best_model.pkl` - Fallback model for compatibility
- `gpu_windows_best_model.pkl` - GPU-optimized Windows model
- `gpu_windows_scaler.pkl` - Scaler for GPU Windows model
- `gpu_windows_config.json` - GPU Windows configuration
- `preprocessors.pkl` - Legacy preprocessors

## 📊 Evaluation Results

### Complete Dataset Analysis (Latest)
- `complete_dataset_metrics.json` - Key performance metrics (96.69% accuracy)
- `complete_dataset_performance.json` - Detailed model comparison
- `complete_dataset_test_results.csv` - Prediction results
- `complete_dataset_test_results.json` - JSON format results
- `comprehensive_test_results.json` - Comprehensive test analysis

### Visualizations
- `complete_dataset_confusion_matrix.png` - Confusion matrix
- `complete_dataset_roc_curve.png` - ROC curve analysis
- `complete_dataset_pr_curve.png` - Precision-Recall curve
- `complete_dataset_model_comparison.png` - Model performance comparison
- `complete_dataset_analysis.png` - Complete analysis dashboard
- `test_set_analysis.png` - Test set analysis
- `training_set_analysis.png` - Training set analysis
- `validation_set_analysis.png` - Validation set analysis

### Reports
- `COMPREHENSIVE_TEST_REPORT.md` - Detailed evaluation report

## 🚀 Usage

### Load Optimized Model (Recommended)
```python
import pickle

# Load the best performing model
with open('./models/optimized_best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('./models/optimized_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
```

### Load Fallback Model
```python
# If optimized model fails, use fallback
fallback_paths = [
    './models/gpu_windows_best_model.pkl',
    './models/best_model.pkl'
]

for path in fallback_paths:
    if os.path.exists(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        break
```

## 📈 Performance Summary

| Model Type | Accuracy | F1-Score | ROC-AUC | Status |
|------------|----------|----------|---------|---------|
| **Optimized** | **90.91%** | **0.909** | **0.989** | 🏆 **Active** |
| Complete Dataset | 96.69% | 0.967 | 0.994 | ✅ Latest |
| GPU Windows | 85.45% | 0.855 | 0.928 | 🔄 Fallback |

## 🧹 Model Management

### Keep These Files
- `optimized_*` - Current production models
- `complete_dataset_*` - Latest evaluation results
- `gpu_windows_*` - Fallback models

### Safe to Remove
- Individual classifier files (`et_*.pkl`, `rf_*.pkl`, etc.)
- Large ensemble files (`*_all_models.pkl`) if space is needed
- Old version results (`enhanced_v2_*`, `improved_v3_*`)

---

**💡 Tip**: Use `predict_optimized.py` which automatically loads the best available model with fallback support.
