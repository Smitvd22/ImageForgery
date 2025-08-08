#!/usr/bin/env python3
"""
Enhanced Configuration for Maximum Accuracy
Implementing state-of-the-art ImageNet models and advanced techniques
Target: 95%+ accuracy
"""

import os
import torch
import warnings
warnings.filterwarnings('ignore')

# GPU Configuration and Detection
GPU_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if GPU_AVAILABLE else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "No GPU"
GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1e9 if GPU_AVAILABLE else 0

# GPU Optimization Settings
if GPU_AVAILABLE:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

# Dataset Selection Configuration
# Change this to switch between datasets: "4cam" or "misd"
ACTIVE_DATASET = "misd"  # Options: "4cam", "misd"

# Dataset Configuration
DATA_ROOT = "./data"

# Dataset configurations
DATASETS = {
    "4cam": {
        "name": "4CAM Dataset",
        "authentic_dir": os.path.join(DATA_ROOT, "4cam_auth"),
        "forged_dir": os.path.join(DATA_ROOT, "4cam_splc"),
        "file_extensions": ["*.tif"],
        "description": "Original 4CAM camera dataset with TIF images",
        "results_dir": "./results_4cam"
    },
    "misd": {
        "name": "MISD Dataset", 
        "authentic_dir": os.path.join(DATA_ROOT, "Dataset", "Au"),
        "forged_dir": os.path.join(DATA_ROOT, "Dataset", "Sp"),
        "file_extensions": ["*.jpg", "*.JPG", "*.bmp", "*.png"],
        "description": "Multiple Image Splicing Dataset with JPG/BMP images",
        "results_dir": "./results_misd"
    }
}

# Active dataset paths (automatically set based on ACTIVE_DATASET)
CURRENT_DATASET = DATASETS[ACTIVE_DATASET]
AUTHENTIC_DIR = CURRENT_DATASET["authentic_dir"]
FORGED_DIR = CURRENT_DATASET["forged_dir"]

# Generated CSV files for training/testing (dataset-specific)
DATASET_PREFIX = f"{ACTIVE_DATASET}_"
DATA_CSV = os.path.join(DATA_ROOT, f"{DATASET_PREFIX}labels.csv")
TRAIN_CSV = os.path.join(DATA_ROOT, f"{DATASET_PREFIX}train_labels.csv")
VAL_CSV = os.path.join(DATA_ROOT, f"{DATASET_PREFIX}val_labels.csv")
TEST_CSV = os.path.join(DATA_ROOT, f"{DATASET_PREFIX}test_labels.csv")

# Model and results paths (dataset-specific)
MODELS_DIR = "./models"
RESULTS_DIR = f"./results_{ACTIVE_DATASET}"  # Dataset-specific results directory
MODEL_PREFIX = f"{ACTIVE_DATASET}_"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, f"{MODEL_PREFIX}best_model.pkl")
ALL_MODELS_PATH = os.path.join(MODELS_DIR, f"{MODEL_PREFIX}all_models.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, f"{MODEL_PREFIX}scaler.pkl")
FEATURE_SELECTOR_PATH = os.path.join(MODELS_DIR, f"{MODEL_PREFIX}feature_selector.pkl")

# Enhanced Model Configuration for Maximum Performance
BATCH_SIZE = 8  # Optimized for stability
IMAGE_SIZE = (384, 384)  # Standardized resolution for compatibility
NUM_EPOCHS = 150  # More epochs for better convergence
LEARNING_RATE = 0.00005  # Very low learning rate for stability

# Training Configuration
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Maximum Performance Architecture Configuration
USE_IMPROVED_MODELS = True
USE_RESNET_PLUS_PLUS = True
USE_UNET = True
USE_UNET_R = True
USE_COMPREHENSIVE_BACKBONE = True
USE_HUGGINGFACE = True
USE_TIMM_MODELS = True  # Enable TIMM models for state-of-the-art architectures
USE_ENSEMBLE = True     # Enable ensemble methods
USE_ADVANCED_FUSION = True  # Advanced feature fusion techniques

# State-of-the-Art ImageNet Models Configuration - Enhanced for Forgery Detection
IMAGENET_MODELS = {
    'resnet152': {
        'enabled': True,
        'pretrained': True,
        'feature_dim': 2048,
        'input_size': (384, 384),
        'forgery_specific': True  # Add forgery-specific layers
    },
    'efficientnet_b7': {
        'enabled': True, 
        'pretrained': True,
        'feature_dim': 2560,
        'input_size': (384, 384),
        'forgery_specific': True
    },
    'convnext_base': {
        'enabled': True,
        'pretrained': True,
        'feature_dim': 1024,
        'input_size': (384, 384),
        'forgery_specific': True
    },
    # Add specialized forgery detection models
    'swin_base_patch4_window7_224': {
        'enabled': True,
        'pretrained': True,
        'feature_dim': 1024,
        'input_size': (384, 384),
        'forgery_specific': True
    },
    'vit_base_patch16_224': {
        'enabled': True,
        'pretrained': True,
        'feature_dim': 768,
        'input_size': (384, 384),
        'forgery_specific': True
    }
}

# HuggingFace Models for Additional Features
HUGGINGFACE_MODELS = [
    "microsoft/resnet-50"
]

# Forgery-Specific Detection Configuration
FORGERY_DETECTION_CONFIG = {
    'use_edge_analysis': True,
    'use_frequency_analysis': True,
    'use_compression_artifacts': True,
    'use_noise_analysis': True,
    'use_texture_analysis': True,
    'use_color_inconsistency': True,
    'use_lighting_analysis': True,
    'use_shadow_analysis': True,
    'use_reflection_analysis': True,
    'use_perspective_analysis': True,
    'patch_size': 64,  # For patch-based analysis
    'overlap_ratio': 0.5,
    'multi_scale_analysis': True,
    'scales': [1.0, 0.75, 0.5, 0.25]
}

# Comprehensive Noise Suppression Configuration
NOISE_SUPPRESSION_CONFIG = {
    'enable_comprehensive_suppression': True,
    'preserve_edges': True,
    'gaussian_noise_threshold': 0.3,
    'salt_pepper_threshold': 0.2,
    'poisson_noise_threshold': 0.4,
    'speckle_noise_threshold': 0.3,
    'uniform_noise_threshold': 0.6,
    'adaptive_filtering': True,
    'noise_detection_enabled': True,
    'bilateral_filter_params': {
        'd': 5,
        'sigma_color': 20,
        'sigma_space': 20
    },
    'gaussian_filter_params': {
        'kernel_size': (3, 3),
        'sigma': 0.5
    },
    'median_filter_kernel_size': 3,
    'wiener_filter_noise_var': 0.1,
    'nlm_filter_params': {
        'h': 10,
        'template_window_size': 7,
        'search_window_size': 21
    }
}

# Enhanced Batch Size Configuration
if GPU_AVAILABLE:
    GPU_BATCH_SIZE = 16  # Larger batch size for GPU
    BATCH_SIZE = GPU_BATCH_SIZE
    FORGERY_BATCH_SIZE = BATCH_SIZE * 2
    # GPU-optimized dataloader settings
    NUM_WORKERS = 8
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
else:
    BATCH_SIZE = 8  # Conservative for CPU
    FORGERY_BATCH_SIZE = BATCH_SIZE
    NUM_WORKERS = 4
    PIN_MEMORY = False
    PERSISTENT_WORKERS = False

# Advanced Feature Extraction & Fusion Configuration
FEATURE_FUSION_STRATEGY = "ultra_advanced_fusion"
STATISTICAL_FEATURES = True
FEATURE_NORMALIZATION = True
ENHANCED_STATISTICS = True
USE_POLYNOMIAL_FEATURES = True
USE_FEATURE_SELECTION = True
FEATURE_SELECTION_METHOD = "mutual_info"

# Ultra-Advanced Augmentation Configuration
USE_AUGMENTATION = True
AUGMENTATION_PROBABILITY = 0.9
ADVANCED_AUGMENTATION = True
USE_MIXUP = True
USE_CUTMIX = True
USE_AUTOAUGMENT = True

# Maximum Performance XGBoost Configuration
# Balanced XGBoost Configuration for Better Generalization
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc', 'error'],
    
    # Model complexity - very conservative for small dataset
    'n_estimators': 50,    # Much smaller for small dataset
    'max_depth': 3,        # Very shallow to prevent overfitting  
    'learning_rate': 0.1,  # Increased from 0.01 for faster, more stable learning
    'subsample': 0.8,      # Added randomness for better generalization
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.8,
    'colsample_bynode': 0.8,
    
    # GPU Configuration
    'tree_method': 'gpu_hist' if GPU_AVAILABLE else 'hist',
    'gpu_id': 0 if GPU_AVAILABLE else None,
    
    # Regularization - increased for better generalization
    'min_child_weight': 5,   # Increased for more regularization
    'gamma': 0.2,          # Increased regularization
    'reg_alpha': 0.2,      # Increased L1 regularization  
    'reg_lambda': 2.0,     # Increased L2 regularization
    
    # Performance
    'scale_pos_weight': 1,
    'max_delta_step': 1,
    'min_split_loss': 0.1,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'verbosity': 1,
    'enable_categorical': False,
    'early_stopping_rounds': None,  # Removed - causing issues without validation
    'validation_fraction': None,   # Removed - not used in training
}

# Ensemble Configuration
ENSEMBLE_CONFIG = {
    'use_voting': True,
    'use_stacking': True,
    'use_bagging': True,
    'n_estimators_ensemble': 10,
    'meta_learner': 'lightgbm'
}

# Advanced Cross-Validation Configuration
CV_CONFIG = {
    'cv_folds': 15,  # More folds for better validation
    'stratified': True,
    'shuffle': True,
    'random_state': RANDOM_SEED,
    'repeated_cv': True,
    'n_repeats': 3
}

# Model Paths
MODEL_DIR = "./models"
ENHANCED_MODEL_PREFIX = "ultra_enhanced"
XGB_MODEL_PATH = os.path.join(MODEL_DIR, f"{ENHANCED_MODEL_PREFIX}_xgb_forgery_detector.pkl")
FEATURE_EXTRACTOR_PATH = os.path.join(MODEL_DIR, f"{ENHANCED_MODEL_PREFIX}_feature_extractor.pth")
FEATURE_SCALER_PATH = os.path.join(MODEL_DIR, f"{ENHANCED_MODEL_PREFIX}_feature_scaler.pkl")
TRAINING_HISTORY_PATH = os.path.join(MODEL_DIR, f"{ENHANCED_MODEL_PREFIX}_training_history.json")
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, f"{ENHANCED_MODEL_PREFIX}_ensemble.pkl")

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_ROOT, exist_ok=True)

# Ultra-Advanced Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'brightness_alpha': 1.2,
    'brightness_beta': 10,
    'apply_clahe': True,
    'clahe_clip_limit': 4.0,
    'clahe_tile_grid_size': (12, 12),
    'apply_adaptive_gamma': True,
    'gamma_range': (0.6, 1.4),
    'apply_local_contrast_norm': True,
    'apply_sparkle_suppression': True,
    'multi_scale_morphology': True,
    'morphology_kernels': [(3,3), (5,5), (7,7), (9,9)],
    'morphology_weights': [0.3, 0.3, 0.3, 0.1],
    'apply_bilateral_filter': True,
    'bilateral_iterations': 5,
    'bilateral_d': 15,
    'bilateral_sigma_color': 100,
    'bilateral_sigma_space': 100,
    'apply_nlm_denoising': True,
    'nlm_h': 12,
    'nlm_template_window_size': 9,
    'nlm_search_window_size': 25,
    'apply_gaussian_smoothing': True,
    'gaussian_sigma': 0.3,
    'resize_interpolation': 'LANCZOS4',
    'target_size': IMAGE_SIZE,
    'apply_histogram_equalization': True,
    'apply_edge_enhancement': True,
    'edge_enhancement_strength': 1.5,
    'apply_unsharp_masking': True,
    'unsharp_sigma': 1.0,
    'unsharp_strength': 0.8
}

# Ultra-Advanced Augmentation Configuration
AUGMENTATION_CONFIG = {
    'horizontal_flip': 0.5,
    'vertical_flip': 0.3,
    'rotation_range': 20,
    'zoom_range': 0.15,
    'width_shift_range': 0.15,
    'height_shift_range': 0.15,
    'brightness_range': (0.7, 1.3),
    'contrast_range': (0.7, 1.3),
    'saturation_range': (0.7, 1.3),
    'hue_range': (-0.15, 0.15),
    'gaussian_noise_std': 0.03,
    'gaussian_blur_sigma': (0.1, 3.0),
    'jpeg_compression_quality': (60, 100),
    'cutout_probability': 0.4,
    'cutout_size': (32, 32),
    'mixup_alpha': 0.4,
    'cutmix_alpha': 1.5,
    'autoaugment_policy': 'imagenet',
    'elastic_transform': True,
    'grid_distortion': True,
    'optical_distortion': True
}

# Enhanced Evaluation Configuration
EVALUATION_CONFIG = {
    'cross_validation_folds': 15,
    'test_batch_size': 4,
    'save_predictions': True,
    'save_probabilities': True,
    'save_feature_importance': True,
    'plot_results': True,
    'plot_roc_curves': True,
    'plot_confusion_matrix': True,
    'plot_feature_importance': True,
    'save_detailed_report': True,
    'calculate_precision_recall_curve': True,
    'calculate_feature_correlations': True,
    'save_misclassified_examples': True,
    'threshold_optimization': True,
    'bootstrap_evaluation': True,
    'n_bootstrap_samples': 1000
}

# Advanced Training Strategy Configuration
TRAINING_STRATEGY = {
    'use_class_weights': True,
    'use_focal_loss': True,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'gradient_clipping': True,
    'gradient_clip_value': 0.5,
    'learning_rate_scheduling': True,
    'lr_schedule_type': 'cosine_annealing',
    'warmup_epochs': 15,
    'weight_decay': 5e-5,
    'use_swa': True,
    'swa_start_epoch': 80,
    'swa_lr': 0.00001,
    'use_label_smoothing': True,
    'label_smoothing': 0.1
}

# Advanced Feature Engineering Configuration
FEATURE_ENGINEERING = {
    'use_polynomial_features': True,
    'polynomial_degree': 3,
    'use_interaction_features': True,
    'use_feature_selection': True,
    'feature_selection_k': 'auto',
    'feature_selection_method': 'mutual_info_classif',
    'use_pca': True,
    'pca_components': 0.99,
    'normalize_features': True,
    'standardize_features': True,
    'use_quantile_transform': True,
    'use_power_transform': True
}

# Enhanced Hardware Configuration with GPU Optimization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 12 if torch.cuda.is_available() else 6  # Increased for better throughput
PIN_MEMORY = torch.cuda.is_available()
MIXED_PRECISION = torch.cuda.is_available()
USE_GPU_OPTIMIZATION = torch.cuda.is_available()

# GPU-specific configurations
if torch.cuda.is_available():
    # Optimize for GPU memory usage
    torch.backends.cudnn.benchmark = True  # Optimize cudnn for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for better performance
    GPU_MEMORY_FRACTION = 0.9  # Use 90% of GPU memory

# Logging Configuration
LOG_LEVEL = "INFO"

# Dataset utility functions
def get_dataset_info():
    """Get information about the currently active dataset"""
    return {
        "active": ACTIVE_DATASET,
        "name": CURRENT_DATASET["name"],
        "description": CURRENT_DATASET["description"],
        "authentic_dir": AUTHENTIC_DIR,
        "forged_dir": FORGED_DIR,
        "file_extensions": CURRENT_DATASET["file_extensions"],
        "csv_files": {
            "data": DATA_CSV,
            "train": TRAIN_CSV,
            "val": VAL_CSV,
            "test": TEST_CSV
        },
        "model_files": {
            "best_model": BEST_MODEL_PATH,
            "all_models": ALL_MODELS_PATH,
            "scaler": SCALER_PATH,
            "feature_selector": FEATURE_SELECTOR_PATH
        }
    }

def print_dataset_info():
    """Print current dataset configuration"""
    info = get_dataset_info()
    print("="*60)
    print("CURRENT DATASET CONFIGURATION")
    print("="*60)
    print(f"Active Dataset: {info['active'].upper()}")
    print(f"Name: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"Authentic Images: {info['authentic_dir']}")
    print(f"Forged Images: {info['forged_dir']}")
    print(f"File Extensions: {', '.join(info['file_extensions'])}")
    print(f"Results Directory: {RESULTS_DIR}")
    print("\nGenerated Files:")
    print(f"  Labels CSV: {info['csv_files']['data']}")
    print(f"  Train CSV: {info['csv_files']['train']}")
    print(f"  Validation CSV: {info['csv_files']['val']}")
    print(f"  Test CSV: {info['csv_files']['test']}")
    print("\nModel Files:")
    print(f"  Best Model: {info['model_files']['best_model']}")
    print(f"  All Models: {info['model_files']['all_models']}")
    print("="*60)

# Print current configuration on import
if __name__ != "__main__":
    print_dataset_info()
LOG_FILE = "ultra_enhanced_training.log"
TENSORBOARD_LOG_DIR = "./logs/ultra_enhanced"

# Performance Monitoring
EARLY_STOPPING_PATIENCE = 50
CHECKPOINT_FREQUENCY = 5
VALIDATION_FREQUENCY = 1
SAVE_BEST_MODEL = True
SAVE_LAST_MODEL = True

# Maximum Target Performance
TARGET_ACCURACY = 0.95  # 95% target accuracy
TARGET_PRECISION = 0.95
TARGET_RECALL = 0.95
TARGET_F1_SCORE = 0.95
TARGET_AUC = 0.98

print("Ultra-Enhanced configuration loaded successfully!")
print(f"Using device: {DEVICE}")
print(f"Target accuracy: {TARGET_ACCURACY:.1%}")
print(f"Enhanced image size: {IMAGE_SIZE}")
print(f"XGBoost estimators: {XGB_PARAMS['n_estimators']}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"ImageNet models enabled: {len([m for m in IMAGENET_MODELS.values() if m['enabled']])}")

if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
