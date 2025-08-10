import torch
import os

# Essential configurations only - simplified version

# Dataset configurations
DATASETS = {
    "MISD": {
        "name": "MISD Dataset",
        "description": "Multiple Image Splicing Dataset with JPG/BMP images",
        "authentic_path": "./data/Dataset/Au",
        "forged_path": "./data/Dataset/Sp",
        "file_extensions": ["*.jpg", "*.JPG", "*.bmp", "*.png"],
        "results_dir": "./results_misd",
        "labels_csv": "./data/misd_labels.csv",
        "train_csv": "./data/misd_train_labels.csv",
        "val_csv": "./data/misd_val_labels.csv",
        "test_csv": "./data/misd_test_labels.csv",
        "best_model": "./models/misd_best_model.pkl",
        "all_models": "./models/misd_all_models.pkl",
        "feature_selector": "./models/misd_feature_selector.pkl",
        "scaler": "./models/misd_scaler.pkl"
    },
    "4CAM": {
        "name": "4CAM Dataset", 
        "description": "JPEG compression dataset with authentic and spliced images",
        "authentic_path": "./data/4cam_auth",
        "forged_path": "./data/4cam_splc", 
        "file_extensions": ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"],
        "results_dir": "./results_4cam",
        "labels_csv": "./data/4cam_labels.csv",
        "train_csv": "./data/4cam_train_labels.csv",
        "val_csv": "./data/4cam_val_labels.csv", 
        "test_csv": "./data/4cam_test_labels.csv",
        "best_model": "./models/4cam_best_model.pkl",
        "all_models": "./models/4cam_all_models.pkl",
        "feature_selector": "./models/4cam_feature_selector.pkl",
        "scaler": "./models/4cam_scaler.pkl"
    },
    "IMSLICE": {
        "name": "ImSlice Dataset",
        "description": "Image splicing dataset for forgery detection",
        "authentic_path": "./data/ImSpliceDataset/Original",
        "forged_path": "./data/ImSpliceDataset/Spliced",
        "file_extensions": ["*.jpg", "*.JPG", "*.png", "*.PNG"],
        "results_dir": "./results_imslice",
        "labels_csv": "./data/imslice_labels.csv",
        "train_csv": "./data/imslice_train_labels.csv", 
        "val_csv": "./data/imslice_val_labels.csv",
        "test_csv": "./data/imslice_test_labels.csv",
        "best_model": "./models/imslice_best_model.pkl",
        "all_models": "./models/imslice_all_models.pkl",
        "feature_selector": "./models/imslice_feature_selector.pkl",
        "scaler": "./models/imslice_scaler.pkl"
    }
}

# Current dataset
CURRENT_DATASET = "4CAM"
ACTIVE_DATASET = CURRENT_DATASET
DATASET_CONFIG = DATASETS[ACTIVE_DATASET]

# GPU/CPU configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_GPU = torch.cuda.is_available()
GPU_AVAILABLE = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0

# Image processing parameters
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 32 if USE_GPU else 16
NUM_WORKERS = 4 if USE_GPU else 2

# Dynamic variables based on current dataset
RESULTS_DIR = DATASET_CONFIG["results_dir"]
AUTHENTIC_PATH = DATASET_CONFIG["authentic_path"]
FORGED_PATH = DATASET_CONFIG["forged_path"]
FILE_EXTENSIONS = DATASET_CONFIG["file_extensions"]
LABELS_CSV = DATASET_CONFIG["labels_csv"]
TRAIN_CSV = DATASET_CONFIG["train_csv"]
VAL_CSV = DATASET_CONFIG["val_csv"]
TEST_CSV = DATASET_CONFIG["test_csv"]
BEST_MODEL_PATH = DATASET_CONFIG["best_model"]
ALL_MODELS_PATH = DATASET_CONFIG["all_models"]
FEATURE_SELECTOR_PATH = DATASET_CONFIG["feature_selector"]
SCALER_PATH = DATASET_CONFIG["scaler"]

def set_active_dataset(dataset_name):
    global CURRENT_DATASET, ACTIVE_DATASET, DATASET_CONFIG
    global RESULTS_DIR, AUTHENTIC_PATH, FORGED_PATH, FILE_EXTENSIONS
    global LABELS_CSV, TRAIN_CSV, VAL_CSV, TEST_CSV
    global BEST_MODEL_PATH, ALL_MODELS_PATH, FEATURE_SELECTOR_PATH, SCALER_PATH
    
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(DATASETS.keys())}")
    
    CURRENT_DATASET = dataset_name
    ACTIVE_DATASET = dataset_name
    DATASET_CONFIG = DATASETS[dataset_name]
    
    RESULTS_DIR = DATASET_CONFIG["results_dir"]
    AUTHENTIC_PATH = DATASET_CONFIG["authentic_path"]
    FORGED_PATH = DATASET_CONFIG["forged_path"]
    FILE_EXTENSIONS = DATASET_CONFIG["file_extensions"]
    LABELS_CSV = DATASET_CONFIG["labels_csv"]
    TRAIN_CSV = DATASET_CONFIG["train_csv"]
    VAL_CSV = DATASET_CONFIG["val_csv"]
    TEST_CSV = DATASET_CONFIG["test_csv"]
    BEST_MODEL_PATH = DATASET_CONFIG["best_model"]
    ALL_MODELS_PATH = DATASET_CONFIG["all_models"]
    FEATURE_SELECTOR_PATH = DATASET_CONFIG["feature_selector"]
    SCALER_PATH = DATASET_CONFIG["scaler"]
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return f"Dataset set to {dataset_name}"

def print_dataset_info():
    print(f"\nCurrent Dataset: {ACTIVE_DATASET}")
    print(f"Description: {DATASET_CONFIG['description']}")
    print(f"Authentic Path: {AUTHENTIC_PATH}")
    print(f"Forged Path: {FORGED_PATH}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")

# Initialize with default dataset
if CURRENT_DATASET in DATASETS:
    set_active_dataset(CURRENT_DATASET)
