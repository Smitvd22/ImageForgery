import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
from .preprocessing import preprocess_image, preprocess_batch
from .config import *
import glob
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

class ForgeryDataset(Dataset):
    """
    Enhanced dataset class for image forgery detection
    """
    def __init__(self, csv_path=None, authentic_dir=None, forged_dir=None, 
                 apply_augmentation=False, image_size=(256, 256)):
        self.apply_augmentation = apply_augmentation
        self.image_size = image_size
        
        if csv_path and os.path.exists(csv_path):
            # Load from CSV if it exists
            self.data = pd.read_csv(csv_path)
            logger.info(f"Loaded dataset from {csv_path}: {len(self.data)} samples")
        else:
            # Create dataset from directory structure
            self.data = self._create_dataset_from_dirs(authentic_dir, forged_dir)
            logger.info(f"Created dataset from directories: {len(self.data)} samples")

    def _create_dataset_from_dirs(self, authentic_dir, forged_dir):
        """Create dataset DataFrame from directory structure"""
        data = []
        
        # Get file extensions for current dataset
        file_extensions = CURRENT_DATASET["file_extensions"]
        
        # Handle ImSpliceDataset special structure
        if ACTIVE_DATASET == "imslice" and CURRENT_DATASET.get("use_subdirectories", False):
            return self._create_imslice_dataset(authentic_dir, file_extensions)
        
        # Process authentic images
        if authentic_dir and os.path.exists(authentic_dir):
            authentic_files = []
            for ext in file_extensions:
                authentic_files.extend(glob.glob(os.path.join(authentic_dir, ext)))
            
            for file_path in authentic_files:
                # Extract category from filename
                filename = os.path.basename(file_path)
                category = 'authentic'
                
                # Handle different dataset naming conventions
                if ACTIVE_DATASET == "misd" and filename.startswith('Au_'):
                    # Extract category from MISD naming convention: Au_category_index.jpg
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        category = f"authentic_{parts[1]}"  # e.g., authentic_ani, authentic_art
                elif ACTIVE_DATASET == "4cam":
                    # Extract category from 4cam naming convention if needed
                    category = 'authentic_4cam'
                
                data.append({
                    'filename': filename,
                    'filepath': file_path,
                    'label': 0,  # 0 for authentic
                    'category': category
                })
            logger.info(f"Found {len(authentic_files)} authentic images")
        
        # Process forged images
        if forged_dir and os.path.exists(forged_dir):
            forged_files = []
            for ext in file_extensions:
                forged_files.extend(glob.glob(os.path.join(forged_dir, ext)))
            
            for file_path in forged_files:
                # Extract information from filename
                filename = os.path.basename(file_path)
                category = 'forged'
                
                # Handle different dataset naming conventions
                if ACTIVE_DATASET == "misd" and filename.startswith('Sp_D_'):
                    # MISD spliced images: Sp_D_source_target_..._id.jpg
                    category = 'multiple_spliced'
                elif ACTIVE_DATASET == "4cam":
                    # 4cam forged images
                    category = 'forged_4cam'
                
                data.append({
                    'filename': filename,
                    'filepath': file_path,
                    'label': 1,  # 1 for forged
                    'category': category
                })
            logger.info(f"Found {len(forged_files)} forged images")
        
        return pd.DataFrame(data)

    def _create_imslice_dataset(self, base_dir, file_extensions):
        """Create dataset for ImSplice dataset with subdirectory structure"""
        data = []
        
        if not os.path.exists(base_dir):
            logger.error(f"ImSplice dataset directory not found: {base_dir}")
            return pd.DataFrame(data)
        
        # Get authentic and forged prefixes
        authentic_prefixes = CURRENT_DATASET.get("authentic_prefixes", ["Au-"])
        forged_prefixes = CURRENT_DATASET.get("forged_prefixes", ["Sp-"])
        
        # Process all subdirectories
        for subdir in os.listdir(base_dir):
            subdir_path = os.path.join(base_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            
            # Determine if this is authentic or forged based on directory name
            is_authentic = any(subdir.startswith(prefix) for prefix in authentic_prefixes)
            is_forged = any(subdir.startswith(prefix) for prefix in forged_prefixes)
            
            if not (is_authentic or is_forged):
                logger.warning(f"Skipping unknown subdirectory: {subdir}")
                continue
            
            # Get all files in subdirectory
            subdir_files = []
            for ext in file_extensions:
                subdir_files.extend(glob.glob(os.path.join(subdir_path, ext)))
            
            for file_path in subdir_files:
                filename = os.path.basename(file_path)
                
                if is_authentic:
                    label = 0
                    category = f"authentic_{subdir}"
                else:  # is_forged
                    label = 1
                    category = f"forged_{subdir}"
                
                data.append({
                    'filename': filename,
                    'filepath': file_path,
                    'label': label,
                    'category': category,
                    'subdirectory': subdir
                })
            
            logger.info(f"Found {len(subdir_files)} images in {subdir}")
        
        logger.info(f"Total ImSplice dataset: {len(data)} images")
        authentic_count = sum(1 for item in data if item['label'] == 0)
        forged_count = sum(1 for item in data if item['label'] == 1)
        logger.info(f"  Authentic: {authentic_count} images")
        logger.info(f"  Forged: {forged_count} images")
        
        return pd.DataFrame(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get image path
        if 'filepath' in row:
            image_path = row['filepath']
        else:
            # Fallback to old format
            image_path = row['filename']  # Assumes full path in filename
        
        label = row['label']
        
        try:
            # Preprocess image
            image = preprocess_image(
                image_path, 
                size=self.image_size, 
                apply_augmentation=self.apply_augmentation
            )
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            # Return a dummy tensor in case of error
            dummy_image = torch.zeros(3, self.image_size[0], self.image_size[1])
            return dummy_image, torch.tensor(label, dtype=torch.long)

    def get_class_distribution(self):
        """Get distribution of classes in the dataset"""
        class_counts = self.data['label'].value_counts().sort_index()
        class_names = ['Authentic', 'Forged']
        
        logger.info("Class Distribution:")
        for i, count in enumerate(class_counts):
            logger.info(f"  {class_names[i]}: {count} ({count/len(self.data)*100:.1f}%)")
        
        return class_counts

    def save_to_csv(self, csv_path):
        """Save dataset to CSV file"""
        self.data.to_csv(csv_path, index=False)
        logger.info(f"Dataset saved to {csv_path}")

def create_dataset_splits(authentic_dir=AUTHENTIC_DIR, forged_dir=FORGED_DIR, 
                         train_split=TRAIN_SPLIT, val_split=VAL_SPLIT, 
                         test_split=TEST_SPLIT, random_seed=RANDOM_SEED):
    """
    Create train/validation/test splits and save to CSV files
    """
    logger.info("Creating dataset splits...")
    
    # Create full dataset
    full_dataset = ForgeryDataset(
        authentic_dir=authentic_dir,
        forged_dir=forged_dir
    )
    
    # Get class distribution
    full_dataset.get_class_distribution()
    
    # Split dataset
    data = full_dataset.data
    
    # First split: train + val vs test
    train_val_data, test_data = train_test_split(
        data, 
        test_size=test_split, 
        random_state=random_seed,
        stratify=data['label']
    )
    
    # Second split: train vs val
    val_split_adjusted = val_split / (train_split + val_split)
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_split_adjusted,
        random_state=random_seed,
        stratify=train_val_data['label']
    )
    
    # Save splits to CSV
    train_data.to_csv(TRAIN_CSV, index=False)
    val_data.to_csv(VAL_CSV, index=False)
    test_data.to_csv(TEST_CSV, index=False)
    full_dataset.save_to_csv(DATA_CSV)
    
    logger.info(f"Dataset splits created:")
    logger.info(f"  Train: {len(train_data)} samples ({len(train_data)/len(data)*100:.1f}%)")
    logger.info(f"  Validation: {len(val_data)} samples ({len(val_data)/len(data)*100:.1f}%)")
    logger.info(f"  Test: {len(test_data)} samples ({len(test_data)/len(data)*100:.1f}%)")
    
    return train_data, val_data, test_data

def get_data_loaders(batch_size=BATCH_SIZE, num_workers=0, train_csv=None, val_csv=None, test_csv=None, 
                     image_size=None, use_augmentation=None, pin_memory=None):
    """
    Create data loaders for training, validation, and testing
    """
    # Use provided CSV paths or defaults
    train_csv = train_csv or TRAIN_CSV
    val_csv = val_csv or VAL_CSV  
    test_csv = test_csv or TEST_CSV
    image_size = image_size or IMAGE_SIZE
    use_augmentation = use_augmentation if use_augmentation is not None else USE_AUGMENTATION
    pin_memory = pin_memory if pin_memory is not None else torch.cuda.is_available()
    
    # Create datasets
    train_dataset = ForgeryDataset(train_csv, apply_augmentation=use_augmentation, image_size=image_size)
    val_dataset = ForgeryDataset(val_csv, apply_augmentation=False, image_size=image_size)
    test_dataset = ForgeryDataset(test_csv, apply_augmentation=False, image_size=image_size)
    
    # Create data loaders with GPU optimization
    # Optimize workers for GPU usage
    if num_workers == 0:
        num_workers = 8 if torch.cuda.is_available() else 4
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # For consistent batch sizes on GPU
        persistent_workers=torch.cuda.is_available()  # Keep workers alive on GPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=torch.cuda.is_available()
    )
    
    logger.info(f"Data loaders created:")
    logger.info(f"  Train: {len(train_loader)} batches")
    logger.info(f"  Validation: {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

def verify_dataset():
    """
    Verify that the dataset directories exist and contain images
    """
    logger.info(f"Verifying {ACTIVE_DATASET.upper()} dataset...")
    
    # Print current dataset info
    print_dataset_info()
    
    # Handle ImSpliceDataset special structure
    if ACTIVE_DATASET == "imslice" and CURRENT_DATASET.get("use_subdirectories", False):
        return verify_imslice_dataset()
    
    # Check directories
    if not os.path.exists(AUTHENTIC_DIR):
        logger.error(f"Authentic directory not found: {AUTHENTIC_DIR}")
        return False
    
    if not os.path.exists(FORGED_DIR):
        logger.error(f"Forged directory not found: {FORGED_DIR}")
        return False
    
    # Count files using current dataset extensions
    file_extensions = CURRENT_DATASET["file_extensions"]
    
    auth_files = 0
    forged_files = 0
    
    for ext in file_extensions:
        auth_files += len(glob.glob(os.path.join(AUTHENTIC_DIR, ext)))
        forged_files += len(glob.glob(os.path.join(FORGED_DIR, ext)))
    
    logger.info(f"Found {auth_files} authentic images")
    logger.info(f"Found {forged_files} forged images")
    
    if auth_files == 0 or forged_files == 0:
        logger.error("No images found in one or both directories")
        return False
    
    logger.info("Dataset verification passed!")
    return True

def verify_imslice_dataset():
    """
    Verify ImSpliceDataset with subdirectory structure
    """
    base_dir = AUTHENTIC_DIR  # Both authentic and forged are in same base directory
    
    if not os.path.exists(base_dir):
        logger.error(f"ImSplice dataset directory not found: {base_dir}")
        return False
    
    # Get authentic and forged prefixes
    authentic_prefixes = CURRENT_DATASET.get("authentic_prefixes", ["Au-"])
    forged_prefixes = CURRENT_DATASET.get("forged_prefixes", ["Sp-"])
    file_extensions = CURRENT_DATASET["file_extensions"]
    
    auth_files = 0
    forged_files = 0
    auth_dirs = []
    forged_dirs = []
    
    # Check all subdirectories
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        # Determine if this is authentic or forged
        is_authentic = any(subdir.startswith(prefix) for prefix in authentic_prefixes)
        is_forged = any(subdir.startswith(prefix) for prefix in forged_prefixes)
        
        if is_authentic:
            auth_dirs.append(subdir)
            for ext in file_extensions:
                auth_files += len(glob.glob(os.path.join(subdir_path, ext)))
        elif is_forged:
            forged_dirs.append(subdir)
            for ext in file_extensions:
                forged_files += len(glob.glob(os.path.join(subdir_path, ext)))
    
    logger.info(f"Found {len(auth_dirs)} authentic subdirectories: {auth_dirs}")
    logger.info(f"Found {len(forged_dirs)} forged subdirectories: {forged_dirs}")
    logger.info(f"Found {auth_files} authentic images")
    logger.info(f"Found {forged_files} forged images")
    
    if auth_files == 0 or forged_files == 0:
        logger.error("No images found in authentic or forged subdirectories")
        return False
    
    logger.info("ImSplice dataset verification passed!")
    return True

if __name__ == "__main__":
    # Verify dataset and create splits if running directly
    if verify_dataset():
        create_dataset_splits()
        
        # Test data loaders
        train_loader, val_loader, test_loader = get_data_loaders(batch_size=4)
        
        # Test loading a batch
        for images, labels in train_loader:
            logger.info(f"Batch shape: {images.shape}")
            logger.info(f"Labels: {labels}")
            break

