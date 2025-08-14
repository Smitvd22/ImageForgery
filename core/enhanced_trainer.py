#!/usr/bin/env python3
"""
Enhanced Epoch-Based Training System with Advanced Regularization
Implementing iterative learning with proper validation and early stopping
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier
import pickle
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from .config import *
from .preprocessing import get_advanced_augmentation_pipeline, get_test_transform
from .models import UltraEnhancedImageNetBackbone

warnings.filterwarnings('ignore')

class ForgeryDataset(Dataset):
    """Enhanced dataset with advanced augmentation"""
    
    def __init__(self, image_paths, labels, transform=None, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        
        # Load all images into memory for faster training (if memory allows)
        self.images = []
        self.load_images()
    
    def load_images(self):
        """Pre-load images for faster training"""
        print(f"Loading {len(self.image_paths)} images into memory...")
        for img_path in tqdm(self.image_paths):
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMAGE_SIZE)
                self.images.append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                # Create dummy image
                self.images.append(np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            if hasattr(self.transform, '__call__'):
                transformed = self.transform(image=img)
                img = transformed['image']
            else:
                img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.long)

class EnhancedNeuralClassifier(nn.Module):
    """Enhanced neural classifier with advanced regularization"""
    
    def __init__(self, feature_dim=2048, num_classes=2, dropout_rate=0.5):
        super().__init__()
        
        # Feature extraction backbone
        self.backbone = UltraEnhancedImageNetBackbone({
            'efficientnet_v2_l.in1k': {'enabled': True, 'pretrained': True, 'feature_dim': 1280},
            'convnext_large.fb_in22k_ft_in1k': {'enabled': True, 'pretrained': True, 'feature_dim': 1536},
            'swin_large_patch4_window12_384.ms_in22k_ft_in1k': {'enabled': True, 'pretrained': True, 'feature_dim': 1536}
        })
        
        # Adaptive feature dimensions
        backbone_dim = self.backbone.total_feature_dim
        
        # Advanced classifier head with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(256, num_classes)
        )
        
        # Feature noise for regularization
        self.feature_noise_std = FEATURE_NOISE_STD
        self.training_mode = True
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Add noise during training for regularization
        if self.training_mode and self.training:
            noise = torch.randn_like(features) * self.feature_noise_std
            features = features + noise
        
        # Classify
        output = self.classifier(features)
        return output

class EpochBasedTrainer:
    """Enhanced trainer with epoch-based learning and advanced regularization"""
    
    def __init__(self):
        self.device = DEVICE
        self.results_dir = RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        print(f"🚀 Enhanced Epoch-Based Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    
    def create_model(self):
        """Create enhanced neural classifier"""
        model = EnhancedNeuralClassifier(
            dropout_rate=DROPOUT_RATE
        )
        return model.to(self.device)
    
    def create_data_loaders(self, train_df, val_df):
        """Create enhanced data loaders with augmentation"""
        
        # Training dataset with augmentation
        train_transform = get_advanced_augmentation_pipeline()
        train_dataset = ForgeryDataset(
            train_df['filepath'].values,
            train_df['label'].values,
            transform=train_transform,
            is_training=True
        )
        
        # Validation dataset without augmentation
        val_transform = get_test_transform()
        val_dataset = ForgeryDataset(
            val_df['filepath'].values,
            val_df['label'].values,
            transform=val_transform,
            is_training=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # L2 regularization
            l2_reg = torch.tensor(0.).to(self.device)
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)
            loss += L2_REGULARIZATION * l2_reg
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        # Calculate additional metrics
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        return epoch_loss, epoch_acc, f1
    
    def train_with_cross_validation(self, df):
        """Train with cross-validation for robust evaluation"""
        print("🔄 Starting cross-validation training...")
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        
        cv_results = []
        best_models = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
            print(f"\n📋 Fold {fold + 1}/{CV_FOLDS}")
            
            # Split data
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)
            
            # Create data loaders
            train_loader, val_loader = self.create_data_loaders(train_df, val_df)
            
            # Create model
            model = self.create_model()
            
            # Optimizer and scheduler
            optimizer = optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=L2_REGULARIZATION,
                betas=(0.9, 0.999)
            )
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=REDUCE_LR_FACTOR,
                patience=REDUCE_LR_PATIENCE,
                min_lr=MIN_LR,
                verbose=True
            )
            
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            
            # Training loop
            best_val_acc = 0.0
            patience_counter = 0
            
            for epoch in range(NUM_EPOCHS):
                # Train
                train_loss, train_acc = self.train_epoch(
                    model, train_loader, optimizer, criterion, epoch
                )
                
                # Validate
                val_loss, val_acc, val_f1 = self.validate_epoch(
                    model, val_loader, criterion
                )
                
                # Update scheduler
                scheduler.step(val_loss)
                
                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model for this fold
                    best_model_path = os.path.join(self.results_dir, f'best_model_fold_{fold}.pth')
                    torch.save(model.state_dict(), best_model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Load best model for this fold
            model.load_state_dict(torch.load(best_model_path))
            best_models.append(model)
            
            cv_results.append({
                'fold': fold + 1,
                'best_val_acc': best_val_acc,
                'val_f1': val_f1
            })
        
        # Calculate cross-validation statistics
        mean_acc = np.mean([r['best_val_acc'] for r in cv_results])
        std_acc = np.std([r['best_val_acc'] for r in cv_results])
        
        print(f"\n✅ Cross-Validation Results:")
        print(f"   Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
        
        # Save CV results
        cv_results_path = os.path.join(self.results_dir, 'cv_results.json')
        with open(cv_results_path, 'w') as f:
            json.dump(cv_results, f, indent=2)
        
        return best_models, cv_results
    
    def create_ensemble(self, models):
        """Create ensemble from cross-validation models"""
        print("🎯 Creating ensemble from CV models...")
        
        class EnsembleModel(nn.Module):
            def __init__(self, models):
                super().__init__()
                self.models = nn.ModuleList(models)
            
            def forward(self, x):
                outputs = []
                for model in self.models:
                    with torch.no_grad():
                        output = model(x)
                        outputs.append(torch.softmax(output, dim=1))
                
                # Average predictions
                ensemble_output = torch.mean(torch.stack(outputs), dim=0)
                return torch.log(ensemble_output + 1e-8)  # Convert back to log probabilities
        
        ensemble = EnsembleModel(models)
        return ensemble
    
    def plot_training_history(self):
        """Plot training curves"""
        if not self.train_losses:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Training Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
