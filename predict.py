#!/usr/bin/env python3
"""
Image Prediction Script
Predicts if an image is authentic or forged using trained models
"""

import sys
import logging
import numpy as np
import torch
import joblib
from PIL import Image
from pathlib import Path
import cv2
import os
from core.config import *

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_models():
    """Load the trained models"""
    try:
        # Load the best model
        if os.path.exists("./models/best_model.pkl"):
            classifier = joblib.load("./models/best_model.pkl")
            logger.info("Best model loaded successfully")
        else:
            # Try to load any available model
            model_files = [f for f in os.listdir("./models") if f.endswith("_model.pkl")]
            if model_files:
                model_path = f"./models/{model_files[0]}"
                classifier = joblib.load(model_path)
                logger.info(f"Loaded model: {model_files[0]}")
            else:
                raise FileNotFoundError("No trained models found")
        
        # Load preprocessors
        preprocessors = joblib.load("./models/preprocessors.pkl")
        logger.info("Preprocessors loaded successfully")
        
        return classifier, preprocessors
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.error("Please run 'python train.py' first to create trained models")
        return None, None

def extract_features_with_cnn(image_path):
    """Extract features using the same method as training"""
    try:
        # Import the trainer class
        from train import FinalUltraAdvancedTrainer
        
        # Create a temporary trainer instance just for feature extraction
        class Config:
            batch_size = 8
            image_size = (384, 384)
            device = torch.device('cpu')
        
        config = Config()
        trainer = FinalUltraAdvancedTrainer(config)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize to standard size
        image = cv2.resize(image, config.image_size)
        
        # Convert image to tensor format (similar to data loader)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        image_batch = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Extract CNN features
        cnn_features = trainer.extract_cnn_features(image_batch)
        
        # Extract forensic features
        forensic_features = trainer.extract_forensic_features_from_array(image)
        
        # Combine features (flatten CNN features for single image)
        cnn_features_flat = cnn_features.flatten()
        combined_features = np.concatenate([cnn_features_flat, forensic_features])
        
        return combined_features.reshape(1, -1)  # Add batch dimension
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

def predict_image(image_path):
    """Predict if an image is authentic or forged"""
    try:
        # Load models
        classifier, preprocessors = load_models()
        if classifier is None or preprocessors is None:
            return None
        
        # Extract features
        features = extract_features_with_cnn(image_path)
        if features is None:
            return None
        
        # Apply the same preprocessing as training
        logger.info("Applying preprocessing...")
        
        # Clean features (handle infinite/NaN values)
        features = np.where(np.isinf(features), np.sign(features) * 1e10, features)
        features = np.where(np.isnan(features), 0, features)
        features = np.clip(features, -1e10, 1e10)
        
        # Apply preprocessing steps
        features = preprocessors['imputer'].transform(features)
        features = preprocessors['variance_filter'].transform(features)
        features = preprocessors['scaler'].transform(features)
        features = preprocessors['feature_selector'].transform(features)
        
        # Make prediction
        prediction = classifier.predict(features)[0]
        probability = classifier.predict_proba(features)[0]
        
        return {
            'prediction': int(prediction),
            'label': 'Forged' if prediction == 1 else 'Authentic',
            'confidence': float(max(probability)),
            'probabilities': {
                'Authentic': float(probability[0]),
                'Forged': float(probability[1])
            }
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None

def main():
    """Main prediction function"""
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py data/4cam_auth/canong3_02_sub_01.tif")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        sys.exit(1)
    
    logger.info(f"Analyzing image: {image_path}")
    
    # Make prediction
    result = predict_image(image_path)
    
    if result is None:
        logger.error("Prediction failed")
        sys.exit(1)
    
    # Display results
    print("\n" + "="*60)
    print("🔍 IMAGE FORGERY DETECTION RESULTS")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nDetailed Probabilities:")
    print(f"  Authentic: {result['probabilities']['Authentic']:.2%}")
    print(f"  Forged:    {result['probabilities']['Forged']:.2%}")
    print("="*60)
    
    # Interpretation
    if result['confidence'] > 0.8:
        confidence_level = "Very High"
    elif result['confidence'] > 0.6:
        confidence_level = "High"
    elif result['confidence'] > 0.5:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"
    
    print(f"Confidence Level: {confidence_level}")
    
    if result['label'] == 'Forged' and result['confidence'] > 0.7:
        print("⚠️  High probability of image manipulation detected!")
    elif result['label'] == 'Authentic' and result['confidence'] > 0.7:
        print("✅ Image appears to be authentic")
    else:
        print("ℹ️  Results are inconclusive - manual inspection recommended")
    
    print("="*60)

if __name__ == "__main__":
    main()
