#!/usr/bin/env python3
"""
🚀 Optimized Image Forgery Detection Prediction
GPU-accelerated with CPU fallback, supports both 4CAM and MISD datasets
"""
import os
import sys
import warnings
import numpy as np
import pickle
import json
import argparse
from pathlib import Path
import logging

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import configuration
from core.config import *

class OptimizedPredictor:
    """Optimized predictor with automatic GPU/CPU handling and dataset support"""
    
    def __init__(self, dataset=None):
        self.device = DEVICE
        self.gpu_available = GPU_AVAILABLE
        self.gpu_name = GPU_NAME
        
        # Determine which dataset to use
        if dataset is None:
            self.dataset = ACTIVE_DATASET.lower()
        else:
            self.dataset = dataset.lower()
        
        # Model components
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.rfe_selector = None
        self.loaded_models = {}
        self.config = None
        
        logger.info(f"🎮 Device: {self.device}")
        logger.info(f"📊 Dataset: {self.dataset.upper()}")
        if self.gpu_available:
            logger.info(f"🚀 GPU: {self.gpu_name}")
    
    def load_models(self):
        """Load trained models and components for the specific dataset"""
        try:
            # Use dataset-specific model paths
            model_prefix = f"{self.dataset}_"
            
            # Load best model
            model_path = os.path.join(MODELS_DIR, f"{model_prefix}best_model.pkl")
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"✅ Best model loaded for {self.dataset.upper()}: {model_path}")
            else:
                # Fallback to legacy model paths
                fallback_paths = [
                    './models/complete_best_model.pkl',
                    './models/optimized_best_model.pkl',
                    './models/gpu_windows_best_model.pkl',
                    './models/best_model.pkl'
                ]
                for path in fallback_paths:
                    if os.path.exists(path):
                        with open(path, 'rb') as f:
                            self.model = pickle.load(f)
                        logger.info(f"✅ Fallback model loaded: {path}")
                        break
            
            if self.model is None:
                logger.error(f"❌ No trained model found for {self.dataset.upper()} dataset")
                return False
            
            # Load scaler
            scaler_path = os.path.join(MODELS_DIR, f"{model_prefix}scaler.pkl")
            
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"✅ Scaler loaded for {self.dataset.upper()}")
            else:
                # Fallback scaler
                fallback_scalers = [
                    './models/complete_scaler.pkl',
                    './models/optimized_scaler.pkl',
                    './models/gpu_windows_scaler.pkl',
                    './models/preprocessors.pkl'
                ]
                for path in fallback_scalers:
                    if os.path.exists(path):
                        with open(path, 'rb') as f:
                            self.scaler = pickle.load(f)
                        logger.info(f"✅ Fallback scaler loaded: {path}")
                        break
            
            # Load feature selector if available
            feature_selector_path = os.path.join(MODELS_DIR, f"{model_prefix}feature_selector.pkl")
            if os.path.exists(feature_selector_path):
                with open(feature_selector_path, 'rb') as f:
                    self.feature_selector = pickle.load(f)
                logger.info(f"✅ Feature selector loaded for {self.dataset.upper()}")
            
            # Load RFE selector if available
            rfe_selector_path = os.path.join(MODELS_DIR, f"{model_prefix}rfe_selector.pkl")
            if os.path.exists(rfe_selector_path):
                with open(rfe_selector_path, 'rb') as f:
                    self.rfe_selector = pickle.load(f)
                logger.info(f"✅ RFE selector loaded for {self.dataset.upper()}")
            else:
                self.rfe_selector = None
            
            # Load configuration
            config_path = os.path.join(MODELS_DIR, f"{model_prefix}config.json")
            if not os.path.exists(config_path):
                # Fallback config paths
                fallback_configs = [
                    './models/complete_config.json',
                    './models/optimized_config.json'
                ]
                for path in fallback_configs:
                    if os.path.exists(path):
                        config_path = path
                        break
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info("✅ Configuration loaded")
            
            # Load CNN models
            self.load_cnn_models()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    def load_cnn_models(self):
        """Load pre-trained CNN models for feature extraction"""
        try:
            import timm
            import torch
            
            cnn_models = ['resnet50', 'efficientnet_b2', 'densenet121']
            
            for model_name in cnn_models:
                model = timm.create_model(f'{model_name}.ra_in1k', pretrained=True, num_classes=0)
                model = model.to(self.device)
                model.eval()
                self.loaded_models[model_name] = model
                logger.info(f"✅ {model_name} loaded")
            
            return True
        except Exception as e:
            logger.warning(f"⚠️ Error loading CNN models: {e}")
            return False
    
    def extract_features(self, image_path):
        """Extract features from a single image"""
        import torch
        from PIL import Image
        import torchvision.transforms as T
        import cv2
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Transform for CNN models
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            features = []
            
            # Extract CNN features if models are loaded
            if self.loaded_models:
                img_tensor = transform(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    for model_name, model in self.loaded_models.items():
                        feat = model(img_tensor).cpu().numpy().flatten()
                        features.extend(feat)
            
            # Extract basic features
            basic_features = self.extract_basic_features(image)
            features.extend(basic_features)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"❌ Error extracting features from {image_path}: {e}")
            return None
    
    def extract_basic_features(self, image):
        """Extract basic statistical features from image"""
        import cv2
        import numpy as np
        
        # Convert PIL to numpy
        img_array = np.array(image)
        
        features = []
        
        # Convert to different color spaces
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Basic statistics for each channel
        for channel in [img_array[:,:,0], img_array[:,:,1], img_array[:,:,2], gray]:
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.median(channel),
                np.min(channel),
                np.max(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75)
            ])
        
        # HSV statistics
        for channel in [hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]]:
            features.extend([
                np.mean(channel),
                np.std(channel)
            ])
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        features.extend([
            np.mean(edges),
            np.std(edges),
            np.sum(edges > 0) / edges.size  # Edge density
        ])
        
        return features
    
    def predict(self, image_path):
        """Predict if image is forged or authentic"""
        try:
            # Extract features
            features = self.extract_features(image_path)
            if features is None:
                return None, None
            
            # Apply feature selection first if available
            if self.feature_selector is not None:
                features = self.feature_selector.transform(features)
            
            # Apply RFE selection if available
            if self.rfe_selector is not None:
                features = self.rfe_selector.transform(features)
            
            # Scale features if scaler is available
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            # Get probability if available
            probability = None
            if hasattr(self.model, 'predict_proba'):
                prob = self.model.predict_proba(features)[0]
                # Show confidence for the predicted class
                probability = prob[prediction] if len(prob) > 1 else prob[0]
            
            return prediction, probability
            
        except Exception as e:
            logger.error(f"❌ Error making prediction: {e}")
            return None, None
    
    def predict_batch(self, image_dir):
        """Predict on all images in a directory"""
        from pathlib import Path
        
        image_dir = Path(image_dir)
        
        # Get file extensions based on current dataset
        current_dataset_config = DATASETS.get(self.dataset, {})
        dataset_extensions = current_dataset_config.get('file_extensions', ['*.jpg', '*.png', '*.tif', '*.bmp'])
        
        # Convert to list of extensions for glob
        image_extensions = []
        for ext in dataset_extensions:
            # Remove the * from the beginning
            clean_ext = ext.replace('*', '')
            image_extensions.append(clean_ext)
            image_extensions.append(clean_ext.upper())
        
        logger.info(f"📂 Looking for extensions: {image_extensions}")
        
        results = []
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            pattern = f'*{ext}'
            found_files = list(image_dir.glob(pattern))
            image_files.extend(found_files)
        
        # Remove duplicates
        image_files = list(set(image_files))
        
        logger.info(f"📂 Found {len(image_files)} images")
        
        for image_path in image_files:
            prediction, probability = self.predict(str(image_path))
            
            if prediction is not None:
                result = {
                    'image': image_path.name,
                    'path': str(image_path),
                    'prediction': 'Forged' if prediction == 1 else 'Authentic',
                    'label': int(prediction),
                    'confidence': probability if probability is not None else 0.5,
                    'dataset': self.dataset.upper()
                }
                results.append(result)
                
                print(f"📷 {image_path.name}: {result['prediction']} "
                      f"(Confidence: {result['confidence']:.3f})")
        
        return results

def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description='Image Forgery Detection Prediction - Multi-Dataset Support')
    parser.add_argument('input', nargs='?', help='Image file or directory path')
    parser.add_argument('--dataset', '-d', choices=['4cam', 'misd', 'imsplice','micc-f220'], 
                       help='Dataset to use (4cam, misd, or imsplice). If not specified, uses current active dataset')
    parser.add_argument('--output', '-o', help='Output JSON file for batch results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--list-datasets', action='store_true', help='List available datasets and exit')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle list datasets option
    if args.list_datasets:
        print("=" * 60)
        print("📊 AVAILABLE DATASETS")
        print("=" * 60)
        for key, dataset_info in DATASETS.items():
            status = "✅ Available" if (os.path.exists(dataset_info["authentic_dir"]) and 
                                     os.path.exists(dataset_info["forged_dir"])) else "❌ Not Found"
            model_path = os.path.join(MODELS_DIR, f"{key}_best_model.pkl")
            model_status = "✅ Model Available" if os.path.exists(model_path) else "❌ Model Missing"
            
            print(f"\n{key.upper()} Dataset:")
            print(f"  Name: {dataset_info['name']}")
            print(f"  Description: {dataset_info['description']}")
            print(f"  Data Status: {status}")
            print(f"  Model Status: {model_status}")
        
        current_dataset = ACTIVE_DATASET
        print(f"\n🎯 Current Active Dataset: {current_dataset.upper()}")
        print("=" * 60)
        return
    
    # Check if input is provided when not listing datasets
    if not args.input:
        parser.error("Input path is required unless using --list-datasets")
    
    print("=" * 60)
    print("🚀 OPTIMIZED IMAGE FORGERY DETECTION")
    print("=" * 60)
    
    # Determine which dataset to use
    dataset_to_use = args.dataset if args.dataset else None
    
    # Initialize predictor
    predictor = OptimizedPredictor(dataset=dataset_to_use)
    
    # Load models
    if not predictor.load_models():
        print(f"❌ Failed to load models for {predictor.dataset.upper()} dataset.")
        print("💡 Available options:")
        print("  1. Train the model first using: python train.py")
        print("  2. Switch dataset using: python dataset_manager.py --switch <dataset>")
        print("  3. List available datasets: python predict_optimized.py --list-datasets")
        return
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image prediction
        logger.info(f"📷 Analyzing image: {input_path}")
        
        prediction, probability = predictor.predict(str(input_path))
        
        if prediction is not None:
            result = 'Forged' if prediction == 1 else 'Authentic'
            confidence = probability if probability is not None else 0.5
            
            print(f"\n🎯 RESULT:")
            print(f"� Dataset: {predictor.dataset.upper()}")
            print(f"�📷 Image: {input_path.name}")
            print(f"🔍 Prediction: {result}")
            print(f"📊 Confidence: {confidence:.3f}")
            
            if confidence > 0.8:
                print("✅ High confidence")
            elif confidence > 0.6:
                print("⚠️ Medium confidence")
            else:
                print("❓ Low confidence")
                
            # Additional interpretation
            if result == 'Forged':
                print("🔴 The image appears to be FORGED/MANIPULATED")
            else:
                print("🟢 The image appears to be AUTHENTIC/ORIGINAL")
        else:
            print("❌ Failed to process image")
    
    elif input_path.is_dir():
        # Batch prediction
        logger.info(f"📂 Analyzing directory: {input_path}")
        
        results = predictor.predict_batch(str(input_path))
        
        if results:
            # Summary statistics
            total = len(results)
            forged = sum(1 for r in results if r['label'] == 1)
            authentic = total - forged
            avg_confidence = sum(r['confidence'] for r in results) / total
            
            print(f"\n📊 BATCH RESULTS:")
            print(f"� Dataset: {predictor.dataset.upper()}")
            print(f"�📷 Total Images: {total}")
            print(f"🔴 Forged: {forged} ({forged/total*100:.1f}%)")
            print(f"🟢 Authentic: {authentic} ({authentic/total*100:.1f}%)")
            print(f"📊 Average Confidence: {avg_confidence:.3f}")
            
            # Save results if output specified
            if args.output:
                import json
                batch_summary = {
                    'dataset': predictor.dataset.upper(),
                    'summary': {
                        'total_images': total,
                        'forged_count': forged,
                        'authentic_count': authentic,
                        'forged_percentage': forged/total*100,
                        'average_confidence': avg_confidence
                    },
                    'predictions': results
                }
                with open(args.output, 'w') as f:
                    json.dump(batch_summary, f, indent=2)
                print(f"💾 Results saved to {args.output}")
        else:
            print("❌ No images processed successfully")
    
    else:
        print(f"❌ Invalid input path: {input_path}")

if __name__ == "__main__":
    main()
