#!/usr/bin/env python3
"""
🚀 Optimized Image Forgery Detection Prediction
GPU-accelerated with CPU fallback, simplified and clean
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
    """Optimized predictor with automatic GPU/CPU handling"""
    
    def __init__(self):
        self.device = DEVICE
        self.gpu_available = GPU_AVAILABLE
        self.gpu_name = GPU_NAME
        
        # Model components
        self.model = None
        self.scaler = None
        self.loaded_models = {}
        self.config = None
        
        logger.info(f"🎮 Device: {self.device}")
        if self.gpu_available:
            logger.info(f"🚀 GPU: {self.gpu_name}")
    
    def load_models(self):
        """Load trained models and components"""
        try:
            # Load best model (try complete first, then optimized)
            model_path = './models/complete_best_model.pkl'
            if not os.path.exists(model_path):
                model_path = './models/optimized_best_model.pkl'
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("✅ Best model loaded")
            else:
                # Fallback to other models
                fallback_paths = [
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
                logger.error("❌ No trained model found")
                return False
            
            # Load scaler (try complete first, then optimized)
            scaler_path = './models/complete_scaler.pkl'
            if not os.path.exists(scaler_path):
                scaler_path = './models/optimized_scaler.pkl'
            
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("✅ Scaler loaded")
            else:
                # Fallback scaler
                fallback_scalers = [
                    './models/gpu_windows_scaler.pkl',
                    './models/preprocessors.pkl'
                ]
                for path in fallback_scalers:
                    if os.path.exists(path):
                        with open(path, 'rb') as f:
                            self.scaler = pickle.load(f)
                        logger.info(f"✅ Fallback scaler loaded: {path}")
                        break
            
            # Load configuration (try complete first, then optimized)
            config_path = './models/complete_config.json'
            if not os.path.exists(config_path):
                config_path = './models/optimized_config.json'
            
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
            
            # Scale features if scaler is available
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            # Get probability if available
            probability = None
            if hasattr(self.model, 'predict_proba'):
                prob = self.model.predict_proba(features)[0]
                probability = prob[1]  # Probability of being forged
            
            return prediction, probability
            
        except Exception as e:
            logger.error(f"❌ Error making prediction: {e}")
            return None, None
    
    def predict_batch(self, image_dir):
        """Predict on all images in a directory"""
        from pathlib import Path
        
        image_dir = Path(image_dir)
        image_extensions = ['.tif', '.jpg', '.jpeg', '.png', '.bmp']
        
        results = []
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
            image_files.extend(image_dir.glob(f'*{ext.upper()}'))
        
        logger.info(f"📂 Found {len(image_files)} images")
        
        for image_path in image_files:
            prediction, probability = self.predict(str(image_path))
            
            if prediction is not None:
                result = {
                    'image': image_path.name,
                    'path': str(image_path),
                    'prediction': 'Forged' if prediction == 1 else 'Authentic',
                    'label': int(prediction),
                    'confidence': probability if probability is not None else 0.5
                }
                results.append(result)
                
                print(f"📷 {image_path.name}: {result['prediction']} "
                      f"(Confidence: {result['confidence']:.3f})")
        
        return results

def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description='Image Forgery Detection Prediction')
    parser.add_argument('input', help='Image file or directory path')
    parser.add_argument('--output', '-o', help='Output JSON file for batch results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=" * 60)
    print("🚀 OPTIMIZED IMAGE FORGERY DETECTION")
    print("=" * 60)
    
    # Initialize predictor
    predictor = OptimizedPredictor()
    
    # Load models
    if not predictor.load_models():
        print("❌ Failed to load models. Please train first with train_optimized.py")
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
            print(f"📷 Image: {input_path.name}")
            print(f"🔍 Prediction: {result}")
            print(f"📊 Confidence: {confidence:.3f}")
            
            if confidence > 0.8:
                print("✅ High confidence")
            elif confidence > 0.6:
                print("⚠️ Medium confidence")
            else:
                print("❓ Low confidence")
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
            print(f"📷 Total Images: {total}")
            print(f"🔴 Forged: {forged} ({forged/total*100:.1f}%)")
            print(f"🟢 Authentic: {authentic} ({authentic/total*100:.1f}%)")
            print(f"📊 Average Confidence: {avg_confidence:.3f}")
            
            # Save results if output specified
            if args.output:
                import json
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"💾 Results saved to {args.output}")
        else:
            print("❌ No images processed successfully")
    
    else:
        print(f"❌ Invalid input path: {input_path}")

if __name__ == "__main__":
    main()
