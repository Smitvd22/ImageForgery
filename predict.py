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
                    # Ensure all values are native Python types for JSON serialization
                    result = {
                        'image': str(image_path.name),
                        'path': str(image_path),
                        'prediction': 'Forged' if int(prediction) == 1 else 'Authentic',
                        'label': int(prediction),
                        'confidence': float(probability) if probability is not None else 0.5,
                        'dataset': str(self.dataset.upper())
                    }
                    results.append(result)
                    print(f"📷 {result['image']}: {result['prediction']} (Confidence: {result['confidence']:.3f})")
        
        return results


def generate_test_predictions_csv(dataset):
    """Generate a CSV of predictions for a given dataset"""
    import csv
    input_csv = input("Enter the csv file: ")
    output_csv = f'./results/test_predictions.csv'
    os.makedirs(f'./results', exist_ok=True)
    predictor = OptimizedPredictor(dataset=dataset)
    if not predictor.load_models():
        print(f"❌ Failed to load models for {dataset}")
        return
    with open(input_csv, 'r', newline='') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ['filepath', 'actual_label', 'predicted_label']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            filepath = row['filepath']
            actual_label = 'Authentic' if row['label'] == '0' else 'Forged'
            pred, _ = predictor.predict(filepath)
            predicted_label = 'Authentic' if pred == 0 else 'Forged'
            writer.writerow({'filepath': filepath, 'actual_label': actual_label, 'predicted_label': predicted_label})
    print(f"✅ Test predictions CSV saved to {output_csv}")

def create_filename_blind_test(dataset):
    """Create a test where filenames don't reveal the class"""
    import pandas as pd
    from pathlib import Path
    import shutil
    import os
    print("="*80)
    print("FILENAME-BLIND TEST")
    print("="*80)
    csv_path = f"./data/{dataset}_labels.csv"
    df = pd.read_csv(csv_path)
    test_dir = Path("./temp_blind_test")
    test_dir.mkdir(exist_ok=True)
    authentic_samples = df[df['label'] == 0].sample(n=5)
    forged_samples = df[df['label'] == 1].sample(n=5)
    test_cases = []
    print("Creating test files with neutral names...")
    for i, (_, row) in enumerate(authentic_samples.iterrows()):
        if os.path.exists(row['filepath']):
            original_ext = Path(row['filepath']).suffix
            new_name = f"image_{i+1:03d}{original_ext}"
            new_path = test_dir / new_name
            shutil.copy2(row['filepath'], new_path)
            test_cases.append({
                'new_name': new_name,
                'new_path': str(new_path),
                'original_name': row['filename'],
                'true_label': 0,
                'true_class': 'Authentic'
            })
            print(f"  {row['filename']} -> {new_name} (Authentic)")
    for i, (_, row) in enumerate(forged_samples.iterrows()):
        if os.path.exists(row['filepath']):
            original_ext = Path(row['filepath']).suffix
            new_name = f"image_{i+6:03d}{original_ext}"
            new_path = test_dir / new_name
            shutil.copy2(row['filepath'], new_path)
            test_cases.append({
                'new_name': new_name,
                'new_path': str(new_path),
                'original_name': row['filename'],
                'true_label': 1,
                'true_class': 'Forged'
            })
            print(f"  {row['filename']} -> {new_name} (Forged)")
    return test_cases, test_dir

def run_blind_predictions(test_cases, dataset):
    """Run predictions on renamed files"""
    print(f"\n{'='*80}")
    print("RUNNING BLIND PREDICTIONS")
    print("="*80)
    predictor = OptimizedPredictor(dataset)
    if not predictor.load_models():
        print("❌ Failed to load models")
        return 0.0
    print("\nPredicting on renamed files (no filename cues):")
    print("-" * 80)
    correct_predictions = 0
    total_predictions = 0
    for case in test_cases:
        try:
            prediction, probability = predictor.predict(case['new_path'])
            if prediction is not None:
                predicted_class = 'Forged' if prediction == 1 else 'Authentic'
                is_correct = prediction == case['true_label']
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
                status_icon = "✅" if is_correct else "❌"
                confidence = probability if probability is not None else 0.5
                print(f"{status_icon} {case['new_name']:<15} | True: {case['true_class']:<9} | "
                      f"Pred: {predicted_class:<9} | Conf: {confidence:.3f}")
                print(f"   Original: {case['original_name']}")
                print()
            else:
                print(f"❌ Failed to predict {case['new_name']}")
        except Exception as e:
            print(f"❌ Error predicting {case['new_name']}: {e}")
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print("="*80)
        print(f"BLIND TEST RESULTS:")
        print(f"Correct predictions: {correct_predictions}/{total_predictions}")
        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print("="*80)
        if accuracy >= 0.8:
            print("✅ EXCELLENT: Model is truly analyzing image content!")
        elif accuracy >= 0.6:
            print("⚠️ MODERATE: Model shows some real learning but may have issues")
        else:
            print("❌ POOR: Model appears to rely heavily on filename patterns!")
        return accuracy
    else:
        print("❌ No successful predictions made")
        return 0.0

def cleanup_test_files(test_dir):
    """Clean up temporary test files"""
    import shutil
    try:
        shutil.rmtree(test_dir)
        print(f"\n🧹 Cleaned up test directory: {test_dir}")
    except Exception as e:
        print(f"⚠️ Could not clean up {test_dir}: {e}")

def run_blind_test(dataset):
    """Main function to run the filename-blind test"""
    print("🔍 TESTING IF MODEL USES IMAGE CONTENT OR FILENAME PATTERNS")
    print("This test copies images with neutral names to remove filename cues.")
    print()
    try:
        test_cases, test_dir = create_filename_blind_test(dataset)
        if not test_cases:
            print("❌ No test cases created")
            return
        accuracy = run_blind_predictions(test_cases, dataset)
        if accuracy is None:
            accuracy = 0.0
        print("\n" + "="*80)
        print("FINAL CONCLUSION")
        print("="*80)
        if accuracy >= 0.8:
            print("🎉 CONCLUSION: Your model is LEGITIMATELY LEARNING!")
        elif accuracy >= 0.6:
            print("🤔 CONCLUSION: Model shows MIXED BEHAVIOR")
        else:
            print("🚨 CONCLUSION: Model is primarily using FILENAME PATTERNS!")
    finally:
        if 'test_dir' in locals():
            cleanup_test_files(test_dir)


def option_palette():
    """Interactive option palette for user selection"""
    print("\n=== Image Forgery Detection ===")
    print("Select an option:")
    print("1. Predict a single image or folder")
    print("2. Generate test predictions CSV")
    print("3. Run filename-blind test")
    choice = input("Enter choice [1-3]: ").strip()
    return choice

dataset = ACTIVE_DATASET

def main():
    choice = option_palette()
    if choice == '1':
        path = input("Enter image file or folder path: ").strip()
        if not path:
            print("No path provided.")
        predictor = OptimizedPredictor(dataset=dataset)
        if not predictor.load_models():
            print(f"Failed to load models for {dataset}.")
        from pathlib import Path
        input_path = Path(path)
        if input_path.is_file():
            prediction, probability = predictor.predict(str(input_path))
            if prediction is not None:
                result = 'Forged' if prediction == 1 else 'Authentic'
                confidence = probability if probability is not None else 0.5
                print(f"\nResult: {result} (Confidence: {confidence:.3f})")
            else:
                print("Failed to process image.")
        elif input_path.is_dir():
            results = predictor.predict_batch(str(input_path))
            if results:
                total = len(results)
                forged = sum(1 for r in results if r['label'] == 1)
                authentic = total - forged
                avg_confidence = sum(r['confidence'] for r in results) / total
                print(f"\nBatch Results: {total} images")
                print(f"Forged: {forged} ({forged/total*100:.1f}%)")
                print(f"Authentic: {authentic} ({authentic/total*100:.1f}%)")
                print(f"Average Confidence: {avg_confidence:.3f}")
            else:
                print("No images processed successfully.")
        else:
            print("Invalid path.")
    elif choice == '2':
        generate_test_predictions_csv(dataset)
    elif choice == '3':
        run_blind_test(dataset)
    else:
        print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
