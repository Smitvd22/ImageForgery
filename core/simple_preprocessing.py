#!/usr/bin/env python3
"""
Simple Robust Preprocessing Module
Fallback preprocessing without complex OpenCV operations that cause warnings
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging

logger = logging.getLogger(__name__)

class SimpleRobustPreprocessor:
    """Simple preprocessing without problematic OpenCV operations"""
    
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size
        
    def apply_advanced_enhancement(self, image):
        """Apply robust enhancement without problematic OpenCV operations"""
        try:
            # Convert to proper format
            if isinstance(image, Image.Image):
                img_array = np.array(image).astype(np.float32) / 255.0
            else:
                img_array = image.astype(np.float32)
                if img_array.max() > 1:
                    img_array = img_array / 255.0
            
            # Validate image
            if img_array.size == 0:
                logger.warning("Empty image received")
                return img_array
            
            # Simple enhancements that don't cause OpenCV issues
            
            # 1. Simple Gaussian blur for noise reduction
            img_array = cv2.GaussianBlur(img_array, (3, 3), 0.5)
            
            # 2. Histogram equalization using PIL (safer than OpenCV CLAHE)
            if len(img_array.shape) == 3:
                # Convert to PIL for safe processing
                pil_image = Image.fromarray((img_array * 255).astype(np.uint8))
                
                # Enhance contrast
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.2)
                
                # Enhance sharpness
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(1.1)
                
                # Convert back to numpy
                img_array = np.array(pil_image).astype(np.float32) / 255.0
            
            # 3. Simple normalization
            img_array = np.clip(img_array, 0, 1)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Simple enhancement failed: {e}")
            # Return original image if all processing fails
            if isinstance(image, Image.Image):
                img_array = np.array(image).astype(np.float32) / 255.0
            else:
                img_array = image.astype(np.float32)
                if img_array.max() > 1:
                    img_array = img_array / 255.0
            return np.clip(img_array, 0, 1)
