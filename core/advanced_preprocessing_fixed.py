#!/usr/bin/env python3
"""
Advanced Preprocessing Module for Enhanced Feature Extraction - Fixed Version
Implements sophisticated image enhancement with robust error handling
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import scipy.ndimage as ndimage
from scipy.fftpack import fft2, fftshift
from skimage import filters, restoration, segmentation, feature
from skimage.restoration import denoise_wavelet, estimate_sigma
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class AdvancedPreprocessor:
    """Advanced preprocessing for maximum accuracy improvement with robust error handling"""
    
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size
        
    def _safe_cv2_operation(self, image, operation_func, *args, **kwargs):
        """Safely perform OpenCV operations with format validation"""
        try:
            # Ensure image is in correct format for OpenCV
            if image.dtype == np.float32 and image.max() <= 1.0:
                # Convert to uint8 for OpenCV operations
                img_uint8 = (image * 255.0).astype(np.uint8)
            else:
                img_uint8 = image.astype(np.uint8)
            
            # Ensure contiguous array
            if not img_uint8.flags['C_CONTIGUOUS']:
                img_uint8 = np.ascontiguousarray(img_uint8)
            
            # Handle shape issues for OpenCV
            if len(img_uint8.shape) == 2 and operation_func in [cv2.cvtColor]:
                # Some color operations need 3D input
                pass
            elif len(img_uint8.shape) == 3 and img_uint8.shape[2] == 1:
                # Convert single channel 3D to 2D
                img_uint8 = img_uint8[:, :, 0]
            
            # Perform operation
            result = operation_func(img_uint8, *args, **kwargs)
            
            # Convert back to float32 [0, 1] if needed
            if result.dtype == np.uint8:
                result = result.astype(np.float32) / 255.0
            
            return result
        except cv2.error as e:
            logger.warning(f"OpenCV operation {operation_func.__name__} failed: {e}")
            return image  # Return original image on failure
        except Exception as e:
            logger.warning(f"Safe CV2 operation {operation_func.__name__} failed: {e}")
            return image
    
    def apply_advanced_enhancement(self, image):
        """Apply comprehensive image enhancement pipeline with robust error handling"""
        try:
            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image.copy()
            
            # Ensure proper format and handle different image types
            if img_array.dtype == np.uint16:
                # Convert 16-bit to 8-bit
                img_array = (img_array / 65535.0 * 255.0).astype(np.uint8)
            elif img_array.dtype == np.float64:
                img_array = img_array.astype(np.float32)
            elif img_array.dtype not in [np.uint8, np.float32]:
                # Convert any other format to uint8
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255.0).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
            
            # Ensure 3 channels for RGB
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                # Convert RGBA to RGB
                img_array = img_array[:, :, :3]
            elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
                # Convert single channel 3D to RGB
                img_array = np.repeat(img_array, 3, axis=2)
            
            # Normalize to float32 [0, 1] for processing
            if img_array.dtype == np.uint8:
                img_array = img_array.astype(np.float32) / 255.0
            
            # Validate image
            if img_array.size == 0:
                logger.warning("Empty image received")
                return img_array
            
            # Apply enhancements with error handling
            try:
                img_array = self._multi_stage_denoising(img_array)
            except Exception as e:
                logger.warning(f"Denoising failed: {e}")
            
            try:
                img_array = self._adaptive_contrast_enhancement(img_array)
            except Exception as e:
                logger.warning(f"Contrast enhancement failed: {e}")
            
            try:
                img_array = self._advanced_color_correction(img_array)
            except Exception as e:
                logger.warning(f"Color correction failed: {e}")
            
            try:
                img_array = self._adaptive_sharpening(img_array)
            except Exception as e:
                logger.warning(f"Adaptive sharpening failed: {e}")
            
            # Ensure output is valid
            img_array = np.clip(img_array, 0, 1)
            
        except Exception as e:
            logger.error(f"Advanced enhancement failed completely: {e}")
            # Return simplified version of original image
            if isinstance(image, Image.Image):
                img_array = np.array(image).astype(np.float32)
            else:
                img_array = image.astype(np.float32)
            if img_array.max() > 1:
                img_array = img_array / 255.0
            img_array = np.clip(img_array, 0, 1)
        
        return img_array

    def _multi_stage_denoising(self, image):
        """Multi-stage denoising with robust error handling"""
        try:
            # Stage 1: Wavelet denoising
            try:
                # Try new API first
                sigma_est = estimate_sigma(image, channel_axis=-1, average_sigmas=True)
            except TypeError:
                try:
                    # Fallback to older API
                    sigma_est = estimate_sigma(image, multichannel=True, average_sigmas=True)
                except:
                    # Manual noise estimation as fallback
                    if len(image.shape) == 3:
                        gray = np.dot(image, [0.299, 0.587, 0.114])
                    else:
                        gray = image
                    # Simple noise estimation without OpenCV
                    sigma_est = np.std(gray)
            
            if sigma_est > 0.01:  # Only if significant noise
                try:
                    image = denoise_wavelet(image, channel_axis=-1, 
                                          convert2ycbcr=True, method='BayesShrink')
                except TypeError:
                    try:
                        image = denoise_wavelet(image, multichannel=True, 
                                              convert2ycbcr=True, method='BayesShrink')
                    except:
                        pass  # Skip wavelet denoising if it fails
            
            # Stage 2: Gaussian blur as a simple alternative to non-local means
            # This is much more robust than OpenCV's fastNlMeansDenoising
            try:
                from scipy.ndimage import gaussian_filter
                # Use scipy instead of OpenCV for better compatibility
                if len(image.shape) == 3:
                    for i in range(image.shape[2]):
                        image[:,:,i] = gaussian_filter(image[:,:,i], sigma=0.5)
                else:
                    image = gaussian_filter(image, sigma=0.5)
            except Exception as e:
                logger.warning(f"Gaussian filtering failed: {e}")
                
        except Exception as e:
            logger.warning(f"Multi-stage denoising failed: {e}")
            # Return image as-is if denoising fails
            pass
        
        return image
    
    def _adaptive_contrast_enhancement(self, image):
        """Adaptive contrast enhancement with fallback methods"""
        try:
            # Simple contrast enhancement without CLAHE to avoid OpenCV issues
            # Histogram equalization using numpy
            if len(image.shape) == 3:
                # Process each channel separately
                for i in range(3):
                    channel = image[:,:,i]
                    # Simple contrast stretching
                    p2, p98 = np.percentile(channel, (2, 98))
                    if p98 > p2:
                        image[:,:,i] = np.clip((channel - p2) / (p98 - p2), 0, 1)
            else:
                # Grayscale processing
                p2, p98 = np.percentile(image, (2, 98))
                if p98 > p2:
                    image = np.clip((image - p2) / (p98 - p2), 0, 1)
                    
        except Exception as e:
            logger.warning(f"Adaptive contrast enhancement failed: {e}")
            # Simple fallback enhancement
            image = np.clip(image * 1.1, 0, 1)
            
        return image
    
    def _advanced_color_correction(self, image):
        """Advanced color correction with error handling"""
        try:
            if len(image.shape) == 3:
                # White balance correction
                mean_rgb = np.mean(image, axis=(0, 1))
                gray_world_factor = np.mean(mean_rgb) / mean_rgb
                gray_world_factor = np.clip(gray_world_factor, 0.5, 2.0)  # Limit correction
                
                for i in range(3):
                    image[:,:,i] = np.clip(image[:,:,i] * gray_world_factor[i], 0, 1)
        except Exception as e:
            logger.warning(f"Color correction failed: {e}")
        
        return image
    
    def _adaptive_sharpening(self, image):
        """Adaptive sharpening using scipy instead of OpenCV"""
        try:
            from scipy.ndimage import gaussian_filter
            
            if len(image.shape) == 3:
                for i in range(3):
                    # Create unsharp mask
                    blurred = gaussian_filter(image[:,:,i], sigma=1.0)
                    mask = image[:,:,i] - blurred
                    image[:,:,i] = np.clip(image[:,:,i] + 0.5 * mask, 0, 1)
            else:
                blurred = gaussian_filter(image, sigma=1.0)
                mask = image - blurred
                image = np.clip(image + 0.5 * mask, 0, 1)
                
        except Exception as e:
            logger.warning(f"Adaptive sharpening failed: {e}")
        
        return image
