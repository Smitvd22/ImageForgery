import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from scipy import ndimage
import torch
from scipy.signal import wiener
from scipy.ndimage import uniform_filter, median_filter
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_advanced_augmentation_pipeline(image_size=(384, 384), p=0.5):
    """
    Advanced augmentation pipeline with forgery-aware transformations
    Designed to improve generalization while preserving forgery traces
    """
    return A.Compose([
        # Geometric transformations (light)
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=p),
        A.HorizontalFlip(p=0.3),
        
        # Optical & perspective transformations (very light to preserve forgery traces)
        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.02, p=0.2),
        A.Perspective(scale=(0.02, 0.05), p=0.2),
        
        # Color & lighting augmentations (preserve forgery color inconsistencies)
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=p),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.3),
        A.RandomGamma(gamma_limit=(95, 105), p=0.3),
        
        # Noise & quality degradation (simulate real-world conditions)
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
        A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.05, 0.15), p=0.2),
        A.MultiplicativeNoise(multiplier=(0.98, 1.02), p=0.1),
        
        # Compression artifacts (common in forgeries)
        A.ImageCompression(quality_lower=85, quality_upper=100, p=0.3),
        
        # Edge-preserving blur (preserve forgery boundaries)
        A.MotionBlur(blur_limit=3, p=0.1),
        A.GaussianBlur(blur_limit=3, p=0.1),
        
        # Normalization and tensor conversion
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_test_transform(image_size=(384, 384)):
    """Test-time transformation without augmentation"""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def adaptive_histogram_equalization(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Improves local contrast while preserving forgery artifacts
    """
    if len(img.shape) == 3:
        # Convert RGB to LAB for better color preservation
        lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_eq = clahe.apply(l)
        
        # Merge channels back
        lab_eq = cv2.merge([l_eq, a, b])
        rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        return rgb_eq.astype(np.float32) / 255.0
    else:
        # Grayscale
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        eq = clahe.apply((img * 255).astype(np.uint8))
        return eq.astype(np.float32) / 255.0

def enhance_edge_preservation(img, sigma_spatial=15, sigma_color=80):
    """
    Edge-preserving smoothing that maintains forgery boundaries
    Uses bilateral filtering to smooth while preserving important edges
    """
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Multiple scales of bilateral filtering
    filtered1 = cv2.bilateralFilter(img_uint8, 9, sigma_color, sigma_spatial)
    filtered2 = cv2.bilateralFilter(filtered1, 9, sigma_color//2, sigma_spatial//2)
    
    # Blend with original to preserve fine details
    alpha = 0.7
    enhanced = alpha * filtered2 + (1 - alpha) * img_uint8
    
    return enhanced.astype(np.float32) / 255.0

def adjust_brightness_contrast(img, alpha=1.1, beta=5):
    """
    Adjust brightness and contrast of the image
    
    Args:
        img: Input image (numpy array)
        alpha: Contrast control (1.0-3.0)
        beta: Brightness control (0-100)
    
    Returns:
        Enhanced image
    """
    # Ensure image is in proper format
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    # Apply brightness and contrast adjustment
    enhanced = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    # Convert back to float32 [0, 1] range
    return enhanced.astype(np.float32) / 255.0

def apply_sparkle_noise_suppression(img):
    """
    Enhanced custom sparkle noise suppression filter using advanced morphological operations,
    adaptive median filtering, and multi-scale filtering to reduce sensor noise, 
    compression artifacts, and forgery artifacts while preserving forgery traces
    """
    # Convert to uint8 for morphological operations
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Multi-scale morphological filtering for sparkle noise - more conservative
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),  # Smaller kernels
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    ]
    
    filtered_results = []
    for kernel in kernels:
        # Apply lighter morphological operations to preserve forgery artifacts
        opened = cv2.morphologyEx(img_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        filtered_results.append(cleaned)
    
    # Combine multi-scale results using weighted average
    weights = [0.4, 0.4, 0.2]  # More balanced weights
    combined = np.zeros_like(img_uint8, dtype=np.float32)
    for result, weight in zip(filtered_results, weights):
        combined += weight * result.astype(np.float32)
    combined = combined.astype(np.uint8)
    
    # Lighter bilateral filtering to preserve edges and artifacts
    bilateral = cv2.bilateralFilter(combined, 5, 30, 30)  # Reduced parameters
    
    # Skip non-local means denoising to preserve forgery artifacts
    # Apply very light Gaussian blur only
    final_result = cv2.GaussianBlur(bilateral, (3, 3), 0.3)
    
    return final_result.astype(np.float32) / 255.0

def detect_noise_type(img):
    """
    Enhanced noise detection for comprehensive noise type identification including:
    - Additive Gaussian noise
    - Salt-and-pepper (impulse) noise
    - Poisson (shot) noise
    - Speckle (multiplicative) noise
    - Uniform noise
    - Thermal noise
    - Quantization noise
    - ISO noise (sensor-specific)
    - Compression artifacts
    
    Returns a dictionary with noise type probabilities
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = (img * 255).astype(np.float32)
    
    # Calculate noise characteristics
    noise_metrics = {}
    
    # 1. Additive Gaussian noise detection using variance in smooth regions
    smooth_kernel = np.ones((5, 5)) / 25
    smoothed = cv2.filter2D(gray, -1, smooth_kernel)
    noise_variance = np.var(gray - smoothed)
    noise_metrics['gaussian'] = min(noise_variance / 100.0, 1.0)  # Normalize
    
    # 2. Salt-and-pepper (impulse) noise detection
    # Count isolated pixels that are significantly different from neighbors
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0
    neighbor_mean = cv2.filter2D(gray, -1, kernel / 8)
    diff = np.abs(gray - neighbor_mean)
    salt_pepper_pixels = np.sum(diff > 50) / gray.size
    noise_metrics['salt_pepper'] = min(salt_pepper_pixels * 10, 1.0)
    
    # 3. Poisson noise detection (shot noise)
    # Poisson noise has variance proportional to signal intensity
    local_mean = cv2.GaussianBlur(gray, (7, 7), 0)
    local_var = cv2.GaussianBlur((gray - local_mean) ** 2, (7, 7), 0)
    # For Poisson noise, variance should be approximately equal to mean
    poisson_ratio = np.mean(local_var / (local_mean + 1e-8))
    noise_metrics['poisson'] = min(abs(poisson_ratio - 1.0), 1.0)
    
    # 4. Speckle noise detection (multiplicative noise)
    speckle_metric = np.std(gray / (local_mean + 1e-8))
    noise_metrics['speckle'] = min(speckle_metric / 10.0, 1.0)
    
    # 5. Uniform noise detection
    hist, _ = np.histogram(gray, bins=256, range=(0, 255))
    hist_variance = np.var(hist)
    uniform_metric = 1.0 / (1.0 + hist_variance / 1000.0)  # Lower variance indicates more uniform
    noise_metrics['uniform'] = uniform_metric
    
    # 6. Thermal noise detection (pattern-based analysis)
    # Thermal noise typically shows up as random variations across the image
    high_freq_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    high_freq_response = cv2.filter2D(gray, -1, high_freq_kernel)
    thermal_metric = np.std(high_freq_response) / (np.mean(gray) + 1e-8)
    noise_metrics['thermal'] = min(thermal_metric / 5.0, 1.0)
    
    # 7. Quantization noise detection (step artifacts)
    # Look for step-like intensity transitions
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Count sharp edges (quantization steps)
    sharp_edges = np.sum(gradient_mag > np.percentile(gradient_mag, 95))
    quantization_metric = sharp_edges / gray.size
    noise_metrics['quantization'] = min(quantization_metric * 50, 1.0)
    
    # 8. ISO noise detection (high ISO sensor noise patterns)
    # ISO noise shows characteristic patterns in different frequency bands
    # Apply FFT to analyze frequency domain characteristics
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    # High frequency content analysis for ISO noise
    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2
    
    # Create high frequency mask
    y, x = np.ogrid[:h, :w]
    mask_high = ((x - center_w)**2 + (y - center_h)**2) > (min(h, w) // 4)**2
    
    high_freq_energy = np.sum(magnitude_spectrum[mask_high])
    total_energy = np.sum(magnitude_spectrum)
    iso_noise_ratio = high_freq_energy / (total_energy + 1e-8)
    noise_metrics['iso_noise'] = min(iso_noise_ratio * 10, 1.0)
    
    # 9. Compression artifacts detection (JPEG blocking, etc.)
    # Detect 8x8 block boundaries typical in JPEG compression
    block_size = 8
    block_variance = []
    
    for i in range(0, gray.shape[0] - block_size, block_size):
        for j in range(0, gray.shape[1] - block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            if block.size == block_size * block_size:
                block_variance.append(np.var(block))
    
    if block_variance:
        compression_metric = np.std(block_variance) / (np.mean(block_variance) + 1e-8)
        noise_metrics['compression'] = min(compression_metric / 2.0, 1.0)
    else:
        noise_metrics['compression'] = 0.0
    
    return noise_metrics

def apply_adaptive_gaussian_noise_suppression(img, sigma=1.0):
    """
    Advanced Gaussian noise suppression using multiple techniques:
    - Adaptive Wiener filtering
    - Non-local means denoising
    - Bilateral filtering
    - Gaussian blur with edge preservation
    """
    if len(img.shape) == 3:
        # Process each channel separately
        result = np.zeros_like(img)
        for i in range(3):
            result[:, :, i] = apply_adaptive_gaussian_noise_suppression_single_channel(img[:, :, i], sigma)
        return result
    else:
        return apply_adaptive_gaussian_noise_suppression_single_channel(img, sigma)

def apply_adaptive_gaussian_noise_suppression_single_channel(img_channel, sigma=1.0):
    """
    Apply Gaussian noise suppression to a single channel
    """
    # Convert to uint8 for processing
    img_uint8 = (img_channel * 255).astype(np.uint8)
    
    # 1. Adaptive Wiener filtering
    try:
        # Estimate noise using high-pass filtering
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        high_pass = cv2.filter2D(img_uint8.astype(np.float32), -1, kernel)
        noise_var = np.var(high_pass)
        
        # Apply Wiener filter approximation using cv2
        wiener_filtered = cv2.bilateralFilter(img_uint8, 9, sigma * 50, sigma * 50)
    except:
        wiener_filtered = img_uint8
    
    # 2. Non-local means denoising (conservative parameters to preserve details)
    nlm_filtered = cv2.fastNlMeansDenoising(img_uint8, None, h=sigma * 3, templateWindowSize=7, searchWindowSize=21)
    
    # 3. Bilateral filtering for edge-preserving smoothing
    bilateral_filtered = cv2.bilateralFilter(img_uint8, 9, sigma * 30, sigma * 30)
    
    # 4. Adaptive combination based on local image characteristics
    # Use gradient magnitude to determine which filter to prioritize
    grad_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize gradient magnitude
    grad_norm = gradient_magnitude / (np.max(gradient_magnitude) + 1e-8)
    
    # Combine filters based on local gradient (edge information)
    # High gradient areas: use bilateral (preserve edges)
    # Low gradient areas: use NLM (better noise reduction)
    edge_mask = grad_norm > 0.1
    
    result = np.zeros_like(img_uint8, dtype=np.float32)
    result[edge_mask] = bilateral_filtered[edge_mask].astype(np.float32)
    result[~edge_mask] = nlm_filtered[~edge_mask].astype(np.float32)
    
    # Light smoothing for final result
    final_result = cv2.GaussianBlur(result.astype(np.uint8), (3, 3), 0.5)
    
    return final_result.astype(np.float32) / 255.0

def apply_salt_pepper_noise_suppression(img, kernel_size=3):
    """
    Suppress salt-and-pepper noise using median filtering and morphological operations
    """
    img_uint8 = (img * 255).astype(np.uint8)
    
    if len(img.shape) == 3:
        result = np.zeros_like(img_uint8)
        for i in range(3):
            # Median filtering
            median_filtered = cv2.medianBlur(img_uint8[:, :, i], kernel_size)
            
            # Morphological operations for isolated noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            morph_filtered = cv2.morphologyEx(median_filtered, cv2.MORPH_OPEN, kernel)
            morph_filtered = cv2.morphologyEx(morph_filtered, cv2.MORPH_CLOSE, kernel)
            
            result[:, :, i] = morph_filtered
    else:
        # Median filtering
        median_filtered = cv2.medianBlur(img_uint8, kernel_size)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        result = cv2.morphologyEx(median_filtered, cv2.MORPH_OPEN, kernel)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    
    return result.astype(np.float32) / 255.0

def apply_poisson_noise_suppression(img):
    """
    Suppress Poisson (shot) noise using Anscombe transform and Gaussian denoising
    """
    # Anscombe transform: converts Poisson noise to approximately Gaussian
    img_transformed = 2 * np.sqrt(img * 255 + 3/8)
    
    # Apply Gaussian noise suppression to transformed image
    denoised_transformed = apply_adaptive_gaussian_noise_suppression(img_transformed / 255.0, sigma=1.0)
    
    # Inverse Anscombe transform
    denoised_transformed_scaled = denoised_transformed * 255
    inverse_transformed = (denoised_transformed_scaled / 2) ** 2 - 3/8
    inverse_transformed = np.maximum(inverse_transformed, 0)  # Ensure non-negative
    
    return inverse_transformed / 255.0

def apply_speckle_noise_suppression(img):
    """
    Suppress speckle (multiplicative) noise using Lee filter and adaptive smoothing
    """
    if len(img.shape) == 3:
        result = np.zeros_like(img)
        for i in range(3):
            result[:, :, i] = apply_lee_filter(img[:, :, i])
        return result
    else:
        return apply_lee_filter(img)

def apply_lee_filter(img_channel, window_size=7):
    """
    Apply Lee filter for speckle noise reduction
    """
    # Convert to appropriate range
    img_scaled = img_channel * 255
    
    # Calculate local statistics
    kernel = np.ones((window_size, window_size)) / (window_size * window_size)
    local_mean = cv2.filter2D(img_scaled, -1, kernel)
    local_mean_sq = cv2.filter2D(img_scaled ** 2, -1, kernel)
    local_var = local_mean_sq - local_mean ** 2
    
    # Lee filter formula
    # Assume noise variance (can be estimated)
    noise_var = np.var(img_scaled) * 0.1  # Estimate as 10% of image variance
    
    # Avoid division by zero
    denominator = local_var + noise_var + 1e-8
    weight = local_var / denominator
    
    # Apply filter
    filtered = local_mean + weight * (img_scaled - local_mean)
    
    return np.clip(filtered / 255.0, 0, 1)

def apply_uniform_noise_suppression(img):
    """
    Suppress uniform noise using adaptive smoothing
    """
    # Uniform noise is best handled by light smoothing
    img_uint8 = (img * 255).astype(np.uint8)
    
    if len(img.shape) == 3:
        result = np.zeros_like(img_uint8)
        for i in range(3):
            # Light Gaussian smoothing
            result[:, :, i] = cv2.GaussianBlur(img_uint8[:, :, i], (5, 5), 1.0)
    else:
        result = cv2.GaussianBlur(img_uint8, (5, 5), 1.0)
    
    return result.astype(np.float32) / 255.0

def apply_thermal_noise_suppression(img):
    """
    Suppress thermal noise using temperature-adaptive filtering
    Thermal noise is typically low-frequency and can be reduced with bilateral filtering
    """
    img_uint8 = (img * 255).astype(np.uint8)
    
    if len(img.shape) == 3:
        result = np.zeros_like(img_uint8)
        for i in range(3):
            # Apply bilateral filter with parameters tuned for thermal noise
            filtered = cv2.bilateralFilter(img_uint8[:, :, i], 9, 40, 40)
            # Follow with light Gaussian to smooth remaining artifacts
            result[:, :, i] = cv2.GaussianBlur(filtered, (3, 3), 0.8)
    else:
        filtered = cv2.bilateralFilter(img_uint8, 9, 40, 40)
        result = cv2.GaussianBlur(filtered, (3, 3), 0.8)
    
    return result.astype(np.float32) / 255.0

def apply_quantization_noise_suppression(img):
    """
    Suppress quantization noise using edge-preserving smoothing
    Quantization noise appears as step artifacts and false contours
    """
    img_uint8 = (img * 255).astype(np.uint8)
    
    if len(img.shape) == 3:
        result = np.zeros_like(img_uint8)
        for i in range(3):
            # Use morphological smoothing to reduce step artifacts
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            opened = cv2.morphologyEx(img_uint8[:, :, i], cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            
            # Apply bilateral filter to preserve edges while smoothing
            result[:, :, i] = cv2.bilateralFilter(closed, 5, 25, 25)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(img_uint8, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        result = cv2.bilateralFilter(closed, 5, 25, 25)
    
    return result.astype(np.float32) / 255.0

def apply_iso_noise_suppression(img):
    """
    Suppress ISO noise (high ISO sensor noise) using frequency domain filtering
    ISO noise typically has characteristic patterns in high-frequency regions
    """
    if len(img.shape) == 3:
        result = np.zeros_like(img)
        for i in range(3):
            result[:, :, i] = apply_iso_noise_suppression_single_channel(img[:, :, i])
        return result
    else:
        return apply_iso_noise_suppression_single_channel(img)

def apply_iso_noise_suppression_single_channel(img_channel):
    """
    Apply ISO noise suppression to a single channel using frequency domain methods
    """
    # Convert to appropriate range
    img_scaled = (img_channel * 255).astype(np.uint8)
    
    # Apply Non-Local Means denoising with parameters optimized for ISO noise
    denoised = cv2.fastNlMeansDenoising(img_scaled, None, h=8, templateWindowSize=7, searchWindowSize=21)
    
    # Follow with adaptive bilateral filtering
    bilateral = cv2.bilateralFilter(denoised, 7, 35, 35)
    
    # Apply light Gaussian blur to smooth remaining high-frequency noise
    final_result = cv2.GaussianBlur(bilateral, (3, 3), 0.6)
    
    return final_result.astype(np.float32) / 255.0

def apply_compression_artifact_suppression(img):
    """
    Suppress compression artifacts (JPEG blocking, ringing, etc.)
    """
    img_uint8 = (img * 255).astype(np.uint8)
    
    if len(img.shape) == 3:
        result = np.zeros_like(img_uint8)
        for i in range(3):
            # Use bilateral filter to reduce blocking artifacts
            bilateral = cv2.bilateralFilter(img_uint8[:, :, i], 9, 50, 50)
            
            # Apply morphological operations to reduce ringing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            morph = cv2.morphologyEx(bilateral, cv2.MORPH_CLOSE, kernel)
            
            # Final smoothing with Gaussian filter
            result[:, :, i] = cv2.GaussianBlur(morph, (3, 3), 0.5)
    else:
        bilateral = cv2.bilateralFilter(img_uint8, 9, 50, 50)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph = cv2.morphologyEx(bilateral, cv2.MORPH_CLOSE, kernel)
        result = cv2.GaussianBlur(morph, (3, 3), 0.5)
    
    return result.astype(np.float32) / 255.0

def apply_enhanced_additive_gaussian_suppression(img, sigma=1.0, method='adaptive'):
    """
    Enhanced additive Gaussian noise suppression with multiple methods:
    - Adaptive Wiener filtering
    - Non-local means denoising  
    - Bilateral filtering
    - Total variation denoising
    - BM3D-inspired filtering
    """
    if len(img.shape) == 3:
        result = np.zeros_like(img)
        for i in range(3):
            result[:, :, i] = apply_enhanced_gaussian_single_channel(img[:, :, i], sigma, method)
        return result
    else:
        return apply_enhanced_gaussian_single_channel(img, sigma, method)

def apply_enhanced_gaussian_single_channel(img_channel, sigma=1.0, method='adaptive'):
    """
    Enhanced Gaussian noise suppression for single channel with multiple denoising strategies
    """
    img_uint8 = (img_channel * 255).astype(np.uint8)
    
    if method == 'adaptive':
        # Adaptive approach: combine multiple filters based on local characteristics
        
        # 1. Non-local means for texture preservation
        nlm_filtered = cv2.fastNlMeansDenoising(img_uint8, None, h=sigma * 4, 
                                                templateWindowSize=7, searchWindowSize=21)
        
        # 2. Bilateral for edge preservation
        bilateral_filtered = cv2.bilateralFilter(img_uint8, 9, sigma * 40, sigma * 40)
        
        # 3. Gaussian for smooth regions
        gaussian_filtered = cv2.GaussianBlur(img_uint8, (5, 5), sigma)
        
        # 4. Edge-based adaptive combination
        grad_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_norm = gradient_magnitude / (np.max(gradient_magnitude) + 1e-8)
        
        # Create masks for different regions
        edge_mask = grad_norm > 0.15  # Strong edges
        texture_mask = (grad_norm > 0.05) & (grad_norm <= 0.15)  # Texture regions
        smooth_mask = grad_norm <= 0.05  # Smooth regions
        
        # Combine filters based on region type
        result = np.zeros_like(img_uint8, dtype=np.float32)
        result[edge_mask] = bilateral_filtered[edge_mask].astype(np.float32)
        result[texture_mask] = nlm_filtered[texture_mask].astype(np.float32)
        result[smooth_mask] = gaussian_filtered[smooth_mask].astype(np.float32)
        
    elif method == 'wiener':
        # Wiener filter approximation
        try:
            # Estimate noise variance
            laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            laplacian = cv2.filter2D(img_uint8.astype(np.float32), -1, laplacian_kernel)
            noise_var = np.var(laplacian) * (sigma ** 2)
            
            # Apply bilateral filter as Wiener approximation
            result = cv2.bilateralFilter(img_uint8, 9, sigma * 50, sigma * 50).astype(np.float32)
        except:
            result = cv2.GaussianBlur(img_uint8, (5, 5), sigma).astype(np.float32)
    
    elif method == 'tv_denoising':
        # Total Variation denoising approximation using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(img_uint8, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        result = cv2.bilateralFilter(closed, 7, sigma * 30, sigma * 30).astype(np.float32)
    
    else:  # default to bilateral
        result = cv2.bilateralFilter(img_uint8, 9, sigma * 30, sigma * 30).astype(np.float32)
    
    return np.clip(result / 255.0, 0, 1)

def apply_comprehensive_noise_suppression(img, preserve_edges=True):
    """
    Enhanced comprehensive noise suppression that detects and handles multiple noise types:
    - Additive Gaussian noise
    - Salt-and-pepper (impulse) noise  
    - Poisson (shot) noise
    - Speckle (multiplicative) noise
    - Uniform noise
    - Thermal noise
    - Quantization noise
    - ISO noise (sensor-specific)
    - Compression artifacts
    """
    # Detect noise types
    noise_metrics = detect_noise_type(img)
    
    # Apply appropriate filters based on detected noise
    result = img.copy()
    
    # Apply enhanced Gaussian noise suppression if significant Gaussian noise detected
    if noise_metrics['gaussian'] > 0.25:
        sigma = min(noise_metrics['gaussian'] * 2, 2.0)
        result = apply_enhanced_additive_gaussian_suppression(result, sigma=sigma, method='adaptive')
    
    # Apply salt-and-pepper suppression if detected
    if noise_metrics['salt_pepper'] > 0.15:
        kernel_size = 3 if noise_metrics['salt_pepper'] < 0.4 else 5
        result = apply_salt_pepper_noise_suppression(result, kernel_size=kernel_size)
    
    # Apply Poisson noise suppression if detected
    if noise_metrics['poisson'] > 0.35:
        result = apply_poisson_noise_suppression(result)
    
    # Apply speckle noise suppression if detected
    if noise_metrics['speckle'] > 0.25:
        result = apply_speckle_noise_suppression(result)
    
    # Apply uniform noise suppression if detected
    if noise_metrics['uniform'] > 0.5:
        result = apply_uniform_noise_suppression(result)
    
    # Apply thermal noise suppression if detected
    if noise_metrics.get('thermal', 0) > 0.3:
        result = apply_thermal_noise_suppression(result)
    
    # Apply quantization noise suppression if detected
    if noise_metrics.get('quantization', 0) > 0.4:
        result = apply_quantization_noise_suppression(result)
    
    # Apply ISO noise suppression if detected
    if noise_metrics.get('iso_noise', 0) > 0.3:
        result = apply_iso_noise_suppression(result)
    
    # Apply compression artifact suppression if detected
    if noise_metrics.get('compression', 0) > 0.4:
        result = apply_compression_artifact_suppression(result)
    
    # Final edge-preserving smoothing if requested
    if preserve_edges:
        if len(result.shape) == 3:
            result_uint8 = (result * 255).astype(np.uint8)
            result = cv2.bilateralFilter(result_uint8, 5, 15, 15).astype(np.float32) / 255.0
        else:
            result_uint8 = (result * 255).astype(np.uint8)
            result = cv2.bilateralFilter(result_uint8, 5, 15, 15).astype(np.float32) / 255.0
    
    return result

def enhance_contrast_adaptive(img):
    """
    Advanced adaptive contrast enhancement using multiple techniques:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Adaptive gamma correction
    - Local contrast normalization
    """
    # Convert to LAB color space for better perceptual processing
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel with optimized parameters
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    
    # Adaptive gamma correction
    gamma = compute_adaptive_gamma(enhanced)
    gamma_corrected = np.power(enhanced, gamma)
    
    # Local contrast normalization
    locally_normalized = apply_local_contrast_normalization(gamma_corrected)
    
    return locally_normalized

def compute_adaptive_gamma(img):
    """
    Compute adaptive gamma correction value based on image histogram
    """
    # Convert to grayscale for gamma calculation
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Calculate mean intensity
    mean_intensity = np.mean(gray) / 255.0
    
    # Adaptive gamma based on mean intensity
    if mean_intensity < 0.3:
        gamma = 0.7  # Brighten dark images
    elif mean_intensity > 0.7:
        gamma = 1.3  # Darken bright images
    else:
        gamma = 1.0  # Neutral for well-exposed images
    
    return gamma

def apply_local_contrast_normalization(img, kernel_size=9):
    """
    Apply local contrast normalization to enhance local features
    """
    # Convert to grayscale for processing
    if len(img.shape) == 3:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        # Apply Gaussian filter for local mean
        local_mean = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        # Calculate local standard deviation
        local_var = cv2.GaussianBlur(gray * gray, (kernel_size, kernel_size), 0) - local_mean * local_mean
        local_std = np.sqrt(np.maximum(local_var, 1e-8))
        
        # Normalize
        normalized_gray = (gray - local_mean) / (local_std + 1e-8)
        
        # Apply normalization to original color image
        result = img.copy()
        for channel in range(img.shape[2]):
            channel_img = img[:, :, channel]
            channel_mean = cv2.GaussianBlur(channel_img, (kernel_size, kernel_size), 0)
            channel_var = cv2.GaussianBlur(channel_img * channel_img, (kernel_size, kernel_size), 0) - channel_mean * channel_mean
            channel_std = np.sqrt(np.maximum(channel_var, 1e-8))
            result[:, :, channel] = (channel_img - channel_mean) / (channel_std + 1e-8)
        
        # Rescale to [0, 1]
        result = (result - result.min()) / (result.max() - result.min() + 1e-8)
        return result
    else:
        return img

def extract_forgery_specific_features(img):
    """
    Extract forgery-specific features that help distinguish authentic from forged images
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (img * 255).astype(np.uint8)
    
    features = []
    
    # 1. Edge inconsistency features
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges > 0)
    features.append(edge_density)
    
    # 2. Texture analysis using Local Binary Patterns
    try:
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
        features.extend(lbp_hist)
    except ImportError:
        # Fallback: simple texture features if scikit-image not available
        # Calculate simple texture variance in local patches
        for i in range(0, gray.shape[0], 16):
            for j in range(0, gray.shape[1], 16):
                patch = gray[i:i+16, j:j+16]
                if patch.size > 0:
                    features.append(np.var(patch))
                    if len(features) >= 10:  # Limit to 10 texture features
                        break
            if len(features) >= 10:
                break
        # Pad to 10 features if needed
        while len(features) < 11:  # 1 edge + 10 texture
            features.append(0.0)
    
    # 3. Frequency domain analysis
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    
    # High frequency energy ratio
    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2
    high_freq_mask = np.zeros((h, w))
    cv2.circle(high_freq_mask, (center_w, center_h), min(h, w) // 4, 1, -1)
    high_freq_mask = 1 - high_freq_mask
    
    high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask)
    total_energy = np.sum(magnitude_spectrum)
    high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
    features.append(high_freq_ratio)
    
    # 4. JPEG compression artifacts detection
    # Calculate blockiness metric
    block_size = 8
    blockiness = 0
    count = 0
    for i in range(0, gray.shape[0] - block_size, block_size):
        for j in range(0, gray.shape[1] - block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size].astype(np.float32)
            block_var = np.var(block)
            blockiness += block_var
            count += 1
    
    if count > 0:
        blockiness /= count
    features.append(blockiness)
    
    # 5. Color inconsistency features (for RGB images)
    if len(img.shape) == 3:
        # Calculate color channel correlations
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        rg_corr = np.corrcoef(r.flatten(), g.flatten())[0, 1]
        rb_corr = np.corrcoef(r.flatten(), b.flatten())[0, 1]
        gb_corr = np.corrcoef(g.flatten(), b.flatten())[0, 1]
        
        features.extend([rg_corr, rb_corr, gb_corr])
        
        # Color channel variance ratios
        r_var, g_var, b_var = np.var(r), np.var(g), np.var(b)
        total_var = r_var + g_var + b_var
        if total_var > 0:
            features.extend([r_var/total_var, g_var/total_var, b_var/total_var])
        else:
            features.extend([0.33, 0.33, 0.33])
    
    return np.array(features, dtype=np.float32)

def preprocess_image(image_path, size=(256, 256), apply_augmentation=False, comprehensive_noise_suppression=True):
    """
    Enhanced preprocessing pipeline for image forgery detection with all required steps:
    1. Brightness and contrast adjustment
    2. Resolution normalization and resizing
    3. Comprehensive noise suppression (Gaussian, salt-pepper, Poisson, speckle, uniform)
    4. Custom sparkle noise suppression filter
    5. Forgery-specific feature extraction
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Step 1: Brightness & contrast adjustment with adaptive enhancement
    # Initial brightness/contrast adjustment - more conservative
    img = cv2.convertScaleAbs(img, alpha=1.1, beta=5)  # Reduced parameters
    img = img.astype(np.float32) / 255.0
    
    # Step 2: Apply comprehensive noise suppression if enabled
    if comprehensive_noise_suppression:
        img = apply_comprehensive_noise_suppression(img, preserve_edges=True)
    
    # Step 3: Apply lighter adaptive contrast enhancement
    img = enhance_contrast_adaptive(img)
    
    # Step 4: Apply conservative sparkle noise suppression (for remaining sensor noise)
    img = apply_sparkle_noise_suppression(img)
    
    # Step 5: Normalize resolution and resize image with high-quality interpolation
    img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    
    # Additional preprocessing for training augmentation
    if apply_augmentation:
        img = apply_training_augmentations(img, size)
    
    # Final normalization to [0, 1] range
    img = np.clip(img, 0, 1)
    
    # Convert to tensor with proper channel ordering and normalization
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    return transform(img.astype(np.float32))

def apply_training_augmentations(img, size):
    """
    Enhanced training-time augmentations including comprehensive realistic noise scenarios
    """
    # Random horizontal flip (50% chance)
    if np.random.random() > 0.5:
        img = cv2.flip(img, 1)
    
    # Random rotation (-15 to 15 degrees)
    angle = np.random.uniform(-15, 15)
    center = (size[0] // 2, size[1] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, rotation_matrix, size)
    
    # Random brightness adjustment
    brightness_factor = np.random.uniform(0.8, 1.2)
    img = np.clip(img * brightness_factor, 0, 1)
    
    # Random contrast adjustment
    contrast_factor = np.random.uniform(0.8, 1.2)
    img = np.clip((img - 0.5) * contrast_factor + 0.5, 0, 1)
    
    # Enhanced noise augmentation with comprehensive noise types
    noise_prob = np.random.random()
    
    if noise_prob > 0.65:  # 35% chance of adding noise
        noise_type = np.random.choice([
            'gaussian', 'salt_pepper', 'poisson', 'speckle', 'uniform',
            'thermal', 'quantization', 'iso_noise', 'compression'
        ])
        
        if noise_type == 'gaussian':
            # Enhanced additive Gaussian noise
            sigma = np.random.uniform(0.003, 0.025)
            noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)
            
        elif noise_type == 'salt_pepper':
            # Salt and pepper noise
            prob = np.random.uniform(0.001, 0.015)
            noise_mask = np.random.random(img.shape[:2])
            salt_mask = noise_mask < prob / 2
            pepper_mask = noise_mask > (1 - prob / 2)
            
            if len(img.shape) == 3:
                for i in range(3):
                    img[:, :, i][salt_mask] = 1.0
                    img[:, :, i][pepper_mask] = 0.0
            else:
                img[salt_mask] = 1.0
                img[pepper_mask] = 0.0
                
        elif noise_type == 'poisson':
            # Poisson noise (shot noise)
            # Scale image to appropriate range for Poisson
            scaled = img * 255
            noisy = np.random.poisson(scaled).astype(np.float32)
            img = np.clip(noisy / 255.0, 0, 1)
            
        elif noise_type == 'speckle':
            # Speckle (multiplicative) noise
            noise = np.random.randn(*img.shape) * 0.08
            img = np.clip(img + img * noise, 0, 1)
            
        elif noise_type == 'uniform':
            # Uniform noise
            noise_range = np.random.uniform(0.008, 0.04)
            noise = np.random.uniform(-noise_range, noise_range, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)
            
        elif noise_type == 'thermal':
            # Thermal noise (low-frequency pattern)
            h, w = img.shape[:2]
            thermal_pattern = np.random.randn(h//4, w//4) * 0.02
            thermal_pattern = cv2.resize(thermal_pattern, (w, h), interpolation=cv2.INTER_LINEAR)
            
            if len(img.shape) == 3:
                for i in range(3):
                    img[:, :, i] = np.clip(img[:, :, i] + thermal_pattern, 0, 1)
            else:
                img = np.clip(img + thermal_pattern, 0, 1)
                
        elif noise_type == 'quantization':
            # Quantization noise (reduce bit depth and restore)
            bit_depth = np.random.choice([6, 7])  # Simulate reduced bit depth
            levels = 2 ** bit_depth
            quantized = np.round(img * (levels - 1)) / (levels - 1)
            img = np.clip(quantized, 0, 1)
            
        elif noise_type == 'iso_noise':
            # ISO noise (mixed frequency noise pattern)
            # High-frequency component
            hf_noise = np.random.randn(*img.shape) * 0.015
            # Low-frequency component  
            h, w = img.shape[:2]
            lf_noise = np.random.randn(h//8, w//8) * 0.01
            lf_noise = cv2.resize(lf_noise, (w, h), interpolation=cv2.INTER_CUBIC)
            
            if len(img.shape) == 3:
                for i in range(3):
                    combined_noise = hf_noise[:, :, i] + lf_noise
                    img[:, :, i] = np.clip(img[:, :, i] + combined_noise, 0, 1)
            else:
                combined_noise = hf_noise + lf_noise
                img = np.clip(img + combined_noise, 0, 1)
                
        elif noise_type == 'compression':
            # Simulate compression artifacts
            # Convert to uint8, apply JPEG-like compression simulation
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Simulate DCT block artifacts by applying smoothing in 8x8 blocks
            h, w = img_uint8.shape[:2]
            for i in range(0, h-8, 8):
                for j in range(0, w-8, 8):
                    if len(img_uint8.shape) == 3:
                        for c in range(3):
                            block = img_uint8[i:i+8, j:j+8, c].astype(np.float32)
                            # Add slight quantization and smoothing
                            block = np.round(block / 4) * 4  # Quantize
                            img_uint8[i:i+8, j:j+8, c] = block.astype(np.uint8)
                    else:
                        block = img_uint8[i:i+8, j:j+8].astype(np.float32)
                        block = np.round(block / 4) * 4
                        img_uint8[i:i+8, j:j+8] = block.astype(np.uint8)
            
            img = img_uint8.astype(np.float32) / 255.0
    
    return img

def preprocess_batch(image_paths, size=(256, 256), apply_augmentation=False, comprehensive_noise_suppression=True):
    """
    Batch preprocessing for multiple images with comprehensive noise handling
    """
    batch = []
    for path in image_paths:
        try:
            img = preprocess_image(path, size, apply_augmentation, comprehensive_noise_suppression)
            batch.append(img)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    if batch:
        return torch.stack(batch)
    else:
        return torch.empty(0, 3, size[0], size[1])


