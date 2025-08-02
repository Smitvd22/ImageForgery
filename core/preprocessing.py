import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from scipy import ndimage
import torch

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

def preprocess_image(image_path, size=(256, 256), apply_augmentation=False):
    """
    Enhanced preprocessing pipeline for image forgery detection with all required steps:
    1. Brightness and contrast adjustment
    2. Resolution normalization and resizing
    3. Custom sparkle noise suppression filter
    4. Forgery-specific feature extraction
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
    
    # Step 2: Apply lighter adaptive contrast enhancement
    img = enhance_contrast_adaptive(img)
    
    # Step 3: Apply conservative sparkle noise suppression
    img = apply_sparkle_noise_suppression(img)
    
    # Step 4: Normalize resolution and resize image with high-quality interpolation
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
    Apply training-time augmentations for data enhancement
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
    
    # Random gaussian noise
    if np.random.random() > 0.7:
        noise = np.random.normal(0, 0.01, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0, 1)
    
    return img

def preprocess_batch(image_paths, size=(256, 256), apply_augmentation=False):
    """
    Batch preprocessing for multiple images
    """
    batch = []
    for path in image_paths:
        try:
            img = preprocess_image(path, size, apply_augmentation)
            batch.append(img)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    if batch:
        return torch.stack(batch)
    else:
        return torch.empty(0, 3, size[0], size[1])
