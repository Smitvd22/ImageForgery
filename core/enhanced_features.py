#!/usr/bin/env python3
"""
Enhanced Feature Extraction Module
Comprehensive feature extraction for maximum accuracy improvement
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from scipy import stats, signal
from scipy.fftpack import fft2, dct
from skimage import measure, filters, segmentation, morphology
from skimage.feature import hog, local_binary_pattern
import warnings
warnings.filterwarnings('ignore')

try:
    import pywt
    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False

class EnhancedFeatureExtractor:
    """Enhanced feature extractor with comprehensive analysis"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def extract_comprehensive_features(self, image):
        """Extract all types of features comprehensively"""
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        features = []
        
        # 1. Enhanced statistical features
        features.extend(self.extract_enhanced_statistical_features(img_array))
        
        # 2. Advanced color features
        features.extend(self.extract_advanced_color_features(img_array))
        
        # 3. Texture and pattern features
        features.extend(self.extract_texture_patterns(img_array))
        
        # 4. Frequency domain features
        features.extend(self.extract_frequency_domain_features(img_array))
        
        # 5. Wavelet features
        if WAVELET_AVAILABLE:
            features.extend(self.extract_wavelet_features(img_array))
        else:
            features.extend([0] * 30)  # Placeholder
        
        # 6. Edge and contour features
        features.extend(self.extract_edge_contour_features(img_array))
        
        # 7. Geometric features
        features.extend(self.extract_geometric_features(img_array))
        
        # 8. Noise and quality features
        features.extend(self.extract_noise_quality_features(img_array))
        
        # 9. Gradient and orientation features
        features.extend(self.extract_gradient_features(img_array))
        
        # 10. Advanced forgery-specific features
        features.extend(self.extract_forgery_specific_features(img_array))
        
        return np.array(features, dtype=np.float32)
    
    def extract_enhanced_statistical_features(self, image):
        """Extract enhanced statistical features from all channels"""
        features = []
        
        # Convert to different color spaces
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            
            channels = [
                image[:,:,0], image[:,:,1], image[:,:,2],  # RGB
                hsv[:,:,0], hsv[:,:,1], hsv[:,:,2],        # HSV
                lab[:,:,0], lab[:,:,1], lab[:,:,2],        # LAB
                yuv[:,:,0], yuv[:,:,1], yuv[:,:,2],        # YUV
                gray                                        # Grayscale
            ]
        else:
            gray = image
            channels = [gray]
        
        # Extract comprehensive statistics for each channel
        for channel in channels:
            channel = channel.astype(np.float32)
            
            # Basic statistics
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.median(channel),
                np.min(channel),
                np.max(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75),
                np.percentile(channel, 10),
                np.percentile(channel, 90),
                np.percentile(channel, 5),
                np.percentile(channel, 95)
            ])
            
            # Advanced statistics
            features.extend([
                stats.skew(channel.flatten()),
                stats.kurtosis(channel.flatten()),
                np.var(channel),
                np.ptp(channel),  # Peak-to-peak
                stats.entropy(channel.flatten()),
                np.sum(channel),
                np.prod(channel.shape)  # Number of pixels
            ])
            
            # Histogram features
            hist, _ = np.histogram(channel, bins=64, range=(0, 256))
            hist = hist.astype(np.float32)
            hist /= (np.sum(hist) + 1e-8)
            
            features.extend([
                np.mean(hist),
                np.std(hist),
                np.max(hist),
                np.argmax(hist),  # Mode
                len(hist[hist > 0.01])  # Number of significant bins
            ])
        
        return features
    
    def extract_advanced_color_features(self, image):
        """Extract advanced color analysis features"""
        features = []
        
        if len(image.shape) == 3:
            # Color distribution analysis
            r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
            
            # Channel correlations
            features.extend([
                np.corrcoef(r.flatten(), g.flatten())[0,1],
                np.corrcoef(r.flatten(), b.flatten())[0,1],
                np.corrcoef(g.flatten(), b.flatten())[0,1]
            ])
            
            # Color moments
            for channel in [r, g, b]:
                # Central moments
                mean_val = np.mean(channel)
                features.extend([
                    mean_val,  # 1st moment
                    np.mean((channel - mean_val)**2),  # 2nd moment (variance)
                    np.mean((channel - mean_val)**3),  # 3rd moment (skewness)
                    np.mean((channel - mean_val)**4)   # 4th moment (kurtosis)
                ])
            
            # Color coherence and consistency
            features.extend(self._analyze_color_coherence(image))
            
            # Dominant colors
            features.extend(self._extract_dominant_colors(image))
            
        else:
            # Grayscale - add zeros for color features
            features.extend([0] * 50)
        
        return features
    
    def _analyze_color_coherence(self, image):
        """Analyze color coherence and consistency"""
        features = []
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # Hue consistency
        hue_diff = np.abs(np.gradient(h.astype(np.float32)))
        features.extend([
            np.mean(hue_diff),
            np.std(hue_diff),
            np.max(hue_diff)
        ])
        
        # Saturation consistency
        sat_diff = np.abs(np.gradient(s.astype(np.float32)))
        features.extend([
            np.mean(sat_diff),
            np.std(sat_diff),
            np.max(sat_diff)
        ])
        
        # Color temperature estimation
        b, g, r = image[:,:,2], image[:,:,1], image[:,:,0]
        color_temp = np.mean(b) / (np.mean(r) + 1e-8)
        features.append(color_temp)
        
        return features
    
    def _extract_dominant_colors(self, image):
        """Extract dominant color features using k-means"""
        features = []
        
        # Reshape image for k-means
        pixels = image.reshape(-1, 3)
        
        try:
            from sklearn.cluster import KMeans
            
            # Find 5 dominant colors
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get color centers and their distribution
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # Calculate color distribution
            unique, counts = np.unique(labels, return_counts=True)
            percentages = counts / len(labels)
            
            # Add dominant color features
            for i in range(5):
                if i < len(centers):
                    features.extend(centers[i])  # RGB values
                    features.append(percentages[i] if i < len(percentages) else 0)
                else:
                    features.extend([0, 0, 0, 0])  # RGB + percentage
        except:
            # Fallback if sklearn not available
            features.extend([0] * 20)
        
        return features
    
    def extract_texture_patterns(self, image):
        """Extract comprehensive texture and pattern features"""
        features = []
        
        # Convert to grayscale for texture analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Local Binary Pattern (LBP) - multiple scales
        for radius in [1, 2, 3]:
            for n_points in [8, 16, 24]:
                try:
                    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
                    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2)
                    lbp_hist = lbp_hist.astype(float)
                    lbp_hist /= (lbp_hist.sum() + 1e-8)
                    features.extend([
                        np.mean(lbp_hist),
                        np.std(lbp_hist),
                        np.max(lbp_hist),
                        len(lbp_hist[lbp_hist > 0.01])
                    ])
                except:
                    features.extend([0, 0, 0, 0])
        
        # Gray Level Co-occurrence Matrix (GLCM) features
        features.extend(self._extract_glcm_features(gray))
        
        # Haralick texture features
        features.extend(self._extract_haralick_features(gray))
        
        # Laws texture features
        features.extend(self._extract_laws_features(gray))
        
        return features
    
    def _extract_glcm_features(self, gray):
        """Extract GLCM texture features"""
        features = []
        
        try:
            from skimage.feature import greycomatrix, greycoprops
            
            # Compute GLCM for different distances and angles
            distances = [1, 2, 3]
            angles = [0, 45, 90, 135]
            
            glcm = greycomatrix(gray, distances=distances, angles=angles, 
                              levels=64, symmetric=True, normed=True)
            
            # Extract properties
            properties = ['contrast', 'dissimilarity', 'homogeneity', 
                         'energy', 'correlation', 'ASM']
            
            for prop in properties:
                try:
                    values = greycoprops(glcm, prop).flatten()
                    features.extend([
                        np.mean(values),
                        np.std(values),
                        np.min(values),
                        np.max(values)
                    ])
                except:
                    features.extend([0, 0, 0, 0])
        except:
            # Fallback if skimage not available
            features.extend([0] * 24)
        
        return features
    
    def _extract_haralick_features(self, gray):
        """Extract Haralick texture features"""
        features = []
        
        try:
            # Compute co-occurrence matrix manually
            gray_norm = (gray / np.max(gray) * 63).astype(np.uint8)
            
            # Simple co-occurrence computation
            for d in [1, 2]:
                # Horizontal co-occurrence
                cooc = np.zeros((64, 64))
                for i in range(gray_norm.shape[0]):
                    for j in range(gray_norm.shape[1] - d):
                        cooc[gray_norm[i, j], gray_norm[i, j + d]] += 1
                
                # Normalize
                cooc = cooc / (np.sum(cooc) + 1e-8)
                
                # Compute Haralick features
                features.extend([
                    # Angular Second Moment
                    np.sum(cooc**2),
                    # Contrast
                    np.sum([(i-j)**2 * cooc[i,j] for i in range(64) for j in range(64)]),
                    # Correlation
                    np.sum([i*j*cooc[i,j] for i in range(64) for j in range(64)]),
                    # Variance
                    np.sum([(i-np.mean(gray_norm))**2 * cooc[i,j] for i in range(64) for j in range(64)]),
                    # Entropy
                    -np.sum([cooc[i,j] * np.log(cooc[i,j] + 1e-8) for i in range(64) for j in range(64)])
                ])
        except:
            features.extend([0] * 10)
        
        return features
    
    def _extract_laws_features(self, gray):
        """Extract Laws texture energy features"""
        features = []
        
        # Laws masks (5x5)
        l5 = np.array([1, 4, 6, 4, 1])  # Level
        e5 = np.array([-1, -2, 0, 2, 1])  # Edge
        s5 = np.array([-1, 0, 2, 0, -1])  # Spot
        w5 = np.array([-1, 2, 0, -2, 1])  # Wave
        r5 = np.array([1, -4, 6, -4, 1])  # Ripple
        
        vectors = [l5, e5, s5, w5, r5]
        
        try:
            for i, v1 in enumerate(vectors):
                for j, v2 in enumerate(vectors):
                    if i <= j:  # Avoid duplicate combinations
                        # Create 2D filter
                        filter_2d = np.outer(v1, v2)
                        
                        # Apply filter
                        filtered = cv2.filter2D(gray.astype(np.float32), -1, filter_2d)
                        
                        # Compute texture energy
                        features.extend([
                            np.mean(np.abs(filtered)),
                            np.std(filtered),
                            np.mean(filtered**2)
                        ])
        except:
            features.extend([0] * 45)
        
        return features
    
    def extract_frequency_domain_features(self, image):
        """Extract comprehensive frequency domain features"""
        features = []
        
        # Convert to grayscale for frequency analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # FFT analysis
        features.extend(self._extract_fft_features(gray))
        
        # DCT analysis
        features.extend(self._extract_dct_features(gray))
        
        # Power spectral density
        features.extend(self._extract_psd_features(gray))
        
        return features
    
    def _extract_fft_features(self, gray):
        """Extract FFT-based features"""
        features = []
        
        try:
            # Compute FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            phase_spectrum = np.angle(f_shift)
            
            # Log magnitude for better analysis
            log_mag = np.log(magnitude_spectrum + 1)
            
            # Extract statistics from magnitude spectrum
            features.extend([
                np.mean(log_mag),
                np.std(log_mag),
                np.median(log_mag),
                np.percentile(log_mag, 25),
                np.percentile(log_mag, 75),
                np.max(log_mag),
                np.min(log_mag)
            ])
            
            # Extract statistics from phase spectrum
            features.extend([
                np.mean(phase_spectrum),
                np.std(phase_spectrum),
                np.median(phase_spectrum)
            ])
            
            # Frequency band analysis
            h, w = magnitude_spectrum.shape
            center_h, center_w = h//2, w//2
            
            # Low frequency energy (center 25%)
            low_freq = magnitude_spectrum[center_h-h//8:center_h+h//8, 
                                        center_w-w//8:center_w+w//8]
            features.append(np.sum(low_freq**2))
            
            # High frequency energy (outer regions)
            high_freq_mask = np.ones_like(magnitude_spectrum)
            high_freq_mask[center_h-h//4:center_h+h//4, 
                          center_w-w//4:center_w+w//4] = 0
            high_freq_energy = np.sum((magnitude_spectrum * high_freq_mask)**2)
            features.append(high_freq_energy)
            
        except:
            features.extend([0] * 12)
        
        return features
    
    def _extract_dct_features(self, gray):
        """Extract DCT-based features"""
        features = []
        
        try:
            # Apply DCT
            dct_coeffs = cv2.dct(gray.astype(np.float32))
            
            # Extract statistics
            features.extend([
                np.mean(dct_coeffs),
                np.std(dct_coeffs),
                np.median(dct_coeffs),
                np.percentile(dct_coeffs, 25),
                np.percentile(dct_coeffs, 75),
                np.max(dct_coeffs),
                np.min(dct_coeffs)
            ])
            
            # AC/DC ratio
            dc_coeff = dct_coeffs[0, 0]
            ac_coeffs = dct_coeffs.copy()
            ac_coeffs[0, 0] = 0
            ac_energy = np.sum(ac_coeffs**2)
            features.append(ac_energy / (dc_coeff**2 + 1e-8))
            
            # High frequency DCT energy
            h, w = dct_coeffs.shape
            high_freq_dct = dct_coeffs[h//2:, w//2:]
            features.append(np.sum(high_freq_dct**2))
            
        except:
            features.extend([0] * 9)
        
        return features
    
    def _extract_psd_features(self, gray):
        """Extract Power Spectral Density features"""
        features = []
        
        try:
            # Compute PSD using FFT
            f_transform = np.fft.fft2(gray)
            psd = np.abs(f_transform)**2
            
            # Log PSD for better analysis
            log_psd = np.log(psd + 1)
            
            # Extract statistics
            features.extend([
                np.mean(log_psd),
                np.std(log_psd),
                np.median(log_psd),
                np.max(log_psd),
                np.min(log_psd)
            ])
            
            # Spectral centroid
            freq_x = np.fft.fftfreq(gray.shape[1])
            freq_y = np.fft.fftfreq(gray.shape[0])
            
            total_power = np.sum(psd)
            centroid_x = np.sum(np.sum(psd, axis=0) * freq_x) / (total_power + 1e-8)
            centroid_y = np.sum(np.sum(psd, axis=1) * freq_y) / (total_power + 1e-8)
            
            features.extend([centroid_x, centroid_y])
            
        except:
            features.extend([0] * 7)
        
        return features
    
    def extract_wavelet_features(self, image):
        """Extract wavelet-based features"""
        features = []
        
        if not WAVELET_AVAILABLE:
            return [0] * 30
        
        # Convert to grayscale for wavelet analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        try:
            # Multi-level wavelet decomposition
            wavelets = ['db4', 'haar', 'bior2.2']
            
            for wavelet in wavelets:
                try:
                    # 2-level decomposition
                    coeffs = pywt.wavedec2(gray, wavelet, level=2)
                    
                    # Extract features from each subband
                    for coeff in coeffs:
                        if isinstance(coeff, tuple):
                            for subband in coeff:
                                features.extend([
                                    np.mean(subband),
                                    np.std(subband),
                                    np.mean(np.abs(subband))
                                ])
                        else:
                            features.extend([
                                np.mean(coeff),
                                np.std(coeff),
                                np.mean(np.abs(coeff))
                            ])
                except:
                    features.extend([0] * 10)
        except:
            features.extend([0] * 30)
        
        return features
    
    def extract_edge_contour_features(self, image):
        """Extract edge and contour features"""
        features = []
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Multiple edge detectors
        # Canny edges
        edges_canny = cv2.Canny(gray, 50, 150)
        features.extend([
            np.mean(edges_canny),
            np.std(edges_canny),
            np.sum(edges_canny > 0) / edges_canny.size
        ])
        
        # Sobel edges
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        
        features.extend([
            np.mean(sobel_mag),
            np.std(sobel_mag),
            np.max(sobel_mag),
            np.percentile(sobel_mag, 95)
        ])
        
        # Laplacian edges
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.extend([
            np.mean(np.abs(laplacian)),
            np.std(laplacian),
            np.max(np.abs(laplacian))
        ])
        
        # Contour analysis
        contours, _ = cv2.findContours(edges_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Contour statistics
            areas = [cv2.contourArea(c) for c in contours]
            perimeters = [cv2.arcLength(c, True) for c in contours]
            
            features.extend([
                len(contours),
                np.mean(areas) if areas else 0,
                np.std(areas) if areas else 0,
                np.max(areas) if areas else 0,
                np.mean(perimeters) if perimeters else 0,
                np.std(perimeters) if perimeters else 0
            ])
        else:
            features.extend([0] * 6)
        
        return features
    
    def extract_geometric_features(self, image):
        """Extract geometric and morphological features"""
        features = []
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Binary image for morphological analysis
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = np.ones((5,5), np.uint8)
        
        # Opening and closing
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        features.extend([
            np.sum(opening > 0) / opening.size,
            np.sum(closing > 0) / closing.size,
            np.sum(binary > 0) / binary.size
        ])
        
        # Gradient and erosion/dilation
        gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
        erosion = cv2.erode(binary, kernel, iterations=1)
        dilation = cv2.dilate(binary, kernel, iterations=1)
        
        features.extend([
            np.sum(gradient > 0) / gradient.size,
            np.sum(erosion > 0) / erosion.size,
            np.sum(dilation > 0) / dilation.size
        ])
        
        # Shape analysis using moments
        moments = cv2.moments(binary)
        if moments['m00'] != 0:
            # Centroids
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            
            # Hu moments (invariant to translation, rotation, scale)
            hu_moments = cv2.HuMoments(moments)
            features.extend(hu_moments.flatten())
            
            # Additional shape features
            features.extend([cx, cy])
        else:
            features.extend([0] * 9)  # 7 Hu moments + 2 centroids
        
        return features
    
    def extract_noise_quality_features(self, image):
        """Extract noise and image quality features"""
        features = []
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            gray = image.astype(np.float32) / 255.0
        
        # Noise estimation
        try:
            from skimage.restoration import estimate_sigma
            try:
                # Try new API first
                noise_sigma = estimate_sigma(gray, channel_axis=None, average_sigmas=True)
            except TypeError:
                try:
                    # Fallback to older API
                    noise_sigma = estimate_sigma(gray, multichannel=False, average_sigmas=True)
                except:
                    # Simple noise estimation without Gaussian blur
                    try:
                        # Use median filter as a safer alternative
                        median_filtered = cv2.medianBlur(gray.astype(np.uint8), 5)
                        noise_sigma = np.std(gray.astype(float) - median_filtered.astype(float))
                    except:
                        # Even simpler fallback
                        noise_sigma = np.std(gray) * 0.1  # Rough estimate
            features.append(noise_sigma)
        except:
            features.append(0)
        
        # Image quality metrics
        # Laplacian variance (focus measure)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features.append(laplacian_var)
        
        # Gradient magnitude variance
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        features.append(np.var(grad_mag))
        
        # Brightness and contrast measures
        features.extend([
            np.mean(gray),  # Brightness
            np.std(gray),   # Contrast
            np.max(gray) - np.min(gray)  # Dynamic range
        ])
        
        # Local contrast variations
        # Divide image into blocks and compute contrast variation
        h, w = gray.shape
        block_size = 32
        contrasts = []
        
        for i in range(0, h-block_size, block_size):
            for j in range(0, w-block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                contrasts.append(np.std(block))
        
        if contrasts:
            features.extend([
                np.mean(contrasts),
                np.std(contrasts),
                np.max(contrasts) - np.min(contrasts)
            ])
        else:
            features.extend([0, 0, 0])
        
        return features
    
    def extract_gradient_features(self, image):
        """Extract gradient and orientation features"""
        features = []
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude and orientation
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        orientation = np.arctan2(grad_y, grad_x)
        
        # Gradient statistics
        features.extend([
            np.mean(magnitude),
            np.std(magnitude),
            np.median(magnitude),
            np.percentile(magnitude, 95),
            np.max(magnitude)
        ])
        
        # Orientation statistics
        features.extend([
            np.mean(orientation),
            np.std(orientation),
            np.median(orientation)
        ])
        
        # Histogram of oriented gradients (HOG)
        try:
            hog_features = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                              cells_per_block=(2, 2), feature_vector=True)
            
            # Summarize HOG features
            features.extend([
                np.mean(hog_features),
                np.std(hog_features),
                np.max(hog_features),
                np.min(hog_features),
                len(hog_features)
            ])
        except:
            features.extend([0] * 5)
        
        return features
    
    def extract_forgery_specific_features(self, image):
        """Extract features specifically designed for forgery detection"""
        features = []
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            rgb = image
        else:
            gray = image
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # 1. Double JPEG compression artifacts
        features.extend(self._detect_double_jpeg_artifacts(gray))
        
        # 2. Inconsistent lighting analysis
        features.extend(self._analyze_lighting_consistency(rgb))
        
        # 3. Copy-move detection features
        features.extend(self._extract_copy_move_features(gray))
        
        # 4. Splicing boundary detection
        features.extend(self._detect_splicing_boundaries(gray))
        
        # 5. Resampling artifacts
        features.extend(self._detect_resampling_artifacts(gray))
        
        return features
    
    def _detect_double_jpeg_artifacts(self, gray):
        """Detect double JPEG compression artifacts"""
        features = []
        
        try:
            # DCT analysis for blocking artifacts
            h, w = gray.shape
            block_features = []
            
            for i in range(0, h-8, 8):
                for j in range(0, w-8, 8):
                    block = gray[i:i+8, j:j+8].astype(np.float32)
                    dct_block = cv2.dct(block)
                    
                    # Check for quantization effects
                    # First few AC coefficients
                    ac_coeffs = dct_block[0, 1:4]  # Horizontal AC
                    ac_coeffs = np.append(ac_coeffs, dct_block[1:4, 0])  # Vertical AC
                    
                    block_features.append(np.std(ac_coeffs))
            
            if block_features:
                features.extend([
                    np.mean(block_features),
                    np.std(block_features),
                    np.percentile(block_features, 95)
                ])
            else:
                features.extend([0, 0, 0])
        except:
            features.extend([0, 0, 0])
        
        return features
    
    def _analyze_lighting_consistency(self, rgb):
        """Analyze lighting consistency across the image"""
        features = []
        
        try:
            # Convert to LAB for better lighting analysis
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0].astype(np.float32)
            
            # Divide image into regions and analyze lighting
            h, w = l_channel.shape
            region_size = min(h, w) // 4
            
            if region_size > 0:
                region_means = []
                for i in range(0, h-region_size, region_size):
                    for j in range(0, w-region_size, region_size):
                        region = l_channel[i:i+region_size, j:j+region_size]
                        region_means.append(np.mean(region))
                
                if region_means:
                    # Lighting consistency measures
                    features.extend([
                        np.std(region_means),  # Lighting variation
                        np.max(region_means) - np.min(region_means),  # Lighting range
                        np.mean(region_means)  # Overall brightness
                    ])
                else:
                    features.extend([0, 0, 0])
            else:
                features.extend([0, 0, 0])
        except:
            features.extend([0, 0, 0])
        
        return features
    
    def _extract_copy_move_features(self, gray):
        """Extract features for copy-move forgery detection"""
        features = []
        
        try:
            # Block-based similarity analysis
            block_size = 16
            h, w = gray.shape
            similarities = []
            
            if h > block_size * 2 and w > block_size * 2:
                for i in range(0, h-block_size, block_size):
                    for j in range(0, w-block_size, block_size):
                        block1 = gray[i:i+block_size, j:j+block_size]
                        
                        # Compare with other blocks
                        for ii in range(i+block_size, h-block_size, block_size):
                            for jj in range(0, w-block_size, block_size):
                                block2 = gray[ii:ii+block_size, jj:jj+block_size]
                                
                                # Normalized cross-correlation
                                correlation = cv2.matchTemplate(block1.astype(np.float32),
                                                               block2.astype(np.float32),
                                                               cv2.TM_CCOEFF_NORMED)
                                similarities.append(correlation[0, 0])
                
                if similarities:
                    features.extend([
                        np.mean(similarities),
                        np.std(similarities),
                        np.max(similarities),
                        len([s for s in similarities if s > 0.8])  # High similarity count
                    ])
                else:
                    features.extend([0, 0, 0, 0])
            else:
                features.extend([0, 0, 0, 0])
        except:
            features.extend([0, 0, 0, 0])
        
        return features
    
    def _detect_splicing_boundaries(self, gray):
        """Detect potential splicing boundaries"""
        features = []
        
        try:
            # Edge detection with multiple scales
            edges1 = cv2.Canny(gray, 50, 150)
            edges2 = cv2.Canny(gray, 100, 200)
            
            # Morphological operations to find potential boundaries
            kernel = np.ones((3,3), np.uint8)
            dilated_edges = cv2.dilate(edges1, kernel, iterations=2)
            
            # Analyze edge patterns
            contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Analyze contour shapes for unnatural boundaries
                straightness_scores = []
                for contour in contours:
                    if len(contour) > 10:
                        # Fit line to contour
                        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                        
                        # Calculate straightness (how well points fit the line)
                        distances = []
                        for point in contour:
                            px, py = point[0]
                            # Distance from point to line
                            dist = abs((py - y) * vx - (px - x) * vy) / np.sqrt(vx**2 + vy**2)
                            distances.append(dist)
                        
                        straightness_scores.append(np.mean(distances))
                
                if straightness_scores:
                    features.extend([
                        np.mean(straightness_scores),
                        np.std(straightness_scores),
                        np.min(straightness_scores)
                    ])
                else:
                    features.extend([0, 0, 0])
            else:
                features.extend([0, 0, 0])
        except:
            features.extend([0, 0, 0])
        
        return features
    
    def _detect_resampling_artifacts(self, gray):
        """Detect resampling/interpolation artifacts"""
        features = []
        
        try:
            # Compute second derivative to detect interpolation
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Analyze patterns in second derivative
            features.extend([
                np.mean(np.abs(laplacian)),
                np.std(laplacian),
                np.percentile(np.abs(laplacian), 95)
            ])
            
            # Periodic patterns analysis (interpolation often creates periodic patterns)
            # FFT analysis of the Laplacian
            fft_laplacian = np.fft.fft2(laplacian)
            magnitude = np.abs(fft_laplacian)
            
            # Look for peaks in frequency domain (indicating periodic patterns)
            # Get the magnitude spectrum in log scale
            log_magnitude = np.log(magnitude + 1)
            
            features.extend([
                np.mean(log_magnitude),
                np.std(log_magnitude),
                np.max(log_magnitude) - np.median(log_magnitude)
            ])
            
        except:
            features.extend([0, 0, 0, 0, 0, 0])
        
        return features
