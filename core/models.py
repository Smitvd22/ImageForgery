#!/usr/bin/env python3
"""
Ultra-Enhanced Models with State-of-the-Art ImageNet Architectures
Implementing multiple SOTA models for maximum accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import TIMM for state-of-the-art models
try:
    import timm
    TIMM_AVAILABLE = True
    print("TIMM available for state-of-the-art models")
except ImportError:
    TIMM_AVAILABLE = False
    print("TIMM not available. Install with: pip install timm")

# Optional HuggingFace import with error handling
try:
    from transformers import AutoModel, AutoImageProcessor, AutoConfig
    HF_AVAILABLE = True
    print("HuggingFace transformers available")
except ImportError:
    HF_AVAILABLE = False
    print("HuggingFace transformers not available")

class UltraEnhancedImageNetBackbone(nn.Module):
    """
    Ultra-enhanced backbone using multiple state-of-the-art ImageNet models
    """
    def __init__(self, model_configs):
        super().__init__()
        self.models = nn.ModuleDict()
        self.feature_dims = {}
        self.total_feature_dim = 0
        
        for model_name, config in model_configs.items():
            if not config['enabled']:
                continue
                
            try:
                if TIMM_AVAILABLE:
                    model = timm.create_model(
                        model_name, 
                        pretrained=config['pretrained'],
                        num_classes=0,  # Remove classification head
                        global_pool='avg'
                    )
                    self.models[model_name] = model
                    self.feature_dims[model_name] = config['feature_dim']
                    self.total_feature_dim += config['feature_dim']
                    print(f"✅ Loaded TIMM model: {model_name} (features: {config['feature_dim']})")
                else:
                    # Fallback to torchvision models
                    if 'efficientnet' in model_name:
                        model = models.efficientnet_v2_l(weights='IMAGENET1K_V1' if config['pretrained'] else None)
                        model.classifier = nn.Identity()
                        self.models[model_name] = model
                        self.feature_dims[model_name] = 1280
                        self.total_feature_dim += 1280
                    elif 'resnet' in model_name:
                        model = models.resnet152(weights='IMAGENET1K_V2' if config['pretrained'] else None)
                        model.fc = nn.Identity()
                        self.models[model_name] = model
                        self.feature_dims[model_name] = 2048
                        self.total_feature_dim += 2048
                        
            except Exception as e:
                print(f"⚠️ Failed to load {model_name}: {e}")
                continue
        
        # Multi-scale attention for each model
        self.attention_modules = nn.ModuleDict()
        for model_name in self.models.keys():
            self.attention_modules[model_name] = MultiScaleAttention(self.feature_dims[model_name])
        
        # Global feature fusion
        if self.total_feature_dim > 0:
            self.global_fusion = nn.Sequential(
                nn.Linear(self.total_feature_dim, self.total_feature_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.total_feature_dim // 2, self.total_feature_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.total_feature_dim // 4, 2048)
            )
        else:
            self.global_fusion = nn.Identity()
            self.total_feature_dim = 2048
    
    def forward(self, x):
        features = []
        
        for model_name, model in self.models.items():
            try:
                # Extract features from each model
                feat = model(x)
                
                # Apply attention if feature dimension matches
                if feat.dim() > 2:
                    feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
                
                # Apply attention module
                if model_name in self.attention_modules:
                    feat = self.attention_modules[model_name](feat)
                
                features.append(feat)
            except Exception as e:
                print(f"⚠️ Error processing {model_name}: {e}")
                continue
        
        if features:
            # Concatenate all features
            combined_features = torch.cat(features, dim=1)
            
            # Apply global fusion
            fused_features = self.global_fusion(combined_features)
            return fused_features
        else:
            # Fallback if no models loaded
            return torch.zeros((x.size(0), 2048), device=x.device)

class MultiScaleAttention(nn.Module):
    """Enhanced attention mechanism for better feature selection"""
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 8),
            nn.ReLU(),
            nn.Linear(feature_dim // 8, feature_dim),
            nn.Sigmoid()
        )
        
        # Feature enhancement
        self.feature_enhancement = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        # Apply channel attention
        attention_weights = self.channel_attention(x)
        attended_features = x * attention_weights
        
        # Apply feature enhancement
        enhanced_features = self.feature_enhancement(attended_features)
        
        # Residual connection
        return enhanced_features + x

class AdvancedHuggingFaceEnsemble(nn.Module):
    """Enhanced HuggingFace ensemble with multiple transformer models"""
    def __init__(self, model_names):
        super().__init__()
        self.models = nn.ModuleDict()
        self.processors = {}
        self.feature_dims = {}
        self.total_feature_dim = 0
        
        if not HF_AVAILABLE:
            print("⚠️ HuggingFace not available, skipping transformer models")
            self.total_feature_dim = 1024  # Fallback dimension
            return
        
        for model_name in model_names:
            try:
                # Load model and processor
                model = AutoModel.from_pretrained(model_name)
                processor = AutoImageProcessor.from_pretrained(model_name)
                config = AutoConfig.from_pretrained(model_name)
                
                # Remove classification head if present
                if hasattr(model, 'classifier'):
                    model.classifier = nn.Identity()
                if hasattr(model, 'head'):
                    model.head = nn.Identity()
                
                self.models[model_name.replace('/', '_')] = model
                self.processors[model_name] = processor
                
                # Get feature dimension
                if hasattr(config, 'hidden_size'):
                    feature_dim = config.hidden_size
                elif hasattr(config, 'embed_dim'):
                    feature_dim = config.embed_dim
                else:
                    feature_dim = 768  # Default
                
                self.feature_dims[model_name] = feature_dim
                self.total_feature_dim += feature_dim
                print(f"✅ Loaded HuggingFace model: {model_name} (features: {feature_dim})")
                
            except Exception as e:
                print(f"⚠️ Failed to load HuggingFace model {model_name}: {e}")
                continue
        
        # Feature fusion for HuggingFace models
        if self.total_feature_dim > 0:
            self.hf_fusion = nn.Sequential(
                nn.Linear(self.total_feature_dim, self.total_feature_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.total_feature_dim // 2, 1024)
            )
        else:
            self.hf_fusion = nn.Linear(1, 1024)  # Dummy layer
            self.total_feature_dim = 1024
    
    def forward(self, x):
        if not HF_AVAILABLE or not self.models:
            return torch.zeros((x.size(0), 1024), device=x.device)
        
        features = []
        
        for model_name, model in self.models.items():
            try:
                # Process input for this specific model
                outputs = model(pixel_values=x)
                
                # Extract pooled features
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    feat = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    feat = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
                else:
                    feat = outputs[0].mean(dim=1)  # Fallback
                
                features.append(feat)
                
            except Exception as e:
                print(f"⚠️ Error processing HuggingFace model {model_name}: {e}")
                continue
        
        if features:
            combined_features = torch.cat(features, dim=1)
            return self.hf_fusion(combined_features)
        else:
            return torch.zeros((x.size(0), 1024), device=x.device)

class UltraAdvancedForgerySpecificCNN(nn.Module):
    """Ultra-advanced CNN specifically designed for forgery detection"""
    def __init__(self, input_channels=3):
        super().__init__()
        
        # Multi-scale feature extraction branches
        self.branch1 = self._build_branch(input_channels, [64, 128, 256], [3, 5, 7])
        self.branch2 = self._build_branch(input_channels, [64, 128, 256], [1, 3, 5])
        self.branch3 = self._build_branch(input_channels, [32, 64, 128], [7, 5, 3])
        
        # Frequency domain analysis branch
        self.freq_branch = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Attention mechanism for branch fusion
        self.branch_attention = nn.Sequential(
            nn.Linear(256 * 3 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Softmax(dim=1)
        )
        
        # Final feature processing
        self.final_processing = nn.Sequential(
            nn.Linear(256 * 3 + 128, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.feature_dim = 1024
    
    def _build_branch(self, in_channels, channel_list, kernel_sizes):
        layers = []
        current_channels = in_channels
        
        for out_channels, kernel_size in zip(channel_list, kernel_sizes):
            layers.extend([
                nn.Conv2d(current_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            current_channels = out_channels
        
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Process through each branch
        feat1 = self.branch1(x).view(x.size(0), -1)
        feat2 = self.branch2(x).view(x.size(0), -1)
        feat3 = self.branch3(x).view(x.size(0), -1)
        
        # Frequency domain analysis
        freq_feat = self.freq_branch(x).view(x.size(0), -1)
        
        # Combine all features
        combined_features = torch.cat([feat1, feat2, feat3, freq_feat], dim=1)
        
        # Apply attention-based fusion
        attention_weights = self.branch_attention(combined_features)
        
        # Weighted combination
        weighted_features = combined_features * attention_weights.repeat(1, combined_features.size(1) // 4)
        
        # Final processing
        final_features = self.final_processing(weighted_features)
        
        return final_features

class UltraComprehensiveMultiBackboneExtractor(nn.Module):
    """
    Ultra-comprehensive multi-backbone architecture with state-of-the-art models
    """
    def __init__(self, imagenet_models, huggingface_models=None, dropout_rate=0.2):
        super().__init__()
        
        print("🚀 Initializing Ultra-Comprehensive Multi-Backbone Architecture...")
        
        # State-of-the-art ImageNet backbone
        self.imagenet_backbone = UltraEnhancedImageNetBackbone(imagenet_models)
        
        # Enhanced ResNet++ (keeping original for comparison)
        self.resnet_plus_plus = ImprovedResNetBackbone(pretrained=True)
        
        # U-Net components
        self.unet = UNet(in_channels=3)
        self.unet_r = UNetR(in_channels=3)
        
        # Ultra-advanced forgery-specific CNN
        self.forgery_cnn = UltraAdvancedForgerySpecificCNN()
        
        # Advanced frequency domain analyzer
        self.frequency_analyzer = AdvancedFrequencyDomainAnalyzer()
        
        # HuggingFace ensemble
        if huggingface_models and HF_AVAILABLE:
            self.huggingface_ensemble = AdvancedHuggingFaceEnsemble(huggingface_models)
        else:
            self.huggingface_ensemble = None
        
        # Calculate total feature dimensions
        self.feature_dims = {
            'imagenet_backbone': self.imagenet_backbone.total_feature_dim,
            'resnet_plus_plus': 6144,
            'unet': 64,
            'unet_r': 64,
            'forgery_cnn': 1024,
            'frequency_analyzer': 512,
        }
        
        if self.huggingface_ensemble:
            self.feature_dims['huggingface_ensemble'] = 1024
        
        self.total_input_dim = sum(self.feature_dims.values())
        
        # Ultra-advanced feature fusion network
        self.advanced_fusion = UltraAdvancedFeatureFusion(
            input_dim=self.total_input_dim,
            output_dim=256,
            dropout_rate=dropout_rate
        )
        
        print(f"✅ Ultra-Comprehensive Architecture initialized:")
        for name, dim in self.feature_dims.items():
            print(f"   - {name}: {dim} dimensions")
        print(f"   - Total input dimensions: {self.total_input_dim}")
        print(f"   - Final fused dimensions: 256")
    
    def forward(self, x):
        features = []
        
        # Extract features from all backbones
        try:
            imagenet_feat = self.imagenet_backbone(x)
            features.append(imagenet_feat)
        except Exception as e:
            print(f"⚠️ ImageNet backbone error: {e}")
            features.append(torch.zeros((x.size(0), self.feature_dims['imagenet_backbone']), device=x.device))
        
        try:
            resnet_feat = self.resnet_plus_plus(x)
            features.append(resnet_feat)
        except Exception as e:
            print(f"⚠️ ResNet++ error: {e}")
            features.append(torch.zeros((x.size(0), 6144), device=x.device))
        
        try:
            unet_feat = self.unet(x)
            features.append(unet_feat)
        except Exception as e:
            print(f"⚠️ U-Net error: {e}")
            features.append(torch.zeros((x.size(0), 64), device=x.device))
        
        try:
            unet_r_feat = self.unet_r(x)
            features.append(unet_r_feat)
        except Exception as e:
            print(f"⚠️ U-Net R error: {e}")
            features.append(torch.zeros((x.size(0), 64), device=x.device))
        
        try:
            forgery_feat = self.forgery_cnn(x)
            features.append(forgery_feat)
        except Exception as e:
            print(f"⚠️ Forgery CNN error: {e}")
            features.append(torch.zeros((x.size(0), 1024), device=x.device))
        
        try:
            freq_feat = self.frequency_analyzer(x)
            features.append(freq_feat)
        except Exception as e:
            print(f"⚠️ Frequency analyzer error: {e}")
            features.append(torch.zeros((x.size(0), 512), device=x.device))
        
        if self.huggingface_ensemble:
            try:
                hf_feat = self.huggingface_ensemble(x)
                features.append(hf_feat)
            except Exception as e:
                print(f"⚠️ HuggingFace ensemble error: {e}")
                features.append(torch.zeros((x.size(0), 1024), device=x.device))
        
        # Concatenate all features
        combined_features = torch.cat(features, dim=1)
        
        # Apply advanced fusion
        fused_features = self.advanced_fusion(combined_features)
        
        return fused_features
    
    def get_feature_dims(self):
        return self.feature_dims

class UltraAdvancedFeatureFusion(nn.Module):
    """Ultra-advanced feature fusion with attention and residual connections"""
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super().__init__()
        
        # Multi-head attention for feature importance
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feature transformation layers
        self.transform_layers = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            
            nn.Linear(input_dim // 4, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        # Residual connection
        self.residual_proj = nn.Linear(input_dim, output_dim)
        
        # Final activation
        self.final_activation = nn.GELU()
    
    def forward(self, x):
        # Add sequence dimension for attention
        x_seq = x.unsqueeze(1)
        
        # Apply self-attention
        attended_features, _ = self.attention(x_seq, x_seq, x_seq)
        attended_features = attended_features.squeeze(1)
        
        # Transform features
        transformed = self.transform_layers(attended_features)
        
        # Residual connection
        residual = self.residual_proj(x)
        
        # Combine and activate
        output = self.final_activation(transformed + residual)
        
        return output

class AdvancedFrequencyDomainAnalyzer(nn.Module):
    """Advanced frequency domain analysis for detecting forgery artifacts"""
    def __init__(self):
        super().__init__()
        
        # DCT analysis layers
        self.dct_conv = nn.Sequential(
            nn.Conv2d(3, 64, 8, stride=8),  # 8x8 DCT blocks
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # FFT analysis layers
        self.fft_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Frequency fusion
        self.freq_fusion = nn.Sequential(
            nn.Linear(128 * 8 * 8 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        
        self.feature_dim = 512
    
    def forward(self, x):
        # DCT analysis
        dct_features = self.dct_conv(x).view(x.size(0), -1)
        
        # FFT analysis
        fft_input = torch.fft.fft2(x, dim=(-2, -1))
        fft_magnitude = torch.abs(fft_input)
        fft_features = self.fft_conv(fft_magnitude).view(x.size(0), -1)
        
        # Combine frequency features
        combined_freq = torch.cat([dct_features, fft_features], dim=1)
        final_features = self.freq_fusion(combined_freq)
        
        return final_features

# Include previous components for compatibility
class ImprovedResNetBackbone(nn.Module):
    """Enhanced ResNet with improved attention mechanisms"""
    def __init__(self, pretrained=True):
        super().__init__()
        base = models.resnet152(weights='IMAGENET1K_V2' if pretrained else None)
        self.features = nn.Sequential(*list(base.children())[:-2])
        
        self.spatial_attention = MultiScaleSpatialAttention()
        self.channel_attention = ImprovedChannelAttention(2048)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.gem_pool = GeM(p=3.0)
        
        self.feature_dim = 2048 * 3
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.features(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        
        avg_features = self.avg_pool(x).view(x.size(0), -1)
        max_features = self.max_pool(x).view(x.size(0), -1)
        gem_features = self.gem_pool(x).view(x.size(0), -1)
        
        features = torch.cat([avg_features, max_features, gem_features], dim=1)
        features = self.dropout(features)
        
        return features

class MultiScaleSpatialAttention(nn.Module):
    """Multi-scale spatial attention"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.final_conv = nn.Conv2d(3, 1, kernel_size=1)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_input = torch.cat([avg_out, max_out], dim=1)
        
        att1 = self.conv1(attention_input)
        att2 = self.conv2(attention_input)
        att3 = self.conv3(attention_input)
        
        combined_att = torch.cat([att1, att2, att3], dim=1)
        attention = torch.sigmoid(self.final_conv(combined_att))
        
        return attention

class ImprovedChannelAttention(nn.Module):
    """Improved channel attention"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_channels // reduction, in_channels // (reduction // 2)),
            nn.ReLU(),
            nn.Linear(in_channels // (reduction // 2), in_channels)
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return attention

class GeM(nn.Module):
    """Generalized Mean Pooling"""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

class UNet(nn.Module):
    """U-Net for semantic features"""
    def __init__(self, in_channels=3):
        super().__init__()
        self.encoder = UNetEncoder(in_channels)
        self.decoder = UNetDecoder()
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_projection = nn.Linear(64, 64)
        
    def forward(self, x):
        encoded_features, skip_connections = self.encoder(x)
        decoded_features = self.decoder(encoded_features, skip_connections)
        pooled_features = self.final_pool(decoded_features).view(x.size(0), -1)
        final_features = self.feature_projection(pooled_features)
        return final_features

class UNetR(nn.Module):
    """U-Net R: Residual-enhanced U-Net"""
    def __init__(self, in_channels=3):
        super().__init__()
        self.unet = UNet(in_channels)
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fusion = nn.Linear(128, 64)
        
    def forward(self, x):
        unet_features = self.unet(x)
        residual_features = self.residual_conv(x).view(x.size(0), -1)
        combined_features = torch.cat([unet_features, residual_features], dim=1)
        final_features = self.fusion(combined_features)
        return final_features

class UNetEncoder(nn.Module):
    """U-Net Encoder"""
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = self._double_conv(in_channels, 64)
        self.conv2 = self._double_conv(64, 128)
        self.conv3 = self._double_conv(128, 256)
        self.conv4 = self._double_conv(256, 512)
        self.pool = nn.MaxPool2d(2)
        
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        skip_connections = []
        
        x1 = self.conv1(x)
        skip_connections.append(x1)
        x = self.pool(x1)
        
        x2 = self.conv2(x)
        skip_connections.append(x2)
        x = self.pool(x2)
        
        x3 = self.conv3(x)
        skip_connections.append(x3)
        x = self.pool(x3)
        
        x4 = self.conv4(x)
        
        return x4, skip_connections[::-1]

class UNetDecoder(nn.Module):
    """U-Net Decoder"""
    def __init__(self):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = self._double_conv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = self._double_conv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = self._double_conv(128, 64)
        
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x, skip_connections):
        x = self.up1(x)
        x = torch.cat([x, skip_connections[0]], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, skip_connections[1]], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, skip_connections[2]], dim=1)
        x = self.conv3(x)
        
        return x

# Alias for backward compatibility
UltraImprovedMultiModelExtractor = UltraComprehensiveMultiBackboneExtractor
ImprovedMultiModelExtractor = UltraComprehensiveMultiBackboneExtractor

print("Ultra-Enhanced Models loaded successfully!")
