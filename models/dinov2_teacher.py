"""
DINOv2 Teacher model wrapper for knowledge distillation.
Uses pretrained DINOv2 as a feature extractor for semantic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2Teacher(nn.Module):
    """
    DINOv2 teacher model for semantic segmentation with knowledge distillation.
    
    Args:
        model_name: DINOv2 model variant ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_classes: number of segmentation classes
        freeze_backbone: whether to freeze the backbone weights
        pretrained: whether to use pretrained weights
        img_size: input image size (default: 518 for best patch alignment)
        output_features: whether to output intermediate features for KD
    """
    
    def __init__(self, model_name='dinov2_vitb14', num_classes=19, freeze_backbone=True,
                 pretrained=True, img_size=518, output_features=True):
        super(DINOv2Teacher, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.output_features = output_features
        self.img_size = img_size
        
        # Load pretrained DINOv2 backbone
        if pretrained:
            self.backbone = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        else:
            self.backbone = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=False)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get embedding dimension from the backbone
        self.embed_dim = self.backbone.embed_dim
        
        # Determine patch size (14 for all DINOv2 models)
        self.patch_size = 14
        
        # Linear projection for segmentation
        self.decode_head = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(self.embed_dim, num_classes, kernel_size=1)
        )
        
        # Feature projection layers for knowledge distillation
        if self.output_features:
            # Project DINOv2 features to student-compatible dimensions
            self.feature_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.embed_dim, 256, kernel_size=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(self.embed_dim, 512, kernel_size=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True)
                )
            ])
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: input image tensor [B, 3, H, W]
            
        Returns:
            If output_features=True: dict with 'logits' and 'features'
            Otherwise: segmentation logits [B, num_classes, H, W]
        """
        B, _, H, W = x.shape
        
        # Calculate feature map size
        feat_h = H // self.patch_size
        feat_w = W // self.patch_size
        
        # Extract features from DINOv2
        # DINOv2 outputs [B, num_patches + 1, embed_dim]
        features = self.backbone.forward_features(x)
        
        # Remove CLS token and reshape to spatial dimensions
        patch_features = features['x_norm_patchtokens']  # [B, num_patches, embed_dim]
        
        # Reshape to [B, embed_dim, H, W]
        patch_features = patch_features.reshape(B, feat_h, feat_w, self.embed_dim)
        patch_features = patch_features.permute(0, 3, 1, 2)  # [B, embed_dim, feat_h, feat_w]
        
        # Generate segmentation logits
        seg_features = F.interpolate(patch_features, size=(H // 8, W // 8), 
                                     mode='bilinear', align_corners=False)
        logits = self.decode_head(seg_features)
        
        if self.output_features:
            # Prepare features for knowledge distillation
            kd_features = []
            for proj in self.feature_proj:
                feat = proj(seg_features)
                kd_features.append(feat)
            
            return {
                'logits': logits,
                'features': kd_features,
                'raw_features': seg_features
            }
        else:
            return logits
    
    def get_features(self, x):
        """
        Extract only features without segmentation head.
        Useful for feature-based distillation.
        """
        B, _, H, W = x.shape
        feat_h = H // self.patch_size
        feat_w = W // self.patch_size
        
        with torch.no_grad():
            features = self.backbone.forward_features(x)
            patch_features = features['x_norm_patchtokens']
            patch_features = patch_features.reshape(B, feat_h, feat_w, self.embed_dim)
            patch_features = patch_features.permute(0, 3, 1, 2)
            
            seg_features = F.interpolate(patch_features, size=(H // 8, W // 8),
                                        mode='bilinear', align_corners=False)
        
        return seg_features


def dinov2_vits14_teacher(num_classes=19, **kwargs):
    """DINOv2-ViT-S/14 teacher model"""
    return DINOv2Teacher('dinov2_vits14', num_classes=num_classes, **kwargs)


def dinov2_vitb14_teacher(num_classes=19, **kwargs):
    """DINOv2-ViT-B/14 teacher model"""
    return DINOv2Teacher('dinov2_vitb14', num_classes=num_classes, **kwargs)


def dinov2_vitl14_teacher(num_classes=19, **kwargs):
    """DINOv2-ViT-L/14 teacher model"""
    return DINOv2Teacher('dinov2_vitl14', num_classes=num_classes, **kwargs)


def dinov2_vitg14_teacher(num_classes=19, **kwargs):
    """DINOv2-ViT-G/14 teacher model (largest)"""
    return DINOv2Teacher('dinov2_vitg14', num_classes=num_classes, **kwargs)
