"""
Knowledge Distillation Loss Functions for Semantic Segmentation.
Includes various distillation strategies for teacher-student learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    """
    Standard Knowledge Distillation Loss using KL Divergence.
    
    Args:
        temperature: temperature for softening probability distributions
        alpha: weight for distillation loss vs task loss
    """
    
    def __init__(self, temperature=4.0, alpha=0.5):
        super(KDLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits):
        """
        Compute KL divergence between student and teacher predictions.
        
        Args:
            student_logits: [B, C, H, W] student model output
            teacher_logits: [B, C, H, W] teacher model output
            
        Returns:
            distillation loss value
        """
        # Resize if dimensions don't match
        if student_logits.shape != teacher_logits.shape:
            teacher_logits = F.interpolate(teacher_logits, size=student_logits.shape[-2:],
                                          mode='bilinear', align_corners=False)
        
        # Apply temperature scaling and softmax
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Compute KL divergence
        kd_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        return kd_loss


class FeatureDistillationLoss(nn.Module):
    """
    Feature-based distillation loss using L2 distance.
    Matches intermediate feature representations between teacher and student.
    
    Args:
        feature_weight: weight for feature distillation loss
    """
    
    def __init__(self, feature_weight=1.0):
        super(FeatureDistillationLoss, self).__init__()
        self.feature_weight = feature_weight
    
    def forward(self, student_features, teacher_features):
        """
        Compute L2 distance between student and teacher features.
        
        Args:
            student_features: list of student feature maps
            teacher_features: list of teacher feature maps
            
        Returns:
            feature distillation loss
        """
        loss = 0.0
        num_features = min(len(student_features), len(teacher_features))
        
        for i in range(num_features):
            s_feat = student_features[i]
            t_feat = teacher_features[i]
            
            # Resize if needed
            if s_feat.shape != t_feat.shape:
                t_feat = F.interpolate(t_feat, size=s_feat.shape[-2:],
                                      mode='bilinear', align_corners=False)
                
                # If channel dimensions don't match, use adaptive pooling or projection
                if s_feat.shape[1] != t_feat.shape[1]:
                    # Simple channel-wise mean pooling for dimension matching
                    if s_feat.shape[1] < t_feat.shape[1]:
                        ratio = t_feat.shape[1] // s_feat.shape[1]
                        t_feat = t_feat.reshape(t_feat.shape[0], s_feat.shape[1], ratio, 
                                               t_feat.shape[2], t_feat.shape[3])
                        t_feat = t_feat.mean(dim=2)
                    else:
                        # Expand student features if needed (shouldn't happen typically)
                        ratio = s_feat.shape[1] // t_feat.shape[1]
                        t_feat = t_feat.repeat(1, ratio, 1, 1)[:, :s_feat.shape[1]]
            
            # Normalize features
            s_feat_norm = F.normalize(s_feat, p=2, dim=1)
            t_feat_norm = F.normalize(t_feat, p=2, dim=1)
            
            # L2 loss
            loss += F.mse_loss(s_feat_norm, t_feat_norm)
        
        return loss * self.feature_weight / num_features


class StructuralDistillationLoss(nn.Module):
    """
    Structural Knowledge Distillation using pixel-wise relations.
    Preserves spatial structure information from teacher to student.
    
    Args:
        weight: weight for structural distillation loss
    """
    
    def __init__(self, weight=1.0):
        super(StructuralDistillationLoss, self).__init__()
        self.weight = weight
    
    def forward(self, student_logits, teacher_logits):
        """
        Compute structural similarity between student and teacher predictions.
        """
        if student_logits.shape != teacher_logits.shape:
            teacher_logits = F.interpolate(teacher_logits, size=student_logits.shape[-2:],
                                          mode='bilinear', align_corners=False)
        
        # Compute pairwise similarities
        student_prob = F.softmax(student_logits, dim=1)
        teacher_prob = F.softmax(teacher_logits, dim=1)
        
        # Flatten spatial dimensions
        B, C, H, W = student_prob.shape
        student_flat = student_prob.view(B, C, -1)  # [B, C, H*W]
        teacher_flat = teacher_prob.view(B, C, -1)  # [B, C, H*W]
        
        # Compute similarity matrices (spatial relationships)
        student_sim = torch.bmm(student_flat.transpose(1, 2), student_flat)  # [B, H*W, H*W]
        teacher_sim = torch.bmm(teacher_flat.transpose(1, 2), teacher_flat)  # [B, H*W, H*W]
        
        # L2 loss on similarity matrices
        loss = F.mse_loss(student_sim, teacher_sim)
        
        return loss * self.weight


class CombinedKDLoss(nn.Module):
    """
    Combined knowledge distillation loss that integrates multiple distillation strategies.
    
    Args:
        kd_weight: weight for logit-based KD loss
        feature_weight: weight for feature-based distillation
        structural_weight: weight for structural distillation
        temperature: temperature for KD loss
    """
    
    def __init__(self, kd_weight=1.0, feature_weight=0.5, structural_weight=0.3, 
                 temperature=4.0):
        super(CombinedKDLoss, self).__init__()
        
        self.kd_weight = kd_weight
        self.feature_weight = feature_weight
        self.structural_weight = structural_weight
        
        self.kd_loss = KDLoss(temperature=temperature)
        self.feature_loss = FeatureDistillationLoss(feature_weight=1.0)
        self.structural_loss = StructuralDistillationLoss(weight=1.0)
    
    def forward(self, student_outputs, teacher_outputs):
        """
        Compute combined distillation loss.
        
        Args:
            student_outputs: dict with 'logits' and optionally 'features'
            teacher_outputs: dict with 'logits' and optionally 'features'
            
        Returns:
            dict with total loss and individual loss components
        """
        losses = {}
        total_loss = 0.0
        
        # Logit-based KD loss
        if self.kd_weight > 0:
            kd_loss = self.kd_loss(student_outputs['logits'], teacher_outputs['logits'])
            losses['kd_loss'] = kd_loss
            total_loss += self.kd_weight * kd_loss
        
        # Feature-based distillation
        if self.feature_weight > 0 and 'features' in student_outputs and 'features' in teacher_outputs:
            feat_loss = self.feature_loss(student_outputs['features'], teacher_outputs['features'])
            losses['feature_loss'] = feat_loss
            total_loss += self.feature_weight * feat_loss
        
        # Structural distillation
        if self.structural_weight > 0:
            struct_loss = self.structural_loss(student_outputs['logits'], teacher_outputs['logits'])
            losses['structural_loss'] = struct_loss
            total_loss += self.structural_weight * struct_loss
        
        losses['total_kd_loss'] = total_loss
        return losses
