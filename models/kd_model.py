"""
Knowledge Distillation Model for Semantic Segmentation.
Combines DINOv2 teacher and DDRNet student for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .kd_losses import CombinedKDLoss


class KnowledgeDistillationModel(nn.Module):
    """
    Knowledge Distillation framework for semantic segmentation.
    Uses DINOv2 as teacher and DDRNet as student.
    
    Args:
        teacher: DINOv2 teacher model
        student: DDRNet student model
        num_classes: number of segmentation classes
        kd_weight: weight for knowledge distillation loss
        feature_weight: weight for feature distillation
        structural_weight: weight for structural distillation
        temperature: temperature for softmax in KD
        task_loss_weight: weight for task (segmentation) loss
    """
    
    def __init__(self, teacher, student, num_classes=19,
                 kd_weight=1.0, feature_weight=0.5, structural_weight=0.3,
                 temperature=4.0, task_loss_weight=1.0):
        super(KnowledgeDistillationModel, self).__init__()
        
        self.teacher = teacher
        self.student = student
        self.num_classes = num_classes
        self.task_loss_weight = task_loss_weight
        
        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Knowledge distillation loss
        self.kd_loss = CombinedKDLoss(
            kd_weight=kd_weight,
            feature_weight=feature_weight,
            structural_weight=structural_weight,
            temperature=temperature
        )
        
        # Task loss (Cross Entropy for segmentation)
        self.task_loss = nn.CrossEntropyLoss(ignore_index=255)
        
        # Feature adaptation layers to match student features to teacher dimensions
        self.feature_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        ])
    
    def forward(self, images, labels=None, return_loss=True):
        """
        Forward pass for training or inference.
        
        Args:
            images: input images [B, 3, H, W]
            labels: ground truth labels [B, H, W] (optional, for training)
            return_loss: whether to compute and return losses
            
        Returns:
            If return_loss=True and labels provided:
                dict with losses and outputs
            Otherwise:
                student predictions
        """
        # Student forward pass
        student_out = self.student(images)
        
        # Handle auxiliary output from student if present
        if isinstance(student_out, list):
            student_logits = student_out[0]
            student_aux = student_out[1] if len(student_out) > 1 else None
        else:
            student_logits = student_out
            student_aux = None
        
        # Prepare student outputs dict
        student_outputs = {'logits': student_logits}
        
        if return_loss and labels is not None:
            # Teacher forward pass (no gradient)
            with torch.no_grad():
                teacher_out = self.teacher(images)
            
            # Extract teacher outputs
            if isinstance(teacher_out, dict):
                teacher_logits = teacher_out['logits']
                teacher_features = teacher_out.get('features', None)
            else:
                teacher_logits = teacher_out
                teacher_features = None
            
            teacher_outputs = {'logits': teacher_logits}
            if teacher_features is not None:
                teacher_outputs['features'] = teacher_features
            
            # Add adapted student features if teacher has features
            if teacher_features is not None:
                # Extract student features (from intermediate layers)
                student_features = self._extract_student_features(images)
                student_outputs['features'] = student_features
            
            # Compute losses
            losses = {}
            
            # Task loss (segmentation)
            # Resize predictions to match label size
            H, W = labels.shape[-2:]
            student_pred = F.interpolate(student_logits, size=(H, W), 
                                        mode='bilinear', align_corners=False)
            task_loss = self.task_loss(student_pred, labels)
            losses['task_loss'] = task_loss
            
            # Auxiliary loss if present
            if student_aux is not None:
                student_aux_pred = F.interpolate(student_aux, size=(H, W),
                                                 mode='bilinear', align_corners=False)
                aux_loss = self.task_loss(student_aux_pred, labels)
                losses['aux_loss'] = aux_loss
                task_loss = task_loss + 0.4 * aux_loss  # Standard aux loss weight
            
            # Knowledge distillation losses
            kd_losses = self.kd_loss(student_outputs, teacher_outputs)
            losses.update(kd_losses)
            
            # Total loss
            total_loss = self.task_loss_weight * task_loss + kd_losses['total_kd_loss']
            losses['total_loss'] = total_loss
            
            return {
                'loss': total_loss,
                'losses': losses,
                'student_pred': student_logits,
                'teacher_pred': teacher_logits
            }
        else:
            # Inference mode
            return student_logits
    
    def _extract_student_features(self, images):
        """
        Extract intermediate features from student model for distillation.
        This is a simplified version - can be enhanced based on student architecture.
        """
        # For now, return empty list - can be enhanced to extract actual features
        # from student's intermediate layers
        features = []
        return features
    
    def get_student_model(self):
        """Return the student model for inference or further training."""
        return self.student
    
    def get_teacher_model(self):
        """Return the teacher model."""
        return self.teacher


def create_kd_model(student_model, teacher_model, num_classes=19, **kd_kwargs):
    """
    Factory function to create a KD model.
    
    Args:
        student_model: DDRNet student model
        teacher_model: DINOv2 teacher model
        num_classes: number of classes
        **kd_kwargs: additional arguments for KD configuration
        
    Returns:
        KnowledgeDistillationModel instance
    """
    return KnowledgeDistillationModel(
        teacher=teacher_model,
        student=student_model,
        num_classes=num_classes,
        **kd_kwargs
    )
