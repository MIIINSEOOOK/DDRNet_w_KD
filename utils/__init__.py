"""
Utility functions for training and evaluation.
"""

import torch
import numpy as np
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import PolynomialLR, CosineAnnealingLR, MultiStepLR


def get_optimizer(model, optimizer_type='sgd', lr=0.01, momentum=0.9, weight_decay=5e-4):
    """
    Create optimizer for model training.
    
    Args:
        model: PyTorch model
        optimizer_type: 'sgd', 'adam', or 'adamw'
        lr: learning rate
        momentum: momentum for SGD
        weight_decay: weight decay
        
    Returns:
        optimizer instance
    """
    if optimizer_type.lower() == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def get_scheduler(optimizer, scheduler_type='poly', max_iters=None, milestones=None, **kwargs):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: optimizer instance
        scheduler_type: 'poly', 'cosine', or 'multistep'
        max_iters: maximum iterations for poly/cosine scheduler
        milestones: milestone epochs for multistep scheduler
        
    Returns:
        scheduler instance
    """
    if scheduler_type.lower() == 'poly':
        assert max_iters is not None, "max_iters required for polynomial scheduler"
        power = kwargs.get('power', 0.9)
        scheduler = PolynomialLR(optimizer, total_iters=max_iters, power=power)
    elif scheduler_type.lower() == 'cosine':
        assert max_iters is not None, "max_iters required for cosine scheduler"
        scheduler = CosineAnnealingLR(optimizer, T_max=max_iters)
    elif scheduler_type.lower() == 'multistep':
        assert milestones is not None, "milestones required for multistep scheduler"
        gamma = kwargs.get('gamma', 0.1)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SegmentationMetrics:
    """
    Compute segmentation metrics including mIoU, pixel accuracy.
    """
    
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred, target):
        """
        Update confusion matrix.
        
        Args:
            pred: predicted labels [H, W] or [B, H, W]
            target: ground truth labels [H, W] or [B, H, W]
        """
        pred = pred.flatten()
        target = target.flatten()
        
        # Remove ignored pixels
        mask = (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]
        
        # Update confusion matrix
        for t, p in zip(target, pred):
            self.confusion_matrix[t, p] += 1
    
    def get_miou(self):
        """Compute mean IoU."""
        intersection = np.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - intersection
        iou = intersection / (union + 1e-10)
        return np.nanmean(iou), iou
    
    def get_pixel_accuracy(self):
        """Compute pixel accuracy."""
        acc = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + 1e-10)
        return acc
    
    def get_class_accuracy(self):
        """Compute per-class accuracy."""
        acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-10)
        return acc


def save_checkpoint(state, filename='checkpoint.pth'):
    """Save model checkpoint."""
    torch.save(state, filename)


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """
    Load model checkpoint.
    
    Args:
        model: model instance
        checkpoint_path: path to checkpoint file
        device: device to load checkpoint
        
    Returns:
        loaded state dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        return checkpoint
    else:
        model.load_state_dict(checkpoint)
        return {'state_dict': checkpoint}


def count_parameters(model):
    """Count number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_learning_rate(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
