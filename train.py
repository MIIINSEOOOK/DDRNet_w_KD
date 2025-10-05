"""
Training script for Knowledge Distillation with DINOv2 teacher and DDRNet student.
"""

import os
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import DDRNet23, DDRNet23Slim, DDRNet39
from models.dinov2_teacher import dinov2_vitb14_teacher, dinov2_vits14_teacher
from models.dinov2_teacher import dinov2_vitl14_teacher, dinov2_vitg14_teacher
from models.kd_model import KnowledgeDistillationModel
from utils import (
    get_optimizer, get_scheduler, AverageMeter, SegmentationMetrics,
    save_checkpoint, count_parameters, get_learning_rate
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Knowledge Distillation Model')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint for resuming training')
    parser.add_argument('--local_rank', type=int, default=0,
                      help='Local rank for distributed training')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model(config):
    """
    Create teacher and student models based on config.
    
    Returns:
        KnowledgeDistillationModel instance
    """
    num_classes = config['dataset']['num_classes']
    
    # Create student model
    student_name = config['model']['student']['name']
    student_kwargs = {
        'num_classes': num_classes,
        'augment': config['model']['student'].get('augment', True)
    }
    
    if student_name == 'DDRNet23':
        student = DDRNet23(**student_kwargs)
    elif student_name == 'DDRNet23Slim':
        student = DDRNet23Slim(**student_kwargs)
    elif student_name == 'DDRNet39':
        student = DDRNet39(**student_kwargs)
    else:
        raise ValueError(f"Unknown student model: {student_name}")
    
    # Create teacher model
    teacher_name = config['model']['teacher']['name']
    teacher_kwargs = {
        'num_classes': num_classes,
        'freeze_backbone': config['model']['teacher'].get('freeze_backbone', True),
        'pretrained': config['model']['teacher'].get('pretrained', True)
    }
    
    if teacher_name == 'dinov2_vits14':
        teacher = dinov2_vits14_teacher(**teacher_kwargs)
    elif teacher_name == 'dinov2_vitb14':
        teacher = dinov2_vitb14_teacher(**teacher_kwargs)
    elif teacher_name == 'dinov2_vitl14':
        teacher = dinov2_vitl14_teacher(**teacher_kwargs)
    elif teacher_name == 'dinov2_vitg14':
        teacher = dinov2_vitg14_teacher(**teacher_kwargs)
    else:
        raise ValueError(f"Unknown teacher model: {teacher_name}")
    
    # Create KD model
    kd_config = config['model']['kd']
    model = KnowledgeDistillationModel(
        teacher=teacher,
        student=student,
        num_classes=num_classes,
        **kd_config
    )
    
    return model


def train_epoch(model, dataloader, optimizer, scheduler, epoch, config, writer=None):
    """Train for one epoch."""
    model.train()
    
    # Only train student (teacher is frozen)
    model.teacher.eval()
    
    loss_meter = AverageMeter()
    task_loss_meter = AverageMeter()
    kd_loss_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.cuda()
        labels = labels.cuda()
        
        # Forward pass
        outputs = model(images, labels, return_loss=True)
        loss = outputs['loss']
        losses = outputs['losses']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update meters
        loss_meter.update(loss.item(), images.size(0))
        task_loss_meter.update(losses['task_loss'].item(), images.size(0))
        kd_loss_meter.update(losses['total_kd_loss'].item(), images.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'task': f'{task_loss_meter.avg:.4f}',
            'kd': f'{kd_loss_meter.avg:.4f}',
            'lr': f'{get_learning_rate(optimizer):.6f}'
        })
        
        # Logging
        if writer is not None and batch_idx % config['logging']['log_interval'] == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/task_loss', losses['task_loss'].item(), global_step)
            writer.add_scalar('train/kd_loss', losses['total_kd_loss'].item(), global_step)
            writer.add_scalar('train/lr', get_learning_rate(optimizer), global_step)
    
    return loss_meter.avg


def validate(model, dataloader, epoch, config, writer=None):
    """Validate the model."""
    model.eval()
    
    num_classes = config['dataset']['num_classes']
    ignore_index = config['dataset']['ignore_index']
    
    metrics = SegmentationMetrics(num_classes, ignore_index)
    loss_meter = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Validation')
        
        for images, labels in pbar:
            images = images.cuda()
            labels = labels.cuda()
            
            # Forward pass (with loss computation if labels available)
            outputs = model(images, labels, return_loss=True)
            loss = outputs['loss']
            student_pred = outputs['student_pred']
            
            # Get predictions
            pred = student_pred.argmax(dim=1)
            
            # Update metrics
            metrics.update(pred.cpu().numpy(), labels.cpu().numpy())
            loss_meter.update(loss.item(), images.size(0))
            
            # Update progress bar
            miou, _ = metrics.get_miou()
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'mIoU': f'{miou:.4f}'
            })
    
    # Compute final metrics
    miou, class_iou = metrics.get_miou()
    pixel_acc = metrics.get_pixel_accuracy()
    
    print(f'\nValidation Results:')
    print(f'  Loss: {loss_meter.avg:.4f}')
    print(f'  mIoU: {miou:.4f}')
    print(f'  Pixel Accuracy: {pixel_acc:.4f}')
    
    # Log to tensorboard
    if writer is not None:
        writer.add_scalar('val/loss', loss_meter.avg, epoch)
        writer.add_scalar('val/mIoU', miou, epoch)
        writer.add_scalar('val/pixel_accuracy', pixel_acc, epoch)
    
    return miou, loss_meter.avg


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config.get('seed', 42))
    
    # Setup CUDA
    torch.backends.cudnn.benchmark = config['cudnn'].get('benchmark', True)
    torch.backends.cudnn.deterministic = config['cudnn'].get('deterministic', False)
    
    # Create directories
    os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    # Create model
    print('Creating model...')
    model = get_model(config)
    model = model.cuda()
    
    print(f'Student parameters: {count_parameters(model.student):,}')
    print(f'Teacher parameters: {count_parameters(model.teacher):,}')
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(
        model.student,  # Only optimize student
        **config['train']['optimizer']
    )
    
    # Calculate total iterations for scheduler
    # Note: This assumes dataloader length is known
    # In practice, you would multiply by actual dataloader length
    total_iters = config['train']['epochs'] * 1000  # Placeholder
    
    scheduler = get_scheduler(
        optimizer,
        max_iters=total_iters,
        **config['train']['scheduler']
    )
    
    # Resume from checkpoint if specified
    start_epoch = config['train']['start_epoch']
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume)
        model.student.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Setup tensorboard
    writer = None
    if config['logging'].get('use_tensorboard', True):
        writer = SummaryWriter(config['logging']['log_dir'])
    
    # Training loop
    print('\nStarting training...')
    print('NOTE: This is a training template. You need to provide dataloaders.')
    print('Please implement dataset loading for your specific dataset.\n')
    
    # Placeholder for dataloader creation
    # In practice, you would create actual dataloaders here
    # train_loader = create_dataloader(config, 'train')
    # val_loader = create_dataloader(config, 'val')
    
    print('Training script is ready. To complete the implementation:')
    print('1. Implement dataset loading (Cityscapes, ADE20K, etc.)')
    print('2. Create train and validation dataloaders')
    print('3. Run the training loop')
    print('4. Save checkpoints periodically')
    
    # Example training loop (commented out - needs dataloaders)
    # best_miou = 0.0
    # for epoch in range(start_epoch, config['train']['epochs']):
    #     # Train
    #     train_loss = train_epoch(model, train_loader, optimizer, scheduler, 
    #                             epoch, config, writer)
    #     
    #     # Validate
    #     if (epoch + 1) % config['val']['eval_interval'] == 0:
    #         miou, val_loss = validate(model, val_loader, epoch, config, writer)
    #         
    #         # Save checkpoint
    #         if miou > best_miou:
    #             best_miou = miou
    #             save_checkpoint({
    #                 'epoch': epoch,
    #                 'state_dict': model.student.state_dict(),
    #                 'optimizer': optimizer.state_dict(),
    #                 'best_miou': best_miou,
    #             }, os.path.join(config['checkpoint']['save_dir'], 'best_model.pth'))
    #     
    #     # Save periodic checkpoint
    #     if (epoch + 1) % config['checkpoint']['save_interval'] == 0:
    #         save_checkpoint({
    #             'epoch': epoch,
    #             'state_dict': model.student.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'miou': miou if 'miou' in locals() else 0.0,
    #         }, os.path.join(config['checkpoint']['save_dir'], f'checkpoint_epoch_{epoch}.pth'))
    
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
