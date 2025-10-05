"""
Inference script for Knowledge Distillation model.
Loads trained student model and performs semantic segmentation.
"""

import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

from models import DDRNet23, DDRNet23Slim, DDRNet39


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with trained model')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                      help='Path to input image')
    parser.add_argument('--output', type=str, default='output.png',
                      help='Path to output segmentation map')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda or cpu)')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config, checkpoint_path, device='cuda'):
    """Load trained student model."""
    num_classes = config['dataset']['num_classes']
    student_name = config['model']['student']['name']
    
    # Create student model (no augment for inference)
    if student_name == 'DDRNet23':
        model = DDRNet23(num_classes=num_classes, augment=False)
    elif student_name == 'DDRNet23Slim':
        model = DDRNet23Slim(num_classes=num_classes, augment=False)
    elif student_name == 'DDRNet39':
        model = DDRNet39(num_classes=num_classes, augment=False)
    else:
        raise ValueError(f"Unknown student model: {student_name}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path, config):
    """Load and preprocess image."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (W, H)
    
    # Get preprocessing parameters
    mean = config['image']['mean']
    std = config['image']['std']
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    return image_tensor, original_size


def colorize_segmentation(seg_map, num_classes):
    """
    Colorize segmentation map for visualization.
    
    Args:
        seg_map: [H, W] numpy array with class indices
        num_classes: number of classes
        
    Returns:
        [H, W, 3] RGB image
    """
    # Create color palette (you can customize this)
    palette = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]  # Background as black
    
    # Map class indices to colors
    colored = palette[seg_map]
    
    return colored


def inference(model, image_tensor, original_size, device='cuda'):
    """
    Perform inference on image.
    
    Args:
        model: trained model
        image_tensor: preprocessed image tensor [1, 3, H, W]
        original_size: (W, H) of original image
        device: device to use
        
    Returns:
        segmentation map [H, W] as numpy array
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Forward pass
        output = model(image_tensor)
        
        # Get predictions
        pred = output.argmax(dim=1)  # [1, H, W]
        pred = pred.squeeze(0)  # [H, W]
        
        # Resize to original size
        pred = pred.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
        pred = F.interpolate(pred, size=(original_size[1], original_size[0]),
                           mode='nearest')
        pred = pred.squeeze().long().cpu().numpy()
    
    return pred


def main():
    args = parse_args()
    config = load_config(args.config)
    
    print(f'Loading model from {args.checkpoint}...')
    model = load_model(config, args.checkpoint, args.device)
    
    print(f'Loading image from {args.image}...')
    image_tensor, original_size = preprocess_image(args.image, config)
    
    print('Running inference...')
    seg_map = inference(model, image_tensor, original_size, args.device)
    
    print('Generating visualization...')
    num_classes = config['dataset']['num_classes']
    colored_seg = colorize_segmentation(seg_map, num_classes)
    
    # Save output
    output_image = Image.fromarray(colored_seg)
    output_image.save(args.output)
    print(f'Segmentation saved to {args.output}')
    
    # Also save raw segmentation map
    raw_output = args.output.replace('.png', '_raw.npy')
    np.save(raw_output, seg_map)
    print(f'Raw segmentation map saved to {raw_output}')


if __name__ == '__main__':
    main()
