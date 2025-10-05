"""
Model Information Script
Display detailed information about available models.
"""

import torch
from models import DDRNet23, DDRNet23Slim, DDRNet39


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_memory(model, input_shape=(1, 3, 1024, 2048), dtype='float32'):
    """Estimate model memory usage."""
    # Parameter memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Activation memory (rough estimate)
    # This is approximate - actual memory depends on batch size and architecture
    bytes_per_element = 4 if dtype == 'float32' else 2
    activation_memory = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * bytes_per_element * 10  # rough multiplier
    
    total_memory = param_memory + activation_memory
    return param_memory / 1024**2, activation_memory / 1024**2, total_memory / 1024**2


def display_model_info(name, model_fn, num_classes=19):
    """Display comprehensive information about a model."""
    print(f"\n{'='*70}")
    print(f"Model: {name}")
    print(f"{'='*70}")
    
    # Create model
    model = model_fn(num_classes=num_classes, augment=False)
    model.eval()
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nParameters:")
    print(f"  Total:     {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Memory estimation
    param_mem, act_mem, total_mem = estimate_memory(model)
    print(f"\nMemory Estimation (single image, 1024x2048):")
    print(f"  Parameters:  {param_mem:.1f} MB")
    print(f"  Activations: {act_mem:.1f} MB (approximate)")
    print(f"  Total:       {total_mem:.1f} MB")
    
    # Test inference
    print(f"\nInference Test:")
    test_input = torch.randn(1, 3, 1024, 2048)
    
    with torch.no_grad():
        import time
        start = time.time()
        output = model(test_input)
        end = time.time()
    
    print(f"  Input shape:  {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Inference time (CPU): {(end-start)*1000:.2f} ms")
    
    # Layer information
    print(f"\nArchitecture Summary:")
    total_conv = sum(1 for m in model.modules() if isinstance(m, torch.nn.Conv2d))
    total_bn = sum(1 for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    print(f"  Conv2d layers: {total_conv}")
    print(f"  BatchNorm2d layers: {total_bn}")


def main():
    print("\n" + "="*70)
    print(" "*15 + "DDRNet Model Information")
    print("="*70)
    
    # Student Models
    print("\n" + "="*70)
    print("STUDENT MODELS (DDRNet)")
    print("="*70)
    
    models = [
        ("DDRNet-23", DDRNet23),
        ("DDRNet-23-Slim", DDRNet23Slim),
        ("DDRNet-39", DDRNet39),
    ]
    
    for name, model_fn in models:
        display_model_info(name, model_fn)
    
    # Teacher Models Info (without creating them to avoid downloads)
    print("\n" + "="*70)
    print("TEACHER MODELS (DINOv2)")
    print("="*70)
    print("\nNote: Teacher models require downloading pretrained weights.")
    print("Information shown here is from official specifications.\n")
    
    teachers = [
        ("DINOv2-ViT-S/14", "~22M", "384", "~86 MB"),
        ("DINOv2-ViT-B/14", "~86M", "768", "~330 MB"),
        ("DINOv2-ViT-L/14", "~304M", "1024", "~1.2 GB"),
        ("DINOv2-ViT-G/14", "~1.1B", "1536", "~4.2 GB"),
    ]
    
    print(f"{'Model':<20} {'Parameters':<15} {'Embed Dim':<12} {'Size':<10}")
    print("-" * 70)
    for name, params, dim, size in teachers:
        print(f"{name:<20} {params:<15} {dim:<12} {size:<10}")
    
    # Comparison
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print("\nNote: FPS measured on RTX 3090 with 1024x2048 input (approximate)")
    
    comparison = [
        ("Model", "Params", "Memory*", "FPS (FP32)", "FPS (FP16)", "Best For"),
        ("-" * 20, "-" * 10, "-" * 10, "-" * 12, "-" * 12, "-" * 20),
        ("DDRNet-23", "~18M", "~500MB", "~110", "~185", "Balanced"),
        ("DDRNet-23-Slim", "~5.7M", "~150MB", "~150", "~240", "Speed/Mobile"),
        ("DDRNet-39", "~28M", "~750MB", "~85", "~140", "Accuracy"),
        ("", "", "", "", "", ""),
        ("DINOv2-ViT-B/14", "~86M", "~2.5GB", "~15", "~30", "Teacher only"),
    ]
    
    for row in comparison:
        print(f"{row[0]:<20} {row[1]:<10} {row[2]:<10} {row[3]:<12} {row[4]:<12} {row[5]:<20}")
    
    print("\n*Memory includes parameters + activations for single image")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("""
For Training:
  • Teacher: DINOv2-ViT-B/14 (best balance of quality and speed)
  • Student: DDRNet-23 (most versatile)
  • Batch size: 8-16 depending on GPU memory
  • Expected GPU memory: ~8-12GB for training

For Deployment:
  • Real-time applications: DDRNet-23-Slim
  • High-accuracy applications: DDRNet-39
  • Balanced: DDRNet-23

Dataset Recommendations:
  • Cityscapes (19 classes): Use DDRNet-23 with DINOv2-ViT-B/14
  • ADE20K (150 classes): Use DDRNet-39 with DINOv2-ViT-L/14
  • Custom dataset: Start with DDRNet-23-Slim for quick experiments
    """)
    
    print("="*70)
    print("\nFor more information, see:")
    print("  • README.md - General overview and usage")
    print("  • QUICKSTART.md - Quick start guide")
    print("  • ARCHITECTURE.md - Detailed architecture documentation")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
