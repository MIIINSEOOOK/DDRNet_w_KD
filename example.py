"""
Example script demonstrating how to use the Knowledge Distillation models.
This shows the basic usage without requiring a full dataset.
"""

import torch
from models import DDRNet23, DDRNet23Slim, DDRNet39
from models.dinov2_teacher import dinov2_vitb14_teacher, dinov2_vits14_teacher
from models.kd_model import KnowledgeDistillationModel


def example_student_model():
    """Example: Create and use student model (DDRNet)"""
    print("=" * 60)
    print("Example 1: DDRNet Student Model")
    print("=" * 60)
    
    # Create DDRNet-23 model for 19 classes (Cityscapes)
    model = DDRNet23(num_classes=19, augment=False)
    model.eval()
    
    # Create dummy input (batch_size=2, 3 channels, 1024x2048)
    dummy_input = torch.randn(2, 3, 1024, 2048)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: [batch_size, num_classes, H/8, W/8]")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()


def example_teacher_model():
    """Example: Create and use teacher model (DINOv2)"""
    print("=" * 60)
    print("Example 2: DINOv2 Teacher Model")
    print("=" * 60)
    print("Note: This will download pretrained DINOv2 weights (~330MB)")
    print("Skipping actual model creation to avoid download in this demo.")
    print("To use the teacher model, uncomment the code below:")
    print()
    print("# model = dinov2_vitb14_teacher(num_classes=19)")
    print("# model.eval()")
    print("# dummy_input = torch.randn(1, 3, 518, 518)")
    print("# output = model(dummy_input)")
    print("# print(f'Output: {output.keys()}')")
    print()


def example_kd_model():
    """Example: Create and use the combined KD model"""
    print("=" * 60)
    print("Example 3: Knowledge Distillation Model")
    print("=" * 60)
    print("This example shows how to create the combined KD model.")
    print("Actual training requires a dataset.")
    print()
    
    # Create student model
    student = DDRNet23(num_classes=19, augment=True)
    
    print("Student model created.")
    print("Teacher model would be: dinov2_vitb14_teacher(num_classes=19)")
    print()
    
    # Show how to create KD model (commented to avoid downloading)
    print("To create the full KD model:")
    print()
    print("teacher = dinov2_vitb14_teacher(num_classes=19)")
    print("kd_model = KnowledgeDistillationModel(")
    print("    teacher=teacher,")
    print("    student=student,")
    print("    num_classes=19,")
    print("    kd_weight=1.0,")
    print("    feature_weight=0.5,")
    print("    structural_weight=0.3,")
    print("    temperature=4.0")
    print(")")
    print()
    
    # Show forward pass for training
    print("Training forward pass:")
    print()
    print("images = torch.randn(4, 3, 1024, 2048)")
    print("labels = torch.randint(0, 19, (4, 1024, 2048))")
    print("outputs = kd_model(images, labels, return_loss=True)")
    print("loss = outputs['loss']")
    print("loss.backward()")
    print()


def example_model_variants():
    """Example: Different model variants"""
    print("=" * 60)
    print("Example 4: Model Variants")
    print("=" * 60)
    
    # Different DDRNet variants
    print("DDRNet Student Variants:")
    print("-" * 40)
    
    models = {
        'DDRNet-23': DDRNet23(num_classes=19),
        'DDRNet-23-Slim': DDRNet23Slim(num_classes=19),
        'DDRNet-39': DDRNet39(num_classes=19),
    }
    
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:20s}: {params:,} parameters")
    
    print()
    print("DINOv2 Teacher Variants:")
    print("-" * 40)
    print("dinov2_vits14 : ~22M parameters,  384 dim")
    print("dinov2_vitb14 : ~86M parameters,  768 dim")
    print("dinov2_vitl14 : ~304M parameters, 1024 dim")
    print("dinov2_vitg14 : ~1.1B parameters, 1536 dim")
    print()


def example_inference():
    """Example: Inference with a trained model"""
    print("=" * 60)
    print("Example 5: Inference")
    print("=" * 60)
    
    # Create model
    model = DDRNet23Slim(num_classes=19, augment=False)
    model.eval()
    
    # Simulate loading checkpoint
    print("To load a trained model:")
    print()
    print("checkpoint = torch.load('checkpoints/best_model.pth')")
    print("model.load_state_dict(checkpoint['state_dict'])")
    print()
    
    # Inference
    print("Inference on an image:")
    print()
    print("import torchvision.transforms as transforms")
    print("from PIL import Image")
    print()
    print("# Load and preprocess image")
    print("image = Image.open('image.jpg').convert('RGB')")
    print("transform = transforms.Compose([")
    print("    transforms.ToTensor(),")
    print("    transforms.Normalize(mean=[0.485, 0.456, 0.406],")
    print("                       std=[0.229, 0.224, 0.225])")
    print("])")
    print("image_tensor = transform(image).unsqueeze(0)")
    print()
    print("# Run inference")
    print("with torch.no_grad():")
    print("    output = model(image_tensor)")
    print("    pred = output.argmax(dim=1)")
    print()
    
    # Show what inference looks like with dummy data
    print("Demo with dummy data:")
    dummy_input = torch.randn(1, 3, 512, 1024)
    with torch.no_grad():
        output = model(dummy_input)
        pred = output.argmax(dim=1)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Prediction shape: {pred.shape}")
    print(f"Unique classes in prediction: {pred.unique().tolist()}")
    print()


def main():
    """Run all examples"""
    print("\n")
    print("#" * 60)
    print("# DDRNet Knowledge Distillation - Usage Examples")
    print("#" * 60)
    print("\n")
    
    example_student_model()
    example_teacher_model()
    example_kd_model()
    example_model_variants()
    example_inference()
    
    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Prepare your dataset (Cityscapes, ADE20K, etc.)")
    print("2. Update the config file in configs/")
    print("3. Implement dataset loading in train.py")
    print("4. Run training: python train.py --config configs/cityscapes_dinov2_ddrnet.yaml")
    print("5. Run inference: python inference.py --config ... --checkpoint ... --image ...")
    print()


if __name__ == '__main__':
    main()
