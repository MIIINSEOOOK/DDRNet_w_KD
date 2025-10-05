# Quick Start Guide

This guide will help you get started with the DINOv2-DDRNet Knowledge Distillation framework.

## Installation

```bash
# Clone the repository
git clone https://github.com/MIIINSEOOOK/DDRNet_w_KD.git
cd DDRNet_w_KD

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Test

Run the example script to verify installation:

```bash
python example.py
```

This will:
- Create DDRNet student models
- Show model architectures and parameter counts
- Demonstrate basic usage patterns
- Run inference with dummy data

## Model Overview

### Student Models (DDRNet)
- **DDRNet-23**: ~18M params - balanced performance/speed
- **DDRNet-23-Slim**: ~4.6M params - fastest, good for edge devices
- **DDRNet-39**: ~28M params - best accuracy

### Teacher Models (DINOv2)
- **DINOv2-ViT-S/14**: ~22M params, 384 dim - fastest teacher
- **DINOv2-ViT-B/14**: ~86M params, 768 dim - recommended
- **DINOv2-ViT-L/14**: ~304M params, 1024 dim - high accuracy
- **DINOv2-ViT-G/14**: ~1.1B params, 1536 dim - best quality

## Basic Usage

### 1. Create Models

```python
from models import DDRNet23
from models.dinov2_teacher import dinov2_vitb14_teacher
from models.kd_model import KnowledgeDistillationModel

# Student model
student = DDRNet23(num_classes=19, augment=True)

# Teacher model (downloads pretrained weights)
teacher = dinov2_vitb14_teacher(num_classes=19, freeze_backbone=True)

# Combined KD model
kd_model = KnowledgeDistillationModel(
    teacher=teacher,
    student=student,
    num_classes=19,
    kd_weight=1.0,
    feature_weight=0.5,
    structural_weight=0.3,
    temperature=4.0
)
```

### 2. Training (Pseudo-code)

```python
# Prepare data
train_loader = ...  # Your dataset loader
val_loader = ...

# Setup optimizer
from utils import get_optimizer, get_scheduler
optimizer = get_optimizer(kd_model.student, optimizer_type='sgd', lr=0.01)
scheduler = get_scheduler(optimizer, scheduler_type='poly', max_iters=total_iters)

# Training loop
for epoch in range(epochs):
    for images, labels in train_loader:
        # Forward pass
        outputs = kd_model(images, labels, return_loss=True)
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
```

### 3. Inference

```python
# Load trained student model
from models import DDRNet23
import torch

model = DDRNet23(num_classes=19, augment=False)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Run inference
with torch.no_grad():
    output = model(image_tensor)
    prediction = output.argmax(dim=1)
```

## Configuration

Edit YAML config files in `configs/` directory:

```yaml
# configs/cityscapes_dinov2_ddrnet.yaml
model:
  teacher:
    name: dinov2_vitb14
  student:
    name: DDRNet23
  kd:
    kd_weight: 1.0
    feature_weight: 0.5
    structural_weight: 0.3
    temperature: 4.0

train:
  batch_size: 8
  epochs: 300
  optimizer:
    type: sgd
    lr: 0.01
```

## Dataset Setup

### Cityscapes

```
data/cityscapes/
├── leftImg8bit/
│   ├── train/
│   └── val/
└── gtFine/
    ├── train/
    └── val/
```

### ADE20K

```
data/ADEChallengeData2016/
├── images/
│   ├── training/
│   └── validation/
└── annotations/
    ├── training/
    └── validation/
```

## Command Line Usage

### Training

```bash
# Train on Cityscapes
python train.py --config configs/cityscapes_dinov2_ddrnet.yaml

# Resume from checkpoint
python train.py --config configs/cityscapes_dinov2_ddrnet.yaml \
    --resume checkpoints/checkpoint_epoch_100.pth

# Train on ADE20K
python train.py --config configs/ade20k_dinov2_ddrnet.yaml
```

### Inference

```bash
# Single image inference
python inference.py \
    --config configs/cityscapes_dinov2_ddrnet.yaml \
    --checkpoint checkpoints/best_model.pth \
    --image test_image.jpg \
    --output result.png
```

## Key Features

### Knowledge Distillation Losses

1. **Logit-based KD**: Transfers soft targets using temperature scaling
2. **Feature Distillation**: Matches intermediate feature representations
3. **Structural Distillation**: Preserves spatial relationships

### Advantages

- **Higher Accuracy**: Learn from powerful DINOv2 teacher
- **Efficient Deployment**: Use lightweight DDRNet student in production
- **Flexible Training**: Configurable distillation strategies
- **Easy Integration**: Compatible with standard PyTorch training loops

## Troubleshooting

### Out of Memory

Reduce batch size in config:
```yaml
train:
  batch_size: 4  # Reduce from 8
```

### Slow Training

Use smaller teacher model:
```yaml
model:
  teacher:
    name: dinov2_vits14  # Instead of dinov2_vitb14
```

Or use slimmer student:
```yaml
model:
  student:
    name: DDRNet23Slim  # Instead of DDRNet23
```

### DINOv2 Download Issues

DINOv2 weights are downloaded from torch.hub. If you have network issues:
1. Download weights manually from [DINOv2 releases](https://github.com/facebookresearch/dinov2)
2. Place in torch hub cache directory
3. Or set `pretrained=False` and provide your own weights

## Next Steps

1. ✅ Verify installation with `python example.py`
2. 📊 Prepare your dataset (Cityscapes or ADE20K)
3. ⚙️ Customize config file for your needs
4. 🚀 Implement dataset loader in `train.py`
5. 🏋️ Train your model
6. 📈 Monitor with TensorBoard
7. 🎯 Run inference on test images

## Resources

- [DDRNet Paper](https://arxiv.org/abs/2101.06085)
- [DINOv2 Paper](https://arxiv.org/abs/2304.07193)
- [Knowledge Distillation Survey](https://arxiv.org/abs/2006.05525)
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [ADE20K Dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

## Support

For issues, questions, or contributions, please open an issue on GitHub.
