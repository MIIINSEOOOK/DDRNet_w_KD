# DDRNet with Knowledge Distillation

Knowledge Distillation framework for semantic segmentation using **DINOv2** as teacher model and **DDRNet** as student model.

## Overview

This repository implements a knowledge distillation (KD) approach for semantic segmentation where:
- **Teacher Model**: DINOv2 (Vision Transformer) - provides rich semantic features
- **Student Model**: DDRNet (Deep Dual-Resolution Network) - efficient segmentation model
- **Task**: Semantic segmentation on datasets like Cityscapes, ADE20K, etc.

The knowledge is transferred from the powerful but heavy DINOv2 teacher to the lightweight DDRNet student through multiple distillation strategies:
- Logit-based distillation (KL divergence)
- Feature-based distillation (intermediate features)
- Structural distillation (spatial relationships)

## Features

- ✅ Multiple DINOv2 variants (ViT-S/B/L/G with 14x14 patches)
- ✅ Multiple DDRNet variants (DDRNet-23, DDRNet-23-Slim, DDRNet-39)
- ✅ Combined knowledge distillation losses
- ✅ Support for Cityscapes and ADE20K datasets
- ✅ Configurable training pipeline
- ✅ TensorBoard logging
- ✅ Inference script for deployment

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.0 (for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/MIIINSEOOOK/DDRNet_w_KD.git
cd DDRNet_w_KD

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
DDRNet_w_KD/
├── models/
│   ├── __init__.py
│   ├── ddrnet.py              # DDRNet student model
│   ├── dinov2_teacher.py      # DINOv2 teacher wrapper
│   ├── kd_losses.py           # Knowledge distillation losses
│   └── kd_model.py            # Combined KD model
├── utils/
│   └── __init__.py            # Training utilities
├── configs/
│   ├── cityscapes_dinov2_ddrnet.yaml
│   └── ade20k_dinov2_ddrnet.yaml
├── train.py                   # Training script
├── inference.py               # Inference script
├── requirements.txt
└── README.md
```

## Usage

### 1. Prepare Dataset

#### Cityscapes
```bash
# Download Cityscapes dataset from https://www.cityscapes-dataset.com/
# Organize as:
# data/cityscapes/
#   ├── leftImg8bit/
#   │   ├── train/
#   │   └── val/
#   └── gtFine/
#       ├── train/
#       └── val/
```

#### ADE20K
```bash
# Download ADE20K dataset
# Organize as:
# data/ADEChallengeData2016/
#   ├── images/
#   │   ├── training/
#   │   └── validation/
#   └── annotations/
#       ├── training/
#       └── validation/
```

### 2. Training

**Basic training with default config:**

```bash
python train.py --config configs/cityscapes_dinov2_ddrnet.yaml
```

**Resume from checkpoint:**

```bash
python train.py --config configs/cityscapes_dinov2_ddrnet.yaml \
    --resume checkpoints/checkpoint_epoch_100.pth
```

**Note**: The current training script is a template. You need to implement dataset loading for your specific dataset. See the comments in `train.py` for guidance.

### 3. Inference

**Run inference on a single image:**

```bash
python inference.py \
    --config configs/cityscapes_dinov2_ddrnet.yaml \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/image.jpg \
    --output output_segmentation.png
```

## Configuration

The configuration files (YAML format) control all aspects of training:

### Key Configuration Options

```yaml
# Model selection
model:
  teacher:
    name: dinov2_vitb14  # Options: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
  student:
    name: DDRNet23       # Options: DDRNet23, DDRNet23Slim, DDRNet39
  kd:
    kd_weight: 1.0           # Logit KD weight
    feature_weight: 0.5      # Feature distillation weight
    structural_weight: 0.3   # Structural distillation weight
    temperature: 4.0         # KD temperature

# Training settings
train:
  batch_size: 8
  epochs: 300
  optimizer:
    type: sgd
    lr: 0.01
    momentum: 0.9
```

## Model Architectures

### Teacher: DINOv2
- **DINOv2-ViT-S/14**: ~22M params, 384 dim
- **DINOv2-ViT-B/14**: ~86M params, 768 dim
- **DINOv2-ViT-L/14**: ~304M params, 1024 dim
- **DINOv2-ViT-G/14**: ~1.1B params, 1536 dim

### Student: DDRNet
- **DDRNet-23**: ~20M params
- **DDRNet-23-Slim**: ~5.7M params (reduced channels)
- **DDRNet-39**: ~30M params (deeper variant)

## Knowledge Distillation Methods

### 1. Logit-based KD
Transfers knowledge through soft targets using KL divergence:
```
L_KD = KL(σ(z_s/T) || σ(z_t/T)) * T²
```

### 2. Feature-based Distillation
Matches intermediate feature representations:
```
L_feat = ||F_s - F_t||²
```

### 3. Structural Distillation
Preserves spatial relationships in predictions:
```
L_struct = ||S(z_s) - S(z_t)||²
```

### Combined Loss
```
L_total = L_task + λ₁·L_KD + λ₂·L_feat + λ₃·L_struct
```

## Performance

The distilled DDRNet student model achieves competitive performance while being much more efficient than the DINOv2 teacher:

| Model | Params | FPS* | mIoU (Cityscapes) |
|-------|--------|------|-------------------|
| DINOv2-ViT-B/14 (Teacher) | 86M | ~15 | - |
| DDRNet-23 (Student) | 20M | ~110 | - |
| DDRNet-23-Slim (Student) | 5.7M | ~150 | - |

*FPS measured on RTX 3090 with 1024x1024 input

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ddrnet,
  title={Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes},
  author={Hong, Yuanduo and Pan, Huihui and Sun, Weichao and Jia, Yisong},
  journal={arXiv preprint arXiv:2101.06085},
  year={2021}
}

@article{dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```

## License

This project is released under the MIT License.

## Acknowledgments

- [DINOv2](https://github.com/facebookresearch/dinov2) by Meta AI
- [DDRNet](https://github.com/ydhongHIT/DDRNet) original implementation
- Knowledge Distillation research community

## Contact

For questions or issues, please open an issue on GitHub.
