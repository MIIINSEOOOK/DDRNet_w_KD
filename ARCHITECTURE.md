# Architecture Documentation

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Knowledge Distillation Framework             │
│                                                                 │
│  ┌──────────────┐                         ┌─────────────────┐  │
│  │              │   Knowledge Transfer    │                 │  │
│  │   DINOv2     │  ─────────────────────> │     DDRNet      │  │
│  │  (Teacher)   │                         │    (Student)    │  │
│  │              │   • Logit KD            │                 │  │
│  │  Frozen      │   • Feature KD          │   Trainable     │  │
│  │  Pretrained  │   • Structural KD       │                 │  │
│  └──────────────┘                         └─────────────────┘  │
│                                                                 │
│  Input Image ──> Teacher (frozen) ──┐                          │
│       │                              ├──> KD Losses ──> Total  │
│       └────────> Student ────────────┘                   Loss  │
│                      │                                          │
│                      └──> Task Loss (Cross Entropy)            │
└─────────────────────────────────────────────────────────────────┘
```

## DINOv2 Teacher Architecture

```
Input Image [B, 3, H, W]
     │
     ├──> Patch Embedding [B, num_patches+1, embed_dim]
     │         │
     │         └──> Vision Transformer Blocks (12/12/24/40 layers)
     │                   │
     │                   ├──> Layer Normalization
     │                   └──> Self-Attention + MLP
     │
     └──> Patch Tokens [B, num_patches, embed_dim]
              │
              ├──> Reshape to [B, embed_dim, H/14, W/14]
              │
              ├──> Interpolate to [B, embed_dim, H/8, W/8]
              │
              ├──> Segmentation Head
              │         │
              │         ├──> Conv3x3 + BN + ReLU
              │         └──> Conv1x1 -> [B, num_classes, H/8, W/8]
              │
              └──> Feature Projection (for KD)
                        │
                        ├──> Conv1x1 -> [B, 256, H/8, W/8]
                        └──> Conv1x1 -> [B, 512, H/8, W/8]

Output:
  • Logits: [B, num_classes, H/8, W/8]
  • Features: List[[B, 256, H/8, W/8], [B, 512, H/8, W/8]]
```

### DINOv2 Variants

| Model | Layers | Embed Dim | Params | Patch Size |
|-------|--------|-----------|--------|------------|
| ViT-S/14 | 12 | 384 | ~22M | 14×14 |
| ViT-B/14 | 12 | 768 | ~86M | 14×14 |
| ViT-L/14 | 24 | 1024 | ~304M | 14×14 |
| ViT-G/14 | 40 | 1536 | ~1.1B | 14×14 |

## DDRNet Student Architecture

```
Input Image [B, 3, H, W]
     │
     ├──> Stem: Conv3x3(stride=2) -> Conv3x3(stride=2)
     │         Output: [B, 64, H/4, W/4]
     │
     ├──> Stage 1 (Layer1): BasicBlock × 2
     │         Output: [B, 64, H/4, W/4]
     │
     ├──> Stage 2 (Layer2): BasicBlock × 2, stride=2
     │         Output: [B, 128, H/8, W/8]
     │
     └──> Dual Branch Processing
              │
              ├─────────────────────────────────────────┐
              │                                         │
        ┌─────▼─────┐                            ┌────▼────┐
        │ Low-Res   │                            │High-Res │
        │ Branch    │                            │ Branch  │
        │           │                            │         │
        │ Layer3    │───Compression──>┌──────┐  │Layer3_  │
        │ [256,     │                 │ Add  │  │ [128,   │
        │  H/16,    │                 │      │  │  H/8,   │
        │  W/16]    │                 └──┬───┘  │  W/8]   │
        │           │                    │      │         │
        │ Layer4    │───Compression──>┌──▼───┐  │Layer4_  │
        │ [512,     │                 │ Add  │  │ [128,   │
        │  H/32,    │                 │      │  │  H/8,   │
        │  W/32]    │                 └──┬───┘  │  W/8]   │
        │           │                    │      │         │
        │ Layer5    │                    │      │Layer5_  │
        │ [1024,    │                    │      │ [128,   │
        │  H/64,    │                    │      │  H/8,   │
        │  W/64]    │                    │      │  W/8]   │
        │           │                    │      │         │
        │ DAPPM     │                    │      │SegHead  │
        │ [256,     │                    │      │ (aux)   │
        │  H/64,    │                    │      │         │
        │  W/64]    │                    │      │         │
        │           │                    │      │         │
        │ Final     │                    │      │         │
        │ Head      │                    │      │         │
        └─────┬─────┘                    │      └────┬────┘
              │                          │           │
              └──────────────────────────┴───────────┘
                          │              │
                    Main Output    Aux Output
                 [B, num_classes,  [B, num_classes,
                    H/8, W/8]         H/8, W/8]
```

### DDRNet Components

#### 1. Basic Block
```
Input [B, C_in, H, W]
  │
  ├──> Conv3x3(stride) + BN + ReLU
  │         │
  │         └──> Conv3x3 + BN
  │                   │
  └──> Downsample ────┤
                      │
                    Add + ReLU
                      │
                   Output
```

#### 2. DAPPM (Deep Aggregation Pyramid Pooling Module)
```
Input [B, C, H, W]
  │
  ├──> AvgPool(5×5) ──> Conv1x1 ──> Scale1
  ├──> AvgPool(9×9) ──> Conv1x1 ──> Scale2
  ├──> AvgPool(17×17) ─> Conv1x1 ──> Scale3
  ├──> AdaptiveAvgPool ─> Conv1x1 ──> Scale4
  └──> Conv1x1 ──────────────────> Scale0
         │
         └──> Aggregate all scales
                   │
              Compression
                   │
                Output
```

### DDRNet Variants

| Model | Blocks | Params | FPS* | Best For |
|-------|--------|--------|------|----------|
| DDRNet-23 | [2,2,2,2] | ~18M | ~110 | Balanced |
| DDRNet-23-Slim | [2,2,2,2] | ~5.7M | ~150 | Speed |
| DDRNet-39 | [3,4,6,3] | ~28M | ~85 | Accuracy |

*FPS on RTX 3090, 1024×1024 input

## Knowledge Distillation Losses

### 1. Logit-based KD Loss

```
Teacher Logits [B, C, H, W]  ──> Softmax(T) ──┐
                                               │
Student Logits [B, C, H, W]  ──> LogSoftmax(T)├──> KL Divergence × T²
                                               │
                                               └──> L_KD
```

Temperature T=4.0 (default)

### 2. Feature Distillation Loss

```
Teacher Features [B, C_t, H, W] ──> Normalize ──┐
                                                 │
Student Features [B, C_s, H, W] ──> Adapt ──────┤
                                    Normalize    │
                                                 ├──> MSE Loss
                                                 │
                                                 └──> L_feat
```

### 3. Structural Distillation Loss

```
Teacher Logits [B, C, H, W]
  │
  ├──> Softmax
  │       │
  │       └──> Flatten [B, C, N] where N=H×W
  │               │
  │               └──> Similarity Matrix [B, N, N]
  │                           │
Student Logits              │
  │                         │
  ├──> Softmax              │
  │       │                 │
  │       └──> Flatten      │
  │               │         │
  │               └──> Similarity Matrix
  │                           │
  │                           ├──> MSE Loss
  │                           │
  │                           └──> L_struct
```

### Combined Loss Function

```
L_total = w_task × L_task 
        + w_kd × L_KD 
        + w_feat × L_feat 
        + w_struct × L_struct

where:
  L_task = CrossEntropy(student_pred, labels)
  w_task = 1.0 (default)
  w_kd = 1.0 (default)
  w_feat = 0.5 (default)
  w_struct = 0.3 (default)
```

## Data Flow During Training

```
┌─────────────────────────────────────────────────────────┐
│                      Training Step                      │
└─────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
        Images [B,3,H,W]            Labels [B,H,W]
              │                           │
              ├───────────┬───────────────┤
              │           │               │
        ┌─────▼──────┐    │               │
        │  Teacher   │    │               │
        │  (frozen)  │    │               │
        │            │    │               │
        │  Forward   │    │               │
        └─────┬──────┘    │               │
              │           │               │
      Teacher Outputs     │               │
        • logits          │               │
        • features   ┌────▼────┐          │
              │      │ Student │          │
              │      │         │          │
              │      │ Forward │          │
              │      └────┬────┘          │
              │           │               │
              │    Student Outputs        │
              │      • logits             │
              │      • aux_logits         │
              │           │               │
              └──────┬────┴───────────────┤
                     │                    │
              ┌──────▼────────┐    ┌──────▼──────┐
              │  KD Losses    │    │  Task Loss  │
              │               │    │             │
              │ • L_KD        │    │ CrossEntropy│
              │ • L_feat      │    │             │
              │ • L_struct    │    └──────┬──────┘
              └──────┬────────┘           │
                     │                    │
                     └──────────┬─────────┘
                                │
                         ┌──────▼──────┐
                         │ Total Loss  │
                         └──────┬──────┘
                                │
                         Backward Pass
                                │
                         Update Student
```

## Performance Characteristics

### Memory Usage

| Component | Memory (FP32) | Memory (FP16) |
|-----------|---------------|---------------|
| DINOv2-ViT-B/14 | ~2.5GB | ~1.3GB |
| DDRNet-23 | ~500MB | ~250MB |
| Batch (8×1024×2048) | ~6GB | ~3GB |
| **Total Training** | **~9GB** | **~4.5GB** |

### Inference Speed

Input: 1024×2048, RTX 3090

| Model | FPS (FP32) | FPS (FP16) | Latency |
|-------|------------|------------|---------|
| DINOv2-ViT-B/14 | ~15 | ~30 | 66ms |
| DDRNet-23 | ~110 | ~185 | 9ms |
| DDRNet-23-Slim | ~150 | ~240 | 6.7ms |

## Implementation Details

### Initialization
- Teacher: Pretrained DINOv2 weights (frozen)
- Student: Random initialization (ImageNet pretrain optional)
- KD adapters: Xavier uniform initialization

### Training Strategy
- Epochs: 300 (Cityscapes), 150 (ADE20K)
- Batch size: 8-16
- Optimizer: SGD with momentum 0.9
- Learning rate: 0.01-0.02 (poly decay, power=0.9)
- Weight decay: 5e-4 (Cityscapes), 1e-4 (ADE20K)

### Data Augmentation
- Random horizontal flip
- Random scaling [0.5, 2.0]
- Random crop to fixed size
- Color jitter
- Normalization: ImageNet mean/std

### Evaluation
- Metric: Mean IoU (mIoU)
- Multi-scale testing (optional)
- Slide inference for large images (optional)

## File Structure

```
DDRNet_w_KD/
├── models/
│   ├── ddrnet.py          # Student architecture
│   ├── dinov2_teacher.py  # Teacher wrapper
│   ├── kd_losses.py       # Distillation losses
│   └── kd_model.py        # Combined KD model
├── utils/
│   └── __init__.py        # Training utilities
├── configs/
│   ├── cityscapes_*.yaml  # Config files
│   └── ade20k_*.yaml
├── train.py               # Training script
├── inference.py           # Inference script
└── example.py             # Usage examples
```
