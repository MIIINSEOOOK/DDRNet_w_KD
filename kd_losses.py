
import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_average(tensor, mask, eps: float = 1e-6):
    """
    tensor: (B, H, W) loss map or any map reduced along channel
    mask: (B, H, W) boolean or {0,1} where True/1 means keep
    """
    mask = mask.float()
    num = (mask * tensor).sum()
    den = mask.sum().clamp_min(eps)
    return num / den

class LogitsKDLoss(nn.Module):
    """
    Classic Hinton KD for segmentation.
    - teacher_logits: (B, C, Ht, Wt)
    - student_logits: (B, C, Hs, Ws)
    We resize teacher to student resolution before computing KL.
    Ignore-label regions in ground-truth are masked out.
    """
    def __init__(self, temperature: float = 4.0, ignore_index: int = 255):
        super().__init__()
        self.T = temperature
        self.ignore_index = ignore_index

    def forward(self, student_logits, teacher_logits, labels):
        # Resize teacher to student size
        if teacher_logits.shape[-2:] != student_logits.shape[-2:]:
            teacher_logits = F.interpolate(
                teacher_logits, size=student_logits.shape[-2:], mode='bilinear', align_corners=False
            )
        T = self.T
        log_p_s = F.log_softmax(student_logits / T, dim=1)
        p_t = F.softmax(teacher_logits / T, dim=1)

        # KL divergence per spatial location (sum over classes)
        kl_map = torch.sum(p_t * (torch.log(p_t.clamp_min(1e-12)) - log_p_s), dim=1)  # (B,H,W)

        # mask out ignore label
        mask = (labels != self.ignore_index)
        kd = masked_average(kl_map, mask)
        # multiply by T^2 as in Hinton KD
        return kd * (T * T)


class FeatureProjector(nn.Module):
    """
    1x1 conv to map student feature channels -> teacher embedding dim
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        return self.proj(x)


class CosineFeatureKDLoss(nn.Module):
    """
    Distill student features to teacher patch embeddings with cosine distance.
    - student_feat: B, Cs, Hs, Ws  (comes from DDRNet high-res branch via hook)
    - teacher_feat: B, Ct, Ht, Wt  (e.g., DINOv2 patch tokens as feature map)
    We'll project student to Ct first.
    """
    def __init__(self, projector: FeatureProjector, ignore_index: int = 255):
        super().__init__()
        self.projector = projector
        self.ignore_index = ignore_index

    def forward(self, student_feat, teacher_feat, labels):
        # Resize teacher spatially to student
        if teacher_feat.shape[-2:] != student_feat.shape[-2:]:
            teacher_feat = F.interpolate(
                teacher_feat, size=student_feat.shape[-2:], mode='bilinear', align_corners=False
            )

        s = self.projector(student_feat)
        # normalize along channel
        s = F.normalize(s, dim=1)
        t = F.normalize(teacher_feat, dim=1)

        # cosine distance = 1 - cosine_similarity
        cos_map = 1.0 - torch.sum(s * t, dim=1)  # (B,H,W)

        mask = (labels != self.ignore_index)
        return masked_average(cos_map, mask)
