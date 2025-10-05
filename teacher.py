
import torch
import torch.nn as nn
import torch.nn.functional as F

DINOV2_EMBED_DIM = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}
DINOV2_PATCH = {
    "dinov2_vits14": 14,
    "dinov2_vitb14": 14,
    "dinov2_vitl14": 14,
    "dinov2_vitg14": 14,
}

def _to_bchw(x, h: int, w: int):
    # x: (B, N, C) -> (B, C, H, W)
    B, N, C = x.shape
    assert N == h * w, f"Token count {N} != H*W ({h}*{w})"
    x = x.transpose(1, 2).contiguous().view(B, C, h, w)
    return x

class DINOv2Teacher(nn.Module):
    """
    Wrap a DINOv2 backbone from torch.hub to return per-patch feature maps (B, C, H_patch, W_patch).
    """
    def __init__(self, arch: str = "dinov2_vits14", device: torch.device = torch.device("cpu")):
        super().__init__()
        try:
            self.backbone = torch.hub.load('facebookresearch/dinov2', arch)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINOv2 ({arch}) via torch.hub. "
                f"Make sure internet is available or the model is cached locally. Original error: {e}"
            )
        self.arch = arch
        self.patch = DINOV2_PATCH.get(arch, 14)
        self.embed_dim = DINOV2_EMBED_DIM.get(arch, None)
        self.eval().to(device)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns feature map (B, C, H', W'), where H' ~= H/patch, W' ~= W/patch.
        The code is defensive to support several possible APIs.
        """
        m = self.backbone

        # Try forward_features -> dict case (official dinov2 returns dict sometimes)
        feat_map = None
        if hasattr(m, "forward_features"):
            out = m.forward_features(x)
            # common keys observed in dinov2 code
            candidates = [
                "x_norm_patchtokens", "patch_tokens", "x_prenorm_patchtokens", "tokens"
            ]
            token = None
            if isinstance(out, dict):
                for k in candidates:
                    if k in out:
                        token = out[k]
                        break
            else:
                token = out

            if token is not None:
                # token can be (B,N,C) or (B, H', W', C) or (B, C, H', W')
                if token.dim() == 3:
                    B, N, C = token.shape
                    H = x.shape[-2] // self.patch
                    W = x.shape[-1] // self.patch
                    feat_map = _to_bchw(token, H, W)
                elif token.dim() == 4:
                    if token.shape[1] == x.shape[-2] // self.patch and token.shape[2] == x.shape[-1] // self.patch:
                        # (B, H, W, C)
                        feat_map = token.permute(0, 3, 1, 2).contiguous()
                    else:
                        # assume (B, C, H, W)
                        feat_map = token
        # Fallback: get_intermediate_layers
        if feat_map is None and hasattr(m, "get_intermediate_layers"):
            try:
                out = m.get_intermediate_layers(x, n=1, reshape=False)[0]  # (B, N, C)
                H = x.shape[-2] // self.patch
                W = x.shape[-1] // self.patch
                feat_map = _to_bchw(out, H, W)
            except Exception:
                pass

        if feat_map is None:
            # last fallback: assume forward returns BCHW features already
            y = m(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            if y.dim() == 3:
                H = x.shape[-2] // self.patch
                W = x.shape[-1] // self.patch
                feat_map = _to_bchw(y, H, W)
            elif y.dim() == 4:
                feat_map = y
            else:
                raise RuntimeError("Unsupported DINOv2 output format.")

        return feat_map
