
import torch
from typing import Optional

class FeatureHook:
    """
    Forward hook to capture a module's output feature map.
    """
    def __init__(self, module: torch.nn.Module):
        self.feat: Optional[torch.Tensor] = None
        self.handle = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inp, out):
        self.feat = out

    def close(self):
        self.handle.remove()
        self.handle = None

def attach_layer_hook(model, layer_path: str):
    """
    model may be wrapped by DDP, in which case real module is model.module
    layer_path: attribute path under the real module (e.g., 'layer5_')
    """
    real = model.module if hasattr(model, "module") else model
    target = real
    for name in layer_path.split("."):
        target = getattr(target, name)
    return FeatureHook(target)
