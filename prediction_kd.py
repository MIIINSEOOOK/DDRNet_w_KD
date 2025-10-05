
import os
import argparse
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from DDRNet import DDRNet
from torch.utils.data import Dataset, DataLoader
import matplotlib.cm as cm

class TestSegmentationDataset(Dataset):
    def __init__(self, root_dir, subset='test'):
        self.image_dir = os.path.join(root_dir, "image", subset)
        self.image_paths = sorted(glob(os.path.join(self.image_dir, "*", "*.*"), recursive=True))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        tensor = self.to_tensor(img)
        return tensor, img_path

def load_student(weight_path, num_classes, device):
    model = DDRNet(num_classes=num_classes)
    model = torch.nn.DataParallel(model)

    ckpt = torch.load(weight_path, map_location=device)
    # If KD checkpoint
    if isinstance(ckpt, dict) and "student" in ckpt:
        state_dict = ckpt["student"]
    else:
        state_dict = ckpt

    model_keys = model.state_dict().keys()
    state_keys = state_dict.keys()

    has_module_prefix = any(k.startswith("module.") for k in state_keys)
    needs_prefix = any(k.startswith("module.") for k in model_keys)

    if has_module_prefix != needs_prefix:
        if needs_prefix:
            state_dict = {"module." + k: v for k, v in state_dict.items()}
        else:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model

def save_prediction(pred, save_path, colormap_root, result_dir):
    pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(pred_np).save(save_path)

    # also save a color preview
    normed = pred_np.astype(np.float32) / 20.0
    cmap = cm.get_cmap('turbo')
    colored = cmap(normed)
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    rgb_img = Image.fromarray(rgb)

    rel_path = os.path.relpath(save_path, start=os.path.join(result_dir, "label"))
    cmap_path = os.path.join(colormap_root, rel_path)
    os.makedirs(os.path.dirname(cmap_path), exist_ok=True)
    rgb_img.save(cmap_path)

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TestSegmentationDataset(args.dataset_dir, subset='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = load_student(args.weight_path, args.num_classes, device)

    colormap_root = os.path.join(args.result_dir, "colormap")

    for img_tensor, img_path in tqdm(dataloader, desc="Predicting..."):
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            output = model(img_tensor)
            if isinstance(output, tuple):
                output = output[0]
            pred = torch.argmax(F.softmax(output, dim=1), dim=1)

        # dataset_dir/image/... → result_dir/label/...
        rel_path = os.path.relpath(img_path[0], os.path.join(args.dataset_dir, "image"))
        save_path = os.path.join(args.result_dir, "label", rel_path)

        # rename file to *_leftImg8bit.png to be compatible with evaluation.py
        dirname, basename = os.path.split(save_path)
        if not basename.endswith("_leftImg8bit.png"):
            base, ext = os.path.splitext(basename)
            basename = base + "_leftImg8bit.png"
            save_path = os.path.join(dirname, basename)

        save_prediction(pred, save_path, colormap_root, args.result_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to test images root")
    parser.add_argument("--weight_path", type=str, required=True, help="Path to KD checkpoint or vanilla DDRNet state dict")
    parser.add_argument("--result_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--num_classes", type=int, default=19)
    args = parser.parse_args()
    test(args)
