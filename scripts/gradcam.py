# -*- coding: utf-8 -*-
"""
Grad-CAM and Grad-CAM++ visualization for MLPMC-Net.
Generates class activation maps for individual OCT B-scans or batch evaluation.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.mlpmc_net import get_model_cls_mlpmc

LABELS = ['RS', 'MH', 'RD', 'CNV']


class GradCAM:
    """Computes Grad-CAM overlays for MLPMC-Net classification head."""
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        out_cls = self.model(input_tensor, need_fp=False)
        if isinstance(out_cls, (tuple, list)):
            out_cls = out_cls[0]

        target_score = out_cls[0, class_idx]
        target_score.backward(retain_graph=True)

        # Global average pool of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def apply_heatmap(img_np: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlays CAM heatmap onto grayscale image [H, W] in [0, 255]."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    if img_np.ndim == 2:
        img_rgb = np.stack([img_np] * 3, axis=-1)
    else:
        img_rgb = img_np

    overlay = np.uint8(alpha * heatmap + (1.0 - alpha) * img_rgb)
    return overlay


def run_gradcam(ckpt_path: str, npz_path: str, out_dir: str, target_class: str = 'all'):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model_cls_mlpmc(task_mode='both', n_class_cls=len(LABELS)).to(device)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck.get('network_weights', ck)
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Target layer: PSFFM or stage 4
    target_layer = model.chead.c_conv4
    cam_extractor = GradCAM(model, target_layer)

    z = np.load(npz_path)
    img = z['data'][0]  # [H, W]
    img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

    inp_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)

    classes_to_run = range(len(LABELS)) if target_class.lower() == 'all' else [LABELS.index(target_class.upper())]
    base_name = Path(npz_path).stem

    for c_idx in classes_to_run:
        c_name = LABELS[c_idx]
        cam = cam_extractor.generate(inp_tensor, c_idx)
        overlay = apply_heatmap(img_norm, cam)

        out_file = os.path.join(out_dir, f"{base_name}_cam_{c_name}.png")
        cv2.imwrite(out_file, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"  [DONE] Saved Grad-CAM for {c_name} to: {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for MLPMC-Net")
    parser.add_argument('--ckpt', type=str, required=True, help="Checkpoint path")
    parser.add_argument('--input', type=str, required=True, help="Path to input .npz file")
    parser.add_argument('--out-dir', type=str, default='./cam_outputs', help="Output directory")
    parser.add_argument('--target-class', type=str, default='all', choices=['all', 'RS', 'MH', 'RD', 'CNV'])
    args = parser.parse_args()

    run_gradcam(args.ckpt, args.input, args.out_dir, args.target_class)
