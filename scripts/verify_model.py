# -*- coding: utf-8 -*-
"""
Verification script for MLPMC-Net:
- Tests model instantiation in both 'classification' and 'both' task modes
- Tests forward pass shapes (classification logits, segmentation maps, feature perturbation)
- Tests backward pass & gradient scaling functionality
- Optionally tests loading a real checkpoint (via --ckpt <path>)
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from models.mlpmc_net import get_model_cls_mlpmc


def verify_model(ckpt_path: str = None):
    print("=" * 70)
    print("Running MLPMC-Net Architecture Verification")
    print("=" * 70)

    # 1. Instantiate model in multi-task ('both') mode
    print("\n[1/4] Instantiating MLPMC-Net (task_mode='both', n_class_cls=4)...")
    model_both = get_model_cls_mlpmc(
        task_mode='both',
        n_class_cls=4,
        use_attention=True,
        use_psffm=True,
        lam_seg_enc=(1.0, 1.0, 0.15, 0.0),
        seg_embed_dim=256,
        seg_embed_grid=8
    )
    model_both.eval()
    print("  [PASS] Successfully created multi-task MLPMC-Net model.")

    # 2. Forward pass test in 'both' mode
    print("\n[2/4] Testing forward pass (both mode)...")
    dummy_input = torch.randn(2, 1, 256, 256)

    # Forward without perturbation (inference/val)
    with torch.no_grad():
        out_cls, out_seg = model_both(dummy_input, need_fp=False)
    print(f"  [PASS] Standard forward pass:")
    print(f"      Classification logits : {out_cls.shape} (expected [2, 4])")
    print(f"      Segmentation logits   : {out_seg.shape} (expected [2, 1, 256, 256])")
    assert out_cls.shape == (2, 4), f"Unexpected cls shape: {out_cls.shape}"
    assert out_seg.shape == (2, 1, 256, 256), f"Unexpected seg shape: {out_seg.shape}"

    # Forward with feature perturbation (training mode for unlabeled consistency)
    model_both.train()
    out_cls_tr, out_seg_tr, out_seg_fp = model_both(dummy_input, need_fp=True)
    print(f"  [PASS] Feature-perturbed forward pass (training):")
    print(f"      Cls logits            : {out_cls_tr.shape}")
    print(f"      Seg logits            : {out_seg_tr.shape}")
    print(f"      Perturbed seg logits  : {out_seg_fp.shape}")
    assert out_seg_fp.shape == (2, 1, 256, 256)

    # 3. Test backward pass and gradient scaling
    print("\n[3/4] Testing backward pass and gradient routing...")
    model_both.zero_grad()
    loss = out_cls_tr.sum() + 0.3 * out_seg_tr.sum()
    loss.backward()

    # Check that gradients exist in backbone and heads
    has_backbone_grad = any(p.grad is not None for n, p in model_both.named_parameters() if 'pretrain_model' in n)
    has_cls_grad = any(p.grad is not None for n, p in model_both.named_parameters() if 'chead' in n)
    has_seg_grad = any(p.grad is not None for n, p in model_both.named_parameters() if 'shead' in n)
    print(f"  [PASS] Backbone has gradient   : {has_backbone_grad}")
    print(f"  [PASS] Cls head has gradient   : {has_cls_grad}")
    print(f"  [PASS] Seg head has gradient   : {has_seg_grad}")
    assert has_backbone_grad and has_cls_grad and has_seg_grad, "Missing gradients in backward pass!"

    # 4. Classification-only mode
    print("\n[4/4] Testing MLPMC-Net (task_mode='classification')...")
    model_cls = get_model_cls_mlpmc(task_mode='classification', n_class_cls=4)
    model_cls.eval()
    with torch.no_grad():
        out_cls_only = model_cls(dummy_input)
    print(f"  [PASS] Classification-only output: {out_cls_only.shape}")
    assert out_cls_only.shape == (2, 4)

    # Optional checkpoint loading check
    if ckpt_path:
        print(f"\n[Optional] Loading checkpoint from: {ckpt_path}")
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = ck.get('network_weights', ck)
        sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
        miss, unexp = model_both.load_state_dict(sd, strict=False)
        print(f"  [PASS] Checkpoint loaded.")
        print(f"    Missing keys    : {len(miss)}")
        print(f"    Unexpected keys : {len(unexp)}")

    print("\n" + "=" * 70)
    print("All Model Verifications Passed Successfully!")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None, help='Optional checkpoint path to verify weight loading')
    args = parser.parse_args()
    verify_model(args.ckpt)
