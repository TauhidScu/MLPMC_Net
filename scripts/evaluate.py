# -*- coding: utf-8 -*-
"""
Offline evaluation script for MLPMC-Net.
Computes multi-label classification metrics (Accuracy, Macro-F1, Precision, Recall)
and segmentation Dice score on the validation split.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile

from models.mlpmc_net import get_model_cls_mlpmc

LABELS = ['RS', 'MH', 'RD', 'CNV']


def evaluate_split(ckpt_path: str, preprocessed_dir: str, splits_file: str, fold: int = 0, device: str = 'cuda'):
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    print("=" * 70)
    print("Evaluating MLPMC-Net Model")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Dataset:    {preprocessed_dir}")
    print(f"  Fold:       {fold}")
    print(f"  Device:     {device}")
    print("=" * 70)

    # Load splits
    if not isfile(splits_file):
        raise FileNotFoundError(f"Splits file not found: {splits_file}")
    splits = load_json(splits_file)
    val_keys = splits[fold]['val']
    print(f"Found {len(val_keys)} validation cases.")

    # Initialize model
    model = get_model_cls_mlpmc(
        task_mode='both',
        n_class_cls=len(LABELS),
        use_attention=True,
        use_psffm=True,
        lam_seg_enc=(1.0, 1.0, 0.15, 0.0),
        seg_embed_dim=256,
        seg_embed_grid=8
    ).to(device)

    # Load weights
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck.get('network_weights', ck)
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    all_preds = []
    all_targets = []
    all_val_masks = []
    dice_scores = []

    print("\nRunning inference...")
    with torch.no_grad():
        for case in val_keys:
            npz_path = join(preprocessed_dir, f"{case}.npz")
            if not isfile(npz_path):
                continue

            z = np.load(npz_path)
            data = z['data']  # [C, H, W]
            seg_data = z['seg']  # [1, C_seg, H, W]

            # Prepare classification ground truth
            # [segmentation, reg, cls] -> cls is channel index 2 or 1 depending on prep
            if seg_data.shape[1] >= 3:
                target_cls_raw = seg_data[0, 2, 0, :len(LABELS)].astype(float)
            else:
                target_cls_raw = seg_data[0, 1, 0, :len(LABELS)].astype(float)

            valid_mask = (target_cls_raw >= 0) & (target_cls_raw <= 1)
            target_cls = np.clip(target_cls_raw, 0.0, 1.0)

            # Prepare image tensor
            img_tensor = torch.from_numpy(data[:1]).unsqueeze(0).float().to(device)

            # Predict
            pred_cls, pred_seg = model(img_tensor, need_fp=False)

            pred_bin = (torch.sigmoid(pred_cls).cpu().squeeze(0).numpy() > 0.5).astype(float)

            all_preds.append(pred_bin)
            all_targets.append(target_cls)
            all_val_masks.append(valid_mask)

            # Segmentation Dice evaluation if ground truth mask is present
            gt_seg = seg_data[0, 0]
            if gt_seg.sum() > 0 and (gt_seg.mean() < 0.9):  # non-dummy mask
                pred_seg_bin = (torch.sigmoid(pred_seg).cpu().squeeze().numpy() > 0.5).astype(float)
                inter = (pred_seg_bin * (gt_seg > 0)).sum()
                denom = pred_seg_bin.sum() + (gt_seg > 0).sum()
                if denom > 0:
                    d = 2.0 * inter / denom
                    dice_scores.append(d)

    preds = np.array(all_preds)
    targets = np.array(all_targets)
    masks = np.array(all_val_masks)

    # Metrics
    K = len(LABELS)
    correct_c = ((preds == targets) & masks).sum(axis=0)
    valid_c = masks.sum(axis=0)
    acc_per_class = 100.0 * correct_c / np.maximum(valid_c, 1)

    tp = ((preds == 1) & (targets == 1) & masks).sum(axis=0)
    fp = ((preds == 1) & (targets == 0) & masks).sum(axis=0)
    fn = ((preds == 0) & (targets == 1) & masks).sum(axis=0)

    pre_c = tp / np.maximum(tp + fp, 1e-8)
    rec_c = tp / np.maximum(tp + fn, 1e-8)
    f1_c = 2.0 * pre_c * rec_c / np.maximum(pre_c + rec_c, 1e-8)

    macro_f1 = float(np.mean(f1_c))
    mean_acc = float(np.mean(acc_per_class))
    mean_dice = float(np.mean(dice_scores)) if dice_scores else -1.0

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS (Validation Split)")
    print("=" * 65)
    print(f"{'Class':<8}{'Accuracy':>10}{'Precision':>12}{'Recall':>12}{'F1-Score':>12}{'Cases':>10}")
    print("-" * 65)
    for i in range(K):
        print(f"{LABELS[i]:<8}{acc_per_class[i]:>9.2f}%{pre_c[i]:>12.4f}{rec_c[i]:>12.4f}{f1_c[i]:>12.4f}{int(valid_c[i]):>10d}")
    print("-" * 65)
    print(f"{'Macro':<8}{mean_acc:>9.2f}%{np.mean(pre_c):>12.4f}{np.mean(rec_c):>12.4f}{macro_f1:>12.4f}{int(masks.sum()):>10d}")
    print("=" * 65)
    if mean_dice >= 0:
        print(f"Segmentation Mean Dice Score: {mean_dice:.4f} (across {len(dice_scores)} annotated slices)")
    print("=" * 65)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate MLPMC-Net Checkpoint")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to checkpoint (e.g. checkpoint_best_val.pth)")
    parser.add_argument('--preprocessed-dir', type=str, required=True, help="Path to preprocessed dataset folder")
    parser.add_argument('--splits', type=str, required=True, help="Path to splits_semi_chd.json")
    parser.add_argument('--fold', type=int, default=0, help="Fold index (default: 0)")
    parser.add_argument('--device', type=str, default='cuda', help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()

    evaluate_split(args.ckpt, args.preprocessed_dir, args.splits, args.fold, args.device)
