# -*- coding: utf-8 -*-
"""
Training entrypoint for MLPMC-Net.
Enables standalone training without requiring modification of nnUNet site-packages.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="MLPMC-Net Standalone Training Launcher")
    parser.add_argument('dataset_name_or_id', type=str, help="Dataset ID or name (e.g. 702 or Dataset702_cls)")
    parser.add_argument('configuration', type=str, default='2d', nargs='?', help="Configuration (default: 2d)")
    parser.add_argument('fold', type=str, default='0', nargs='?', help="Fold index (0..4 or all)")
    parser.add_argument('--c', action='store_true', help="Continue training from checkpoint_latest.pth")
    parser.add_argument('--device', type=str, default='cuda', help="Compute device ('cuda' or 'cpu')")
    args = parser.parse_args()

    import torch
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    from training.trainer_mlpmc import nnUNetTrainercls_mlpmc_final

    fold = int(args.fold) if args.fold != 'all' else 'all'
    dataset_name = maybe_convert_to_dataset_name(args.dataset_name_or_id)
    prep_folder = os.path.join(nnUNet_preprocessed, dataset_name)

    plans_file = os.path.join(prep_folder, 'nnUNetPlans.json')
    dataset_json_file = os.path.join(prep_folder, 'dataset.json')

    if not os.path.isfile(plans_file):
        raise FileNotFoundError(
            f"Plans file not found at: {plans_file}\n"
            f"Ensure you have exported nnUNet_preprocessed and ran nnUNetv2_plan_and_preprocess."
        )
    if not os.path.isfile(dataset_json_file):
        raise FileNotFoundError(f"dataset.json not found at: {dataset_json_file}")

    with open(plans_file, 'r') as f:
        plans = json.load(f)
    with open(dataset_json_file, 'r') as f:
        dataset_json = json.load(f)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    trainer = nnUNetTrainercls_mlpmc_final(
        plans=plans,
        configuration=args.configuration,
        fold=fold,
        dataset_json=dataset_json,
        device=device
    )
    trainer.initialize()

    if args.c:
        latest_ckpt = os.path.join(trainer.output_folder, 'checkpoint_latest.pth')
        if os.path.isfile(latest_ckpt):
            print(f"Resuming training from: {latest_ckpt}")
            trainer.load_checkpoint(latest_ckpt)
        else:
            print(f"[Warning] No checkpoint found at {latest_ckpt}. Starting fresh training.")

    trainer.run_training()


if __name__ == '__main__':
    main()
