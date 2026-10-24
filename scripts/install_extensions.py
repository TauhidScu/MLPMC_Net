# -*- coding: utf-8 -*-
"""
MLPMC nnU-Net Extension Installer

This script integrates the MLPMC custom components (multi-task data loaders
and the nnUNetTrainercls_mlpmc_final trainer variant) into an installed
nnU-Net v2 environment.

Usage:
    python scripts/install_extensions.py
    python scripts/install_extensions.py --uninstall
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_nnunet_dir() -> Path:
    try:
        import nnunetv2
        return Path(nnunetv2.__path__[0])
    except ImportError:
        print("[ERROR] nnU-Net v2 is not installed in the current Python environment.")
        print("Please install official nnU-Net v2 first:")
        print("    pip install nnunetv2")
        print("or clone from https://github.com/MIC-DKFZ/nnUNet.git and install.")
        sys.exit(1)


def install_extensions():
    nnunet_dir = get_nnunet_dir()
    print("=" * 65)
    print("Installing MLPMC Extensions into nnU-Net v2")
    print(f"Target nnU-Net Location: {nnunet_dir}")
    print("=" * 65)

    files_to_install = [
        (
            REPO_ROOT / "custom_nnunet" / "variants" / "nnUNetTrainercls_mlpmc_final.py",
            nnunet_dir / "training" / "nnUNetTrainer" / "variants" / "nnUNetTrainercls_mlpmc_final.py"
        ),
        (
            REPO_ROOT / "custom_nnunet" / "dataloading" / "nnunet_dataset.py",
            nnunet_dir / "training" / "dataloading" / "nnunet_dataset.py"
        ),
        (
            REPO_ROOT / "custom_nnunet" / "dataloading" / "data_loader_2d.py",
            nnunet_dir / "training" / "dataloading" / "data_loader_2d.py"
        ),
    ]

    for src, dst in files_to_install:
        if not src.is_file():
            print(f"[ERROR] Source file not found: {src}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)

        # Create backup if destination exists and hasn't been backed up
        backup = dst.with_suffix(dst.suffix + ".backup_orig")
        if dst.is_file() and not backup.is_file():
            shutil.copy2(dst, backup)
            print(f"  [BACKUP] {dst.name} -> {backup.name}")

        if src.name == "nnUNetTrainercls_mlpmc_final.py":
            content = src.read_text(encoding='utf-8')
            repo_path_str = str(REPO_ROOT.resolve())
            content = content.replace(
                '_MLPMC_INSTALLED_ROOT = None',
                f'_MLPMC_INSTALLED_ROOT = r"{repo_path_str}"'
            )
            dst.write_text(content, encoding='utf-8')
        else:
            shutil.copy2(src, dst)

        print(f"  [INSTALLED] {src.name} -> {dst}")

    print("\n[SUCCESS] MLPMC extensions successfully installed!")
    print("You can now run MLPMC using the standard nnU-Net command:")
    print("    nnUNetv2_train <DATASET_ID> 2d <FOLD> -tr nnUNetTrainercls_mlpmc_final")
    print("=" * 65)


def uninstall_extensions():
    nnunet_dir = get_nnunet_dir()
    print("=" * 65)
    print("Restoring Original nnU-Net v2 Files")
    print("=" * 65)

    files_to_restore = [
        (
            nnunet_dir / "training" / "nnUNetTrainer" / "variants" / "nnUNetTrainercls_mlpmc_final.py",
            None
        ),
        (
            nnunet_dir / "training" / "dataloading" / "nnunet_dataset.py",
            nnunet_dir / "training" / "dataloading" / "nnunet_dataset.py.backup_orig"
        ),
        (
            nnunet_dir / "training" / "dataloading" / "data_loader_2d.py",
            nnunet_dir / "training" / "dataloading" / "data_loader_2d.py.backup_orig"
        ),
    ]

    for target, backup in files_to_restore:
        if backup and backup.is_file():
            shutil.copy2(backup, target)
            os.remove(backup)
            print(f"  [RESTORED] {target.name} from backup")
        elif not backup and target.is_file():
            os.remove(target)
            print(f"  [REMOVED] {target.name}")

    print("\n[SUCCESS] Uninstalled MLPMC extensions from nnU-Net v2.")
    print("=" * 65)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Install MLPMC custom extensions into nnU-Net v2")
    parser.add_argument('--uninstall', action='store_true', help="Uninstall extensions and restore backups")
    args = parser.parse_args()

    if args.uninstall:
        uninstall_extensions()
    else:
        install_extensions()
