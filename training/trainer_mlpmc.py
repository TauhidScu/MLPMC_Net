# -*- coding: utf-8 -*-
"""
Standalone Trainer for Multi-Label Pathology Multi-Task Classification + Segmentation (MLPMC).
Class: nnUNetTrainercls_mlpmc_final
"""

import os
import sys
import functools
from time import time
from typing import Union, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch import distributed as dist
from sklearn.metrics import precision_score, recall_score, f1_score
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json

# nnU-Net framework imports
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
try:
    from custom_nnunet.dataloading.nnunet_dataset import nnUNetDataset
    from custom_nnunet.dataloading.data_loader_2d import nnUNetDataLoader2D
except ImportError:
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
    from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.training.dataloading.utils import get_case_identifiers

# MLPMC internal components
from models.mlpmc_net import get_model_cls_mlpmc
from training.losses import focal_loss, DC_and_BCE_loss, consistency_weight

__all__ = ['nnUNetTrainercls_mlpmc_final']


class nnUNetTrainercls_mlpmc_final(nnUNetTrainer):
    """
    Multi-label pathology multi-task trainer for MLPMC-Net.

    Task mode is chosen dynamically from the splits data:
        train_u empty      -> task_mode = 'classification'
        train_u non-empty  -> task_mode = 'both'

    Model Contract:
        task_mode='classification'          -> logits                      [tensor]
        task_mode='both', need_fp=False     -> (cls, seg)                  [2-tuple]
        task_mode='both', need_fp=True      -> (cls, seg, seg_fp)          [3-tuple]
    """

    # Label names (single source of truth for 4-class multi-label OCT task)
    CLS_NAMES = ('RS', 'MH', 'RD', 'CNV')

    VAL_IDX = {
        'loss': 0, 'acc_t': 1, 'pre_t': 2, 'recall_t': 3, 'f1_t': 4,
        'pre_0': 5,    'pre_1': 6,    'pre_2': 7,    'pre_3': 8,
        'recall_0': 9, 'recall_1': 10, 'recall_2': 11, 'recall_3': 12,
        'f1_0': 13,    'f1_1': 14,    'f1_2': 15,    'f1_3': 16,
        'dice': 17, 'loss_seg': 18,
        'acc_0': 19,   'acc_1': 20,   'acc_2': 21,   'acc_3': 22,
    }

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):

        self.batch_size_no_seg       = None
        self.batch_size_val          = None
        self.has_samples_without_seg = False
        self.num_samples_without_seg = 0
        self.task_mode               = None

        # Gradient probing state
        self._train_batch_counter = 0
        self._last_grad_probe     = None

        # Checkpoint trackers
        self._best_ema      = None   # best train acc
        self._best_acc      = None   # best val acc
        self._best_dice     = None   # best val dice
        self._best_seg_loss = None   # best val seg loss
        self._best_score    = None   # best val macro-F1

        # Model hyperparameters
        self.n_class_cls      = 4            # RS, MH, RD, CNV
        self.base_ratio       = 0.08
        self.reduction        = 16
        self.use_attention    = True
        self.use_psffm        = True
        self.lam_seg_enc      = (1.0, 1.0, 0.15, 0.0)
        self.seg_embed_dim    = 256
        self.seg_embed_grid   = 8

        self.use_lcsca        = False
        self.lcsca_dim        = 256
        self.lcsca_grid       = 16
        self.lcsca_detach_seg = True
        self.lcsca_heads      = 4

        # Loss weights: total = L_cls + lam_seg * L_seg + w_cons * L_cons
        self.lam_seg          = 0.3
        self.lam_cons         = 0.2
        self.cons_threshold   = 0.95
        self.cons_rampup      = 40      # epochs

        # Runtime
        self.detect_anomaly   = False
        self.grad_clip_norm   = 1.0

        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.print_to_log_file("=" * 70)
        self.print_to_log_file("Initializing nnUNetTrainercls_mlpmc_final")
        self.print_to_log_file("Model: MLPMC-Net — shared encoder + SegFeatureEmbed")
        self.print_to_log_file(f"  attention={self.use_attention}  psffm={self.use_psffm}  lcsca={self.use_lcsca}")
        self.print_to_log_file(f"  lam_seg_enc={self.lam_seg_enc}  lam_seg={self.lam_seg}  lam_cons={self.lam_cons}")
        self.print_to_log_file("=" * 70)

    def configure_optimizers(self):
        backbone_params, head_params = [], []
        module = self.network.module if hasattr(self.network, 'module') else self.network

        for name, param in module.named_parameters():
            if 'pretrain_model' in name or 'base' in name:
                backbone_params.append(param)   # pretrained backbone -> 0.1x lr
            else:
                head_params.append(param)       # new heads -> 1.0x lr

        optimizer = torch.optim.SGD(
            [
                {'params': backbone_params, 'lr': self.initial_lr * 0.1},
                {'params': head_params,     'lr': self.initial_lr},
            ],
            momentum=0.99,
            weight_decay=3e-5,
            nesterov=True
        )

        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        self.grad_scaler = GradScaler(
            init_scale=2. ** 14,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
            enabled=(self.device.type == 'cuda')
        )
        return optimizer, lr_scheduler

    def determine_task_mode(self):
        """Determine 'classification' vs 'both' based on train_u data availability."""
        if self.task_mode is not None:
            return

        tr_keys, val_keys, tr_no_seg_keys = self.do_split()
        self.num_samples_without_seg = len(tr_no_seg_keys)

        if self.num_samples_without_seg == 0:
            self.print_to_log_file("\n" + "=" * 70)
            self.print_to_log_file("TASK MODE: CLASSIFICATION ONLY (no train_u data found)")
            self.print_to_log_file("=" * 70)
            self.task_mode               = 'classification'
            self.has_samples_without_seg = False
            self.batch_size_no_seg       = self.batch_size
        else:
            if self.num_samples_without_seg < self.batch_size_no_seg:
                self.batch_size_no_seg = self.num_samples_without_seg

            self.print_to_log_file("\n" + "=" * 70)
            self.print_to_log_file("TASK MODE: BOTH (classification + segmentation)")
            self.print_to_log_file(f"Found {self.num_samples_without_seg} samples without segmentation in train_u")
            self.print_to_log_file("=" * 70)
            self.task_mode               = 'both'
            self.has_samples_without_seg = True

    def plot_network_architecture(self):
        self.print_to_log_file("Network architecture plotting disabled (BatchNorm needs batch_size > 1).")

    def _set_batch_size_and_oversample(self):
        if not self.is_ddp:
            self.batch_size        = 4    # samples WITH segmentation
            self.batch_size_no_seg = 16   # samples WITHOUT segmentation
            self.batch_size_val    = 8
        else:
            super()._set_batch_size_and_oversample()
            self.batch_size_no_seg = self.batch_size
            self.batch_size_val    = self.batch_size

    def do_split(self):
        """Loads splits from splits_semi_chd.json or fallback splits."""
        if self.fold == "all":
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            return case_identifiers, case_identifiers, []

        splits_file = join(self.preprocessed_dataset_folder_base, "splits_semi_chd.json")
        if not isfile(splits_file):
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
        if not isfile(splits_file):
            raise FileNotFoundError(f"Required splits file not found in: {self.preprocessed_dataset_folder_base}")

        splits = load_json(splits_file)
        if self.fold >= len(splits):
            raise ValueError(f"Invalid fold: {self.fold}. Splits contains {len(splits)} folds.")

        tr_keys        = splits[self.fold]['train']
        tr_no_seg_keys = splits[self.fold].get('train_u', [])
        val_keys       = splits[self.fold]['val']

        return tr_keys, val_keys, tr_no_seg_keys

    def get_tr_and_val_datasets(self):
        tr_keys, val_keys, tr_no_seg_keys = self.do_split()
        self.num_samples_without_seg = len(tr_no_seg_keys)

        dataset_tr = nnUNetDataset(
            self.preprocessed_dataset_folder, tr_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0
        )
        if self.num_samples_without_seg > 0:
            dataset_tr_no_seg = nnUNetDataset(
                self.preprocessed_dataset_folder, tr_no_seg_keys,
                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                num_images_properties_loading_threshold=0
            )
        else:
            dataset_tr_no_seg = dataset_tr

        dataset_val = nnUNetDataset(
            self.preprocessed_dataset_folder, val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0
        )
        return dataset_tr, dataset_val, dataset_tr_no_seg

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val, dataset_tr_no_seg = self.get_tr_and_val_datasets()
        DL = nnUNetDataLoader2D if dim == 2 else nnUNetDataLoader3D

        dl_tr = DL(
            dataset_tr, self.batch_size, initial_patch_size,
            self.configuration_manager.patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None
        )

        if self.task_mode == 'both':
            dl_tr_no_seg = DL(
                dataset_tr_no_seg, self.batch_size_no_seg, initial_patch_size,
                self.configuration_manager.patch_size, self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None
            )
        else:
            dl_tr_no_seg = dl_tr

        dl_val = DL(
            dataset_val, self.batch_size_val,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None
        )
        return dl_tr, dl_tr_no_seg, dl_val

    def on_train_start(self, **kwargs):
        super().on_train_start(**kwargs)
        if self.local_rank == 0:
            self.print_to_log_file("=" * 60)
            self.print_to_log_file("BATCH SHAPE VERIFICATION")
            self.print_to_log_file("=" * 60)
            batch = next(self.dataloader_train)
            batch_val = next(self.dataloader_val)
            self.print_to_log_file(f"  task mode        : {self.task_mode}")
            self.print_to_log_file(f"  labeled batch    : {batch['data'].shape}")
            self.print_to_log_file(f"  validation batch : {batch_val['data'].shape}")
            if self.task_mode == 'both':
                batch_no_seg = next(self.dataloader_train_u)
                self.print_to_log_file(f"  unlabeled batch  : {batch_no_seg['data'].shape}")
            self.print_to_log_file("=" * 60)

    def _consistency_weight(self) -> float:
        return consistency_weight(self.current_epoch, self.cons_rampup, self.lam_cons)

    def train_step(self, batch: dict, batch_no_seg: dict) -> dict:
        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)

        # BRANCH A: classification only
        if self.task_mode == 'classification':
            data = torch.unsqueeze(batch['data'][:, 0, :, :], dim=1).to(self.device, non_blocking=True)
            target_cls_raw = batch['target'][:, 1, :, :].to(torch.int).float()[:, 0, :self.n_class_cls].to(self.device, non_blocking=True)
            valid_mask = (target_cls_raw >= 0) & (target_cls_raw <= 1)
            target_cls = target_cls_raw.clamp(0., 1.)

            with (torch.amp.autocast(device_type='cuda', enabled=True) if self.device.type == 'cuda' else dummy_context()):
                outputs = self.network(data, False, False)
                pred_cls = outputs
                total_loss = focal_loss(pred_cls, target_cls, valid_mask=valid_mask)

                with torch.no_grad():
                    pred_bin = (torch.sigmoid(pred_cls) > 0.5).float()
                    n_valid  = valid_mask.sum().item()
                    accuracy = (100. * float(((pred_bin == target_cls) & valid_mask).sum().item()) / n_valid) if n_valid > 0 else 0.0

            self._backward_and_step(total_loss)
            return {'loss': total_loss.detach().cpu().item(), 'acc': accuracy, 'dice': -1.0}

        # BRANCH B: both (supervised seg + semi-supervised consistency)
        data_with_seg = torch.unsqueeze(batch['data'][:, 0, :, :], dim=1).to(self.device, non_blocking=True)
        target_seg    = torch.unsqueeze(batch['target'][:, 0, :, :], dim=1).to(self.device, non_blocking=True)
        target_cls_ws_raw = batch['target'][:, 1, :, :].to(torch.int).float()[:, 0, :self.n_class_cls].to(self.device, non_blocking=True)
        valid_mask_ws = (target_cls_ws_raw >= 0) & (target_cls_ws_raw <= 1)
        target_cls_ws = target_cls_ws_raw.clamp(0., 1.)

        num_with_seg = data_with_seg.shape[0]

        data_no_seg = torch.unsqueeze(batch_no_seg['data'][:, 0, :, :], dim=1).to(self.device, non_blocking=True)
        target_cls_ns_raw = batch_no_seg['target'][:, 1, :, :].to(torch.int).float()[:, 0, :self.n_class_cls].to(self.device, non_blocking=True)
        valid_mask_ns = (target_cls_ns_raw >= 0) & (target_cls_ns_raw <= 1)
        target_cls_ns = target_cls_ns_raw.clamp(0., 1.)

        combined_data       = torch.cat((data_with_seg, data_no_seg))
        combined_cls_labels = torch.cat((target_cls_ws, target_cls_ns))
        combined_valid_mask = torch.cat((valid_mask_ws, valid_mask_ns))

        with (torch.amp.autocast(device_type='cuda', enabled=True) if self.device.type == 'cuda' else dummy_context()):
            pred_cls, pred_seg, pred_seg_perturbed = self.network(combined_data, False, True)

            pred_seg_with_gt         = pred_seg[:num_with_seg]
            pred_seg_no_gt           = pred_seg[num_with_seg:]
            pred_seg_no_gt_perturbed = pred_seg_perturbed[num_with_seg:]

            # Loss 1: Classification over all samples
            loss_cls = focal_loss(pred_cls, combined_cls_labels, valid_mask=combined_valid_mask)

            # Loss 2: Supervised Segmentation
            loss_seg_sup = self.loss(pred_seg_with_gt, target_seg)
            loss_seg = loss_seg_sup[0]
            dice = loss_seg_sup[1]

            # Loss 3: Feature Perturbation Consistency on unlabeled samples
            with torch.no_grad():
                conf = torch.sigmoid(pred_seg_no_gt.detach())
                pseudo = (conf >= self.cons_threshold).float()
                cmask = pseudo

            l_cons = self.loss2(pred_seg_no_gt_perturbed, pseudo)[2]
            if l_cons.dim() == cmask.dim() - 1:
                cmask = cmask.squeeze(1)
            elif l_cons.dim() == cmask.dim() and l_cons.shape != cmask.shape:
                cmask = cmask.expand_as(l_cons)
            loss_consistency = (l_cons * cmask).mean()

            w_cons = self._consistency_weight()
            total_loss = loss_cls + self.lam_seg * loss_seg + w_cons * loss_consistency

            with torch.no_grad():
                pred_bin = (torch.sigmoid(pred_cls) > 0.5).float()
                n_valid  = combined_valid_mask.sum().item()
                accuracy = (100. * float(((pred_bin == combined_cls_labels) & combined_valid_mask).sum().item()) / n_valid) if n_valid > 0 else 0.0

        self._backward_and_step(total_loss)

        return {
            'loss': total_loss.detach().cpu().item(),
            'acc': accuracy,
            'dice': dice.detach().cpu().item(),
            'loss_cls': loss_cls.detach().cpu().item(),
            'loss_seg': loss_seg.detach().cpu().item(),
            'loss_cons': loss_consistency.detach().cpu().item(),
        }

    def _backward_and_step(self, total_loss: torch.Tensor):
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_norm)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)
        loss_here = float(np.mean(outputs['loss']))
        acc = float(np.mean(outputs['acc']))

        if self.task_mode == 'both':
            dice = float(np.mean(outputs['dice']))
            self.logger.log('train_losses', [loss_here, acc, dice], self.current_epoch)
            self.print_to_log_file(
                f"Epoch {self.current_epoch} train  cls={np.mean(outputs['loss_cls']):.4f}  "
                f"seg={np.mean(outputs['loss_seg']):.4f}  cons={np.mean(outputs['loss_cons']):.4f}"
            )
        else:
            self.logger.log('train_losses', [loss_here, acc, -1.0], self.current_epoch)

    def validation_step(self, batch: dict) -> dict:
        self.network.eval()
        data = torch.unsqueeze(batch['data'][:, 0, :, :], dim=1).to(self.device, non_blocking=True)

        target_cls_raw = batch['target'][:, 1, :, :].to(torch.int).float()[:, 0, :self.n_class_cls].to(self.device, non_blocking=True)
        valid_mask = (target_cls_raw >= 0) & (target_cls_raw <= 1)
        target_cls = target_cls_raw.clamp(0., 1.)

        target_seg_raw = batch['target'][:, 0, :, :]
        fill_ratio = float(target_seg_raw.float().mean().item())
        is_dummy_mask = fill_ratio > 0.9
        has_seg_mask = (target_seg_raw.sum().item() > 0) and (not is_dummy_mask)

        target_seg = None
        if has_seg_mask and self.task_mode == 'both':
            target_seg = torch.unsqueeze(target_seg_raw, dim=1).to(self.device, non_blocking=True)

        with torch.no_grad():
            with (torch.amp.autocast(device_type='cuda', enabled=True) if self.device.type == 'cuda' else dummy_context()):
                outputs = self.network(data, False, False)
                if self.task_mode == 'classification':
                    pred_cls = outputs
                    l = focal_loss(pred_cls, target_cls, valid_mask=valid_mask)
                    loss_seg_val = dice_val = -1.0
                else:
                    pred_cls, pred_seg = outputs
                    l = focal_loss(pred_cls, target_cls, valid_mask=valid_mask)
                    if has_seg_mask:
                        loss_all = self.loss(pred_seg, target_seg)
                        loss_seg_val = float(loss_all[0].item())
                        dice_val = float(loss_all[1].item())
                    else:
                        loss_seg_val = dice_val = -1.0

                pred_bin = (torch.sigmoid(pred_cls) > 0.5).float()
                n_valid = valid_mask.sum().item()
                acc_t = (100. * float(((pred_bin == target_cls) & valid_mask).sum().item()) / n_valid) if n_valid > 0 else 0.0

                correct_c = ((pred_bin == target_cls) & valid_mask).float().sum(0).cpu().numpy()
                valid_c   = valid_mask.float().sum(0).cpu().numpy()
                with np.errstate(invalid='ignore', divide='ignore'):
                    acc_per_class = np.where(valid_c > 0, 100. * correct_c / np.maximum(valid_c, 1), -1.0)

                tp_c = ((pred_bin == 1) & (target_cls == 1) & valid_mask).float().sum(0).cpu().numpy()
                fp_c = ((pred_bin == 1) & (target_cls == 0) & valid_mask).float().sum(0).cpu().numpy()
                fn_c = ((pred_bin == 0) & (target_cls == 1) & valid_mask).float().sum(0).cpu().numpy()

        label_true = target_cls.cpu().numpy()
        label_pred = pred_bin.cpu().numpy()
        pre_t    = float(precision_score(label_true, label_pred, zero_division=1, average='weighted'))
        recall_t = float(recall_score(label_true, label_pred, zero_division=1, average='weighted'))
        f1_t     = float(f1_score(label_true, label_pred, zero_division=1, average='weighted'))

        return {
            'loss': float(l.detach().cpu().item()),
            'loss_seg': loss_seg_val,
            'acc_t': acc_t,
            'pre_t': pre_t,
            'recall_t': recall_t,
            'f1_t': f1_t,
            'acc_correct': correct_c,
            'acc_valid': valid_c,
            'tp': tp_c, 'fp': fp_c, 'fn': fn_c,
            'dice': dice_val,
            'has_seg_mask': has_seg_mask,
        }

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs = collate_outputs(val_outputs)
        K = self.n_class_cls

        loss_here = float(np.mean(outputs['loss']))
        acc_t     = float(np.mean(outputs['acc_t']))
        pre_t     = float(np.mean(outputs['pre_t']))
        recall_t  = float(np.mean(outputs['recall_t']))
        f1_t      = float(np.mean(outputs['f1_t']))

        corr_sum  = np.sum(np.array(outputs['acc_correct']), axis=0)
        valid_sum = np.sum(np.array(outputs['acc_valid']), axis=0)
        acc_c = np.where(valid_sum > 0, 100. * corr_sum / np.maximum(valid_sum, 1), -1.0)

        tp = np.sum(np.array(outputs['tp']), axis=0)
        fp = np.sum(np.array(outputs['fp']), axis=0)
        fn = np.sum(np.array(outputs['fn']), axis=0)
        pooled_pre = tp / np.maximum(tp + fp, 1e-8)
        pooled_rec = tp / np.maximum(tp + fn, 1e-8)
        pooled_f1  = 2 * pooled_pre * pooled_rec / np.maximum(pooled_pre + pooled_rec, 1e-8)
        macro_f1   = float(np.mean(pooled_f1))

        dice_arr = np.array(outputs['dice'])
        valid_dice = dice_arr[dice_arr >= 0.0]
        dice_mean = float(np.mean(valid_dice)) if len(valid_dice) else -1.0

        loss_seg_arr = np.array(outputs['loss_seg'])
        valid_seg = loss_seg_arr[loss_seg_arr >= 0.0]
        loss_seg_mean = float(np.mean(valid_seg)) if len(valid_seg) else -1.0

        self.logger.log('val_losses', [
            loss_here, acc_t, pre_t, recall_t, f1_t,
            *[float(pooled_pre[i]) for i in range(K)],
            *[float(pooled_rec[i]) for i in range(K)],
            *[float(pooled_f1[i])  for i in range(K)],
            dice_mean, loss_seg_mean,
            *[float(acc_c[i])      for i in range(K)]
        ], self.current_epoch)

        self._last_macro_f1 = macro_f1
        self.print_to_log_file(f"Val Macro-F1: {macro_f1:.4f}  Val Acc: {acc_t:.2f}%  Val Dice: {dice_mean:.4f}")

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)
        v = self.logger.my_fantastic_logging['val_losses'][-1]
        I = self.VAL_IDX

        acc_val = v[I['acc_t']]
        dice_val = v[I['dice']]
        loss_seg_val = v[I['loss_seg']]
        macro_f1 = getattr(self, '_last_macro_f1', None)

        # Checkpoints
        if (self.current_epoch + 1) % self.save_every == 0 and self.current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        if self._best_acc is None or acc_val > self._best_acc:
            self._best_acc = acc_val
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best_val.pth'))

        if macro_f1 is not None:
            if self._best_score is None or macro_f1 > self._best_score:
                self._best_score = macro_f1
                self.save_checkpoint(join(self.output_folder, 'checkpoint_best_mlscore.pth'))

        if self.task_mode == 'both' and dice_val >= 0:
            if self._best_dice is None or dice_val > self._best_dice:
                self._best_dice = dice_val
                self.save_checkpoint(join(self.output_folder, 'checkpoint_best_dice.pth'))

        if self.task_mode == 'both' and loss_seg_val >= 0:
            if self._best_seg_loss is None or loss_seg_val < self._best_seg_loss:
                self._best_seg_loss = loss_seg_val
                self.save_checkpoint(join(self.output_folder, 'checkpoint_best_seg_loss.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def run_training(self, **kwargs):
        self.on_train_start(**kwargs)
        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            self._train_batch_counter = 0
            self.on_train_epoch_start()

            train_outputs = []
            for _ in range(self.num_iterations_per_epoch):
                batch = next(self.dataloader_train)
                batch_no_seg = next(self.dataloader_train_u) if self.task_mode == 'both' else batch
                train_outputs.append(self.train_step(batch, batch_no_seg))
                self._train_batch_counter += 1

            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = [
                    self.validation_step(next(self.dataloader_val))
                    for _ in range(self.num_val_iterations_per_epoch)
                ]
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
        self.on_train_end()

    def initialize(self, **kwargs):
        if self.was_initialized:
            raise RuntimeError("Trainer already initialized!")

        self.determine_task_mode()
        self.num_input_channels = determine_num_input_channels(
            self.plans_manager, self.configuration_manager, self.dataset_json
        )

        model_kwargs = dict(
            task_mode        = self.task_mode,
            n_class_cls      = self.n_class_cls,
            is_train         = True,
            base_ratio       = self.base_ratio,
            reduction        = self.reduction,
            use_attention    = self.use_attention,
            use_psffm        = self.use_psffm,
            lam_seg_enc      = self.lam_seg_enc,
            seg_embed_dim    = self.seg_embed_dim,
            seg_embed_grid   = self.seg_embed_grid,
            use_lcsca        = self.use_lcsca,
            lcsca_dim        = self.lcsca_dim,
            lcsca_grid       = self.lcsca_grid,
            lcsca_detach_seg = self.lcsca_detach_seg,
        )
        model_kwargs.update(kwargs)

        self.network = self.build_network_architecture(**model_kwargs).to(self.device)

        if not self.is_ddp:
            n_gpu = torch.cuda.device_count()
            if n_gpu > 1:
                primary = self.device.index if self.device.index is not None else 0
                device_ids = [primary] + [i for i in range(n_gpu) if i != primary]
                self.network = torch.nn.DataParallel(self.network, device_ids=device_ids, output_device=primary)
        else:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.network = DDP(self.network, device_ids=[self.local_rank])

        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        self.loss  = self._build_loss(reduction='mean')
        self.loss2 = self._build_loss(reduction='none')
        self.was_initialized = True

    def _build_loss(self, reduction='mean', sig=True):
        return DC_and_BCE_loss(
            reduction=reduction,
            sig=sig
        )

    @staticmethod
    def build_network_architecture(**kwargs) -> nn.Module:
        return get_model_cls_mlpmc(**kwargs)
