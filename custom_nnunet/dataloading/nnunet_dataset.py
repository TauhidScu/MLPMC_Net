# -*- coding: utf-8 -*-
"""
nnUNetDataset with support for multi-task segmentation and classification targets,
and self-contained weak/strong data augmentation views.
"""

import os
import sys
import random
import shutil
from copy import deepcopy
from typing import List

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile
from nnunetv2.training.dataloading.utils import get_case_identifiers


def blur(img: Image.Image, p: float = 0.5) -> Image.Image:
    """Applies random Gaussian blur to an image with probability p."""
    if random.random() < p:
        sigma = random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box_ref(h: int, w: int, p: float = 0.5, size_min: float = 0.02, size_max: float = 0.4) -> np.ndarray:
    """Generates a CutMix binary mask for data augmentation."""
    mask = np.zeros((h, w), dtype=np.float32)
    if random.random() > p:
        return mask
    size = np.random.uniform(size_min, size_max) * h * w
    ratio = np.random.uniform(0.3, 3.33)
    cut_w = int(np.clip(np.sqrt(size / ratio), 1, w))
    cut_h = int(np.clip(cut_w * ratio, 1, h))
    mid_x = np.random.randint(0, w)
    mid_y = np.random.randint(0, h)
    x1 = max(0, mid_x - cut_w // 2)
    y1 = max(0, mid_y - cut_h // 2)
    x2 = min(w, mid_x + cut_w // 2)
    y2 = min(h, mid_y + cut_h // 2)
    mask[y1:y2, x1:x2] = 1.0
    return mask


def normalize(img: Image.Image) -> np.ndarray:
    """Converts a PIL image to a normalized float tensor array."""
    t = transforms.ToTensor()(img)
    t = transforms.Normalize([0.5], [0.5])(t)
    return t.numpy()


class nnUNetDataset(object):
    """
    Dataset loader for nnU-Net preprocessed data. Supports multi-task segmentation
    and classification targets.
    """
    def __init__(self, folder: str, case_identifiers: List[str] = None,
                 num_images_properties_loading_threshold: int = 0,
                 folder_with_segs_from_previous_stage: str = None,
                 folder_with_regs_from_previous_stage: str = None):
        super().__init__()
        if case_identifiers is None:
            case_identifiers = get_case_identifiers(folder)
        case_identifiers.sort()

        self.dataset = {}
        for c in case_identifiers:
            self.dataset[c] = {}
            self.dataset[c]['data_file'] = join(folder, f"{c}.npz")
            self.dataset[c]['properties_file'] = join(folder, f"{c}.pkl")
            if folder_with_segs_from_previous_stage is not None:
                self.dataset[c]['seg_from_prev_stage_file'] = join(folder_with_segs_from_previous_stage, f"{c}.npz")
            if folder_with_regs_from_previous_stage is not None:
                self.dataset[c]['reg_from_prev_stage_file'] = join(folder_with_regs_from_previous_stage, f"{c}.npz")

        if len(case_identifiers) <= num_images_properties_loading_threshold:
            for i in self.dataset.keys():
                self.dataset[i]['properties'] = load_pickle(self.dataset[i]['properties_file'])

        self.keep_files_open = ('nnUNet_keep_files_open' in os.environ.keys()) and \
                               (os.environ['nnUNet_keep_files_open'].lower() in ('true', '1', 't'))

    def __getitem__(self, key):
        ret = {**self.dataset[key]}
        if 'properties' not in ret.keys():
            ret['properties'] = load_pickle(ret['properties_file'])
        return ret

    def __setitem__(self, key, value):
        return self.dataset.__setitem__(key, value)

    def keys(self):
        return self.dataset.keys()

    def __len__(self):
        return self.dataset.__len__()

    def items(self):
        return self.dataset.items()

    def values(self):
        return self.dataset.values()

    def load_case(self, key):
        entry = self[key]

        # ---- load data ----
        if 'open_data_file' in entry.keys():
            data = entry['open_data_file']
        elif isfile(entry['data_file'][:-4] + ".npy"):
            data = np.load(entry['data_file'][:-4] + ".npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_data_file'] = data
        else:
            data = np.load(entry['data_file'])['data']

        # ---- load seg / reg / (optional) classification label ----
        l_cls = None
        if 'open_seg_file' in entry.keys():
            seg = entry['open_seg_file']
            reg = entry['open_reg_file']
            l_cls = entry.get('open_cls_file', None)
        elif isfile(entry['data_file'][:-4] + "_seg.npy"):
            seg = np.load(entry['data_file'][:-4] + "_seg.npy", 'r')
            reg = seg[:, 1, :, :][:, np.newaxis, :, :]
            if seg.shape[1] > 2:
                l_cls = seg[:, 2, :, :][:, np.newaxis, :, :]
            seg = seg[:, 0, :, :][:, np.newaxis, :, :]
            if self.keep_files_open:
                self.dataset[key]['open_seg_file'] = seg
                self.dataset[key]['open_reg_file'] = reg
                self.dataset[key]['open_cls_file'] = l_cls
        else:
            _s = np.load(entry['data_file'])['seg']
            seg = _s[:, 0, :, :][:, np.newaxis, :, :]
            reg = _s[:, 1, :, :][:, np.newaxis, :, :]
            if _s.shape[1] > 2:
                l_cls = _s[:, 2, :, :][:, np.newaxis, :, :]

        if 'seg_from_prev_stage_file' in entry.keys():
            if isfile(entry['seg_from_prev_stage_file'][:-4] + ".npy"):
                seg_prev = np.load(entry['seg_from_prev_stage_file'][:-4] + ".npy", 'r')
            else:
                seg_prev = np.load(entry['seg_from_prev_stage_file'])['seg']
            seg = np.vstack((seg, seg_prev[None]))

        # ---- weak / strong augmented views ----
        data_w = data.copy()
        datap = data.copy()[0][0]
        data_s1 = deepcopy(Image.fromarray(datap).convert('RGB'))
        data_s2 = deepcopy(Image.fromarray(datap).convert('RGB'))

        if random.random() < 0.8:
            data_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(data_s1)
        data_s1 = transforms.RandomGrayscale(p=0.2)(data_s1)
        data_s1 = blur(data_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box_ref(data_s1.size[1], data_s1.size[0], p=0.5)

        if random.random() < 0.8:
            data_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(data_s2)
        data_s2 = transforms.RandomGrayscale(p=0.2)(data_s2)
        data_s2 = blur(data_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box_ref(data_s2.size[1], data_s1.size[0], p=0.5)

        data_s1 = normalize(data_s1.convert("L"))
        data_s2 = normalize(data_s2.convert("L"))

        # ---- assemble seg: attach cls channel if present ----
        seg = np.array(seg)
        if l_cls is not None:
            seg = np.concatenate((seg, l_cls), axis=1)

        data_s1 = np.array(data_s1)
        data_s2 = np.array(data_s2)
        data_s1 = data_s1[np.newaxis, ...]
        data_s2 = data_s2[np.newaxis, ...]
        cutmix_box1 = cutmix_box1[np.newaxis, np.newaxis, ...]
        cutmix_box2 = cutmix_box2[np.newaxis, np.newaxis, ...]
        data = np.concatenate((data_w, data_s1, data_s2, reg, cutmix_box1, cutmix_box2), axis=1)
        return data, seg, entry['properties']