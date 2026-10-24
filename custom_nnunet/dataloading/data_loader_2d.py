# -*- coding: utf-8 -*-
import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
try:
    from custom_nnunet.dataloading.nnunet_dataset import nnUNetDataset
except ImportError:
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset



class nnUNetDataLoader2D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg.
        # seg_all's channel count comes from self.seg_shape, which base_data_loader.determine_shapes()
        # derived by calling load_case once. So it is ALREADY sized to whatever the data provides:
        #   multi-task data  -> 2 seg channels [seg, cls]
        #   seg-only data    -> 1 seg channel  [seg]
        # We just have to make the cls read/write conditional to match.
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        # does this dataset carry a classification label? decided by the seg channel count,
        # not by any flag -> fully data-specific.
        has_cls = self.seg_shape[1] > 1
        n_cls = getattr(self, 'num_cls_labels', 4)
        cls_labels = np.full((self.batch_size, n_cls), -1, np.int16)   # ← ADD
        for j, current_key in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)
            data, seg, properties = self._data.load_case(current_key)
            case_properties.append(properties)

            if not force_fg:
                if self.has_ignore:
                    selected_class_or_region = self.annotated_classes_key
                else:
                    selected_class_or_region = None
            else:
                eligible_classes_or_regions = [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) > 0]
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])
                selected_class_or_region = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                    len(eligible_classes_or_regions) > 0 else None
            if selected_class_or_region is not None:
                selected_slice = np.random.choice(properties['class_locations'][selected_class_or_region][:, 1])
            else:
                selected_slice = np.random.choice(len(data))

            data = data[selected_slice]
            seg = seg[selected_slice]

            # ---- classification label lives at seg channel 1, ONLY if present ----
            # after slicing, seg is [C, H, W] so the channel axis is 0.
            if has_cls and seg.shape[0] > 1:
                cls_seg = seg[1, 0, :n_cls]
            else:
                cls_seg = None

            class_locations = {
                selected_class_or_region: properties['class_locations'][selected_class_or_region][properties['class_locations'][selected_class_or_region][:, 1] == selected_slice][:, (0, 2, 3)]
            } if (selected_class_or_region is not None) else None

            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg if selected_class_or_region is not None else None,
                                               class_locations, overwrite_class=selected_class_or_region)

            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            # data tensor is always 6 channels: [weak, strong1, strong2, reg, cutmix1, cutmix2]
            data_all[j][0] = np.pad(data[0][np.newaxis, ...], ((0, 0), *padding), 'constant', constant_values=0)
            data_all[j][1] = np.pad(data[1][np.newaxis, ...], ((0, 0), *padding), 'constant', constant_values=0)
            data_all[j][2] = np.pad(data[2][np.newaxis, ...], ((0, 0), *padding), 'constant', constant_values=0)
            data_all[j][3] = np.pad(data[3][np.newaxis, ...], ((0, 0), *padding), 'constant', constant_values=0)
            data_all[j][4] = np.pad(data[4][np.newaxis, ...], ((0, 0), *padding), 'constant', constant_values=0)
            data_all[j][5] = np.pad(data[5][np.newaxis, ...], ((0, 0), *padding), 'constant', constant_values=0)

            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)
            # write the per-case classification label back ONLY when it exists.
            if cls_seg is not None:
                cls_labels[j] = cls_seg                                # ← ADD
                seg_all[j, 1, 0, :n_cls] = cls_seg
        # print('[loader PRE-aug ]', seg_all[0, 1, 0, :24].astype(int).tolist(), flush=True)
        return {'data': data_all, 'seg': seg_all, 'cls': cls_labels, 'properties': case_properties, 'keys': selected_keys}
class nnUNetDataLoader2DSequential(nnUNetDataLoader2D):
    """Validation loader: every case exactly once per pass, then wraps.

    nnUNet samples WITH REPLACEMENT, so a 444-case split drawn 448 times yields
    only ~281 distinct cases -- 37% never seen, and a different 37% each epoch.
    Fine when metrics are averaged per batch; NOT fine for pooled mAP/AUC, where
    a repeated case counts twice in the global ranking and an unseen one not at
    all.

    Each worker owns a disjoint stride of the case list (thread_id::n_threads),
    so the union across workers is the full split with no overlap.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ptr = 0
        self._order = list(self.indices)

    def get_indices(self):
        n_thr = getattr(self, 'number_of_threads_in_multithreaded', None) or 1
        tid = getattr(self, 'thread_id', 0) or 0
        mine = self._order[tid::n_thr] or self._order
        out = []
        for _ in range(self.batch_size):
            if self._ptr >= len(mine):
                self._ptr = 0
            out.append(mine[self._ptr])
            self._ptr += 1
        return out

if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset004_Hippocampus/2d'
    ds = nnUNetDataset(folder, None, 1000)
    dl = nnUNetDataLoader2D(ds, 366, (65, 65), (56, 40), 0.33, None, None)
    a = next(dl)