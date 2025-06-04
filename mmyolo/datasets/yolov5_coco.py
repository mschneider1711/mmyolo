# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional

from mmdet.datasets import BaseDetDataset, CocoDataset

from ..registry import DATASETS, TASK_UTILS


class BatchShapePolicyDataset(BaseDetDataset):
    """Dataset with the batch shape policy that makes paddings with least
    pixels during batch inference process, which does not require the image
    scales of all batches to be the same throughout validation."""

    def __init__(self,
                 *args,
                 batch_shapes_cfg: Optional[dict] = None,
                 **kwargs):
        self.batch_shapes_cfg = batch_shapes_cfg
        super().__init__(*args, **kwargs)

    def full_init(self):
        """rewrite full_init() to be compatible with serialize_data in
        BatchShapePolicy."""
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()

        # batch_shapes_cfg
        if self.batch_shapes_cfg:
            batch_shapes_policy = TASK_UTILS.build(self.batch_shapes_cfg)
            self.data_list = batch_shapes_policy(self.data_list)
            del batch_shapes_policy

        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()
        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True

    def prepare_data(self, idx: int) -> Any:
        if self.test_mode is False:
            data_info = self.get_data_info(idx)

            if 'img_path' not in data_info:
                from os import path as osp
                file_name = data_info.get('file_name') or data_info.get('img') or data_info.get('img_id')
                if file_name:
                    prefix = self.data_prefix['img'] if isinstance(self.data_prefix, dict) else self.data_prefix
                    data_info['img_path'] = osp.join(prefix, file_name)

            data_info['dataset'] = self

            # 💥 Albumentations erwartet 'img_path' in 'results' -> also hier direkt in das dict injizieren
            if 'img_path' not in data_info:
                data_info = dict(img_path=data_info['img_path'], **data_info)

            return self.pipeline(data_info)

        else:
            return super().prepare_data(idx)






@DATASETS.register_module()
class YOLOv5CocoDataset(BatchShapePolicyDataset, CocoDataset):
    """Dataset for YOLOv5 COCO Dataset.

    We only add `BatchShapePolicy` function compared with CocoDataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    pass
