"""
@date: 2021/6/25
@description:
"""
import os
import json
import numpy as np

from dataset.communal.read import read_image, read_label, read_seg
from dataset.communal.base_dataset import BaseDataset
from utils.logger import get_logger


class MP3DDataset(BaseDataset):
    def __init__(self, root_dir, mode, shape=None, max_wall_num=0, aug=None, camera_height=1.6, logger=None,
                 split_list=None, patch_num=256, keys=None, for_test_index=None, aux_segmentation=True):
        super().__init__(mode, shape, max_wall_num, aug, camera_height, patch_num, keys)

        if logger is None:
            logger = get_logger()
        self.root_dir = root_dir

        split_dir = os.path.join(root_dir, 'split')
        label_dir = os.path.join(root_dir, 'label')
        img_dir = os.path.join(root_dir, 'image')
        aux_seg_dir = os.path.join(root_dir, 'segmentation-mask')

        if split_list is None:
            with open(os.path.join(split_dir, f"{mode}.txt"), 'r') as f:
                split_list = [x.rstrip().split() for x in f]

        split_list.sort()
        if for_test_index is not None:
            split_list = split_list[:for_test_index]

        self.data = []
        invalid_num = 0
        for name in split_list:
            name = "_".join(name)
            img_path = os.path.join(img_dir, f"{name}.png")
            aux_seg_path = os.path.join(aux_seg_dir, f"{name}.png")
            label_path = os.path.join(label_dir, f"{name}.json")

            if not os.path.exists(img_path):
                logger.warning(f"{img_path} not exists")
                invalid_num += 1
                continue
            if not os.path.exists(label_path):
                logger.warning(f"{label_path} not exists")
                invalid_num += 1
                continue

            with open(label_path, 'r') as f:
                label = json.load(f)

                if self.max_wall_num >= 10:
                    if label['layoutWalls']['num'] < self.max_wall_num:
                        invalid_num += 1
                        continue
                elif self.max_wall_num != 0 and label['layoutWalls']['num'] != self.max_wall_num:
                    invalid_num += 1
                    continue

            self.aux_segmentation = aux_segmentation

            # print(label['layoutWalls']['num'])
            self.data.append([img_path, aux_seg_path, label_path])

        logger.info(
            f"Build dataset mode: {self.mode} max_wall_num: {self.max_wall_num} valid: {len(self.data)} invalid: {invalid_num}")

    def __getitem__(self, idx):
        rgb_path, aux_seg_path, label_path = self.data[idx]
        label = read_label(label_path, data_type='MP3D')
        image = read_image(rgb_path, self.shape)
        if self.aux_segmentation:
            aux_seg_image = read_seg(aux_seg_path, self.shape)
            image = np.concatenate([image, aux_seg_image], axis=2)

        output = self.process_data(label, image, self.patch_num)
        return output
