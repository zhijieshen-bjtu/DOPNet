"""
@Date: 2022/07/02
@description:
"""
import numpy as np
import torch

from utils.boundary import corners2boundary, visibility_corners, get_heat_map
from utils.conversion import xyz2depth, uv2xyz, uv2pixel
from dataset.communal.data_augmentation import PanoDataAugmentation


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, mode, shape=None, max_wall_num=999, aug=None, camera_height=1.6, patch_num=256, keys=None):
        if keys is None or len(keys) == 0:
            keys = ['image', 'id', 'corners', 'ratio', 'depth', 'segmentation']
        if shape is None:
            shape = [512, 1024]

        assert mode == 'train' or mode == 'val' or mode == 'test' or mode is None, 'unknown mode!'
        self.mode = mode
        self.keys = keys
        self.shape = shape
        self.pano_aug = None if aug is None or mode == 'val' else PanoDataAugmentation(aug)
        self.camera_height = camera_height
        self.max_wall_num = max_wall_num
        self.patch_num = patch_num
        self.data = None

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_depth(corners, plan_y=1, length=256, visible=True):
        visible_floor_boundary = corners2boundary(corners, length=length, visible=visible)
        visible_depth = xyz2depth(uv2xyz(visible_floor_boundary, plan_y), plan_y)
        return visible_depth

    def process_data(self, label, image, patch_num):
        corners = label['corners']

        if self.pano_aug is not None:
            corners, image = self.pano_aug.execute_aug(corners, image if 'image' in self.keys else None)
        eps = 1e-3
        corners[:, 1] = np.clip(corners[:, 1], 0.5+eps, 1-eps)

        image, aux_seg = image[:, :, :3], image[:, :, -1]

        output = {}
        if 'image' in self.keys:
            image = image.transpose(2, 0, 1)
            output['image'] = image

        visible_corners = visibility_corners(corners)

        if 'depth' in self.keys:
            depth = self.get_depth(visible_corners, length=patch_num, visible=False)
            output['depth'] = depth

        if 'ratio' in self.keys:
            output['ratio'] = label['ratio']

        if 'segmentation' in self.keys:
            output['segmentation'] = aux_seg

        if 'id' in self.keys:
            output['id'] = label['id']

        if 'corners' in self.keys:
            output['corners'] = np.zeros((32, 2), dtype=np.float32)
            output['corners'][:len(label['corners'])] = label['corners']

        return output
