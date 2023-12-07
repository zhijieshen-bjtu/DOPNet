"""
@date: 2021/06/19
@description:
"""
# import sys
# sys.path.append('/data/cylin/zzs/Pan_layout/LGT-Net-4')

import matplotlib.pyplot as plt
import cv2
import numpy as np
from utils.conversion import uv2pixel
from utils.boundary import corners2boundary, corners2boundaries, find_peaks, connect_corners_uv, get_object_cor, \
    visibility_corners


def draw_boundary(pano_img, corners: np.ndarray = None, boundary: np.ndarray = None, draw_corners=True, show=False,
                  step=0.01, length=None, boundary_color=None, marker_color=None, title=None, visible=True):

    assert corners is not None or boundary is not None, "corners or boundary error"

    if (corners is not None and len(corners) > 2) or \
            (boundary is not None and len(boundary) > 2):
        if isinstance(boundary_color, list) or isinstance(boundary_color, np.array):
            if boundary is None:
                boundary = corners2boundary(corners, step, length, visible)

            boundary = uv2pixel(boundary, 1024, 512)

    return boundary


def draw_boundaries(pano_img, corners_list: list = None, boundary_list: list = None, draw_corners=True, show=True,
                    step=0.01, length=None, boundary_color=None, marker_color=None, title=None, ratio=None, visible=True):

    if boundary_color is None:
        boundary_color = [1, 0, 0]

    shape = sorted(pano_img.shape)
    assert len(shape) > 1, "pano_img shape error"
    w = shape[-1]
    h = shape[-2]

    pano_img = pano_img.copy()

    assert corners_list is not None or boundary_list is not None, "corners_list or boundary_list error"

    corners_list = corners2boundaries(ratio, corners_uv=corners_list[0], step=None, visible=visible)

    boundary1 = draw_boundary(pano_img, corners=corners_list[0], draw_corners=draw_corners,
                             step=step, length=length, boundary_color=boundary_color, marker_color=marker_color,
                             title=title, visible=visible)

    boundary2 = draw_boundary(pano_img, corners=corners_list[1], draw_corners=draw_corners,
                              step=step, length=length, boundary_color=boundary_color, marker_color=marker_color,
                              title=title, visible=visible)

    for i in range(1024):
        point1 = boundary1[i]
        point2 = boundary2[i]

        # for j in range(point2[1], point1[1] + 1):
        #     pano_img[np.clip(j, 0, h - 1), np.clip(i, 0, w - 1)] = boundary_color

        pano_img[np.clip(point2[1], 0, h - 1): np.clip(point1[1], 0, h - 1), np.clip(i, 0, w - 1)] = boundary_color

    if show:
        plt.figure(figsize=(10, 5))
        if title is not None:
            plt.title(title)

        plt.axis('off')
        plt.imshow(pano_img)
        plt.show()

    return pano_img


if __name__ == '__main__':

    import numpy as np
    from PIL import Image
    from dataset.communal.read import read_label
    import os
    from tqdm import tqdm

    split_dir = '.../Datasets/mp3d/split'
    label_dir = '.../Datasets/mp3d/label'
    save_dir = '.../Datasets/mp3d/segmentation-mask'

    with open(os.path.join(split_dir, "val.txt"), 'r') as f:
        split_list = [x.rstrip().split() for x in f]

    for name in tqdm(split_list):
        name = "_".join(name)
        label_path = os.path.join(label_dir, f"{name}.json")
        label = read_label(label_path, data_type='MP3D')

        corners = label['corners']
        ratio = label['ratio']
        pano_img = np.zeros([512, 1024, 3])

        pano_img = draw_boundaries(pano_img, corners_list=[corners], show=True, length=1024, ratio=ratio, visible=True)

        pano_img = pano_img[:, :, 0]

        Image.fromarray((pano_img).astype(np.uint8)).save(
            os.path.join(save_dir, f"{label['id']}.png"))
