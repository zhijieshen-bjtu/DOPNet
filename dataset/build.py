"""
@Date: 2022/07/18
@description:
"""
import numpy as np
import torch.utils.data
from dataset.mp3d_dataset import MP3DDataset
from dataset.pano_s2d3d_dataset import PanoS2D3DDataset
from dataset.pano_s2d3d_mix_dataset import PanoS2D3DMixDataset
from dataset.zind_dataset import ZindDataset


def build_loader(config, logger):
    train_dataset = None
    train_data_loader = None
    if config.MODE == 'train':
        train_dataset = build_dataset(mode='train', config=config, logger=logger)

    val_dataset = build_dataset(mode='test', config=config, logger=logger)

    train_sampler = None
    val_sampler = None

    batch_size = config.DATA.BATCH_SIZE
    num_workers = 0 if config.DEBUG else config.DATA.NUM_WORKERS
    if train_dataset:
        logger.info(f'Train data loader batch size: {batch_size}')
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    batch_size = 1
    logger.info(f'Val data loader batch size: {batch_size}')
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, sampler=val_sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_data_loader, val_data_loader


def build_dataset(mode, config, logger):
    name = config.DATA.DATASET
    if name == 'mp3d':
        dataset = MP3DDataset(
            root_dir=config.DATA.DIR,
            mode=mode,
            shape=config.DATA.SHAPE,
            max_wall_num=config.DATA.WALL_NUM,
            aug=config.DATA.AUG if mode == 'train' else None,
            logger=logger,
            keys=config.DATA.KEYS
        )
    elif name == 'pano_s2d3d':
        dataset = PanoS2D3DDataset(
            root_dir=config.DATA.DIR,
            mode=mode,
            shape=config.DATA.SHAPE,
            max_wall_num=config.DATA.WALL_NUM,
            aug=config.DATA.AUG if mode == 'train' else None,
            logger=logger,
            subset=config.DATA.SUBSET,
            keys=config.DATA.KEYS
        )
    elif name == 'pano_s2d3d_mix':
        dataset = PanoS2D3DMixDataset(
            root_dir=config.DATA.DIR,
            mode=mode,
            shape=config.DATA.SHAPE,
            max_wall_num=config.DATA.WALL_NUM,
            aug=config.DATA.AUG if mode == 'train' else None,
            logger=logger,
            subset=config.DATA.SUBSET,
            keys=config.DATA.KEYS
        )
    elif name == 'zind':
        dataset = ZindDataset(
            root_dir=config.DATA.DIR,
            mode=mode,
            shape=config.DATA.SHAPE,
            max_wall_num=config.DATA.WALL_NUM,
            aug=config.DATA.AUG if mode == 'train' else None,
            logger=logger,
            is_simple=True,
            is_ceiling_flat=False,
            keys=config.DATA.KEYS,
            vp_align=config.EVAL.POST_PROCESSING is not None and 'manhattan' in config.EVAL.POST_PROCESSING
        )
    else:
        raise NotImplementedError(f"Unknown dataset: {name}")

    return dataset
