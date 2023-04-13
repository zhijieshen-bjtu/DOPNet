""" 
@Date: 2021/07/18
@description:
"""
import os
import models
import torch.distributed as dist
import torch

from torch.nn import init
from torch.optim import lr_scheduler
from models.other.optimizer import build_optimizer
from models.other.criterion import build_criterion


def build_model(config, logger):
    name = config.MODEL.NAME

    net = getattr(models, name)
    ckpt_dir = os.path.abspath(os.path.join(config.CKPT.DIR, os.pardir)) if config.DEBUG else config.CKPT.DIR

    model = net(ckpt_dir=ckpt_dir)

    logger.info(f'model dropout: {model.dropout_d}')
    device = config.TRAIN.DEVICE
    model = model.to(device)
    optimizer = None
    scheduler = None

    if config.MODE == 'train':
        optimizer = build_optimizer(config, model, logger)

    config.defrost()
    config.TRAIN.START_EPOCH = model.load(device, logger,  optimizer, best=config.MODE != 'train' or not config.TRAIN.RESUME_LAST)
    config.freeze()

    if config.MODE == 'train':
        if len(config.TRAIN.LR_SCHEDULER.NAME) > 0:
            if 'last_epoch' not in config.TRAIN.LR_SCHEDULER.ARGS[0].keys():
                config.TRAIN.LR_SCHEDULER.ARGS[0]['last_epoch'] = config.TRAIN.START_EPOCH - 1

            scheduler = getattr(lr_scheduler, config.TRAIN.LR_SCHEDULER.NAME)(optimizer=optimizer,
                                                                              **config.TRAIN.LR_SCHEDULER.ARGS[0])
            logger.info(f"Use scheduler: name:{config.TRAIN.LR_SCHEDULER.NAME} args: {config.TRAIN.LR_SCHEDULER.ARGS[0]}")
            logger.info(f"Current scheduler last lr: {scheduler.get_last_lr()}")
        else:
            scheduler = None

    criterion = build_criterion(config, logger)
    if optimizer is not None:
        logger.info(f"Finally lr: {optimizer.param_groups[0]['lr']}")
    return model, optimizer, criterion, scheduler
