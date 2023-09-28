#!/usr/bin/env python
# coding=utf8


import os
import time
import torch
import numpy as np


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, val_n=1):
        self.val = val
        self.sum += val * val_n
        self.count += val_n
        self.avg = self.sum / self.count


def meter_to_str(info_title, avg_meter, round_bit):
    return "{} {}({})\t".format(info_title,
                                round(avg_meter.val, round_bit), round(avg_meter.avg, round_bit))


def save_checkpoint(epoch, file_index, model, model_optimizer, model_folder):
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    filename = 'checkpoint-{}-{}.pth.tar'.format(epoch, file_index)
    opt_filename = 'opt-{}-{}.pth.tar'.format(epoch, file_index)
    for _ in range(5):
        try:
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), os.path.join(model_folder, filename))
            else:
                torch.save(model.state_dict(), os.path.join(model_folder, filename))
            torch.save(model_optimizer.state_dict(), os.path.join(model_folder, opt_filename))
            break
        except:
            time.sleep(5)


def save_checkpoint_ori(epoch, file_index, model, model_optimizer, model_folder):
    state = {'epoch': epoch,
             'model': model,
             'model_optimizer': model_optimizer}
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    filename = 'checkpoint-{}-{}.pth.tar'.format(epoch, file_index)
    for _ in range(5):
        try:
            torch.save(state, os.path.join(model_folder, filename))
            break
        except:
            time.sleep(5)


def load_model(checkpoint_path, model):
    if not os.path.exists(checkpoint_path):
        print("No input model {}".format(checkpoint_path))
        return model
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), True)
    except Exception as err:
        print("Error in load model", err)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), True)
    if hasattr(model, "module"):
        model = model.module  # 如果是多GPU的DataParallel格式则使用model.module，否则（单gpu）则去掉该行
    return model


def load_optimizer(checkpoint_path, optimizer):
    if not os.path.exists(checkpoint_path):
        print("No input optimizer {}".format(checkpoint_path))
        return optimizer
    optim_recover = torch.load(checkpoint_path, map_location='cpu')
    if hasattr(optim_recover, 'state_dict'):
        optim_recover = optim_recover.state_dict()
    optimizer.load_state_dict(optim_recover)
    return optimizer


def load_model_ori(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
    except:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = checkpoint['model']
    if hasattr(model, "module"):
        model = model.module  # 如果是多GPU的DataParallel格式则使用model.module，否则（单gpu）则去掉该行
    return model


def load_model_and_transform(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
    except:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = checkpoint['model']
    if hasattr(model, "module"):
        model = model.module  # 如果是多GPU的DataParallel格式则使用model.module，否则（单gpu）则去掉该行
    transform = checkpoint["transform"]
    return model, transform


def compute_cos_sim(vec1, vec2):
    return float(np.sum(vec1 * vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))