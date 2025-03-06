# --------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 9/5/2024 13:17
# @Author  : Ding Yang
# @Project : OpenMedStereo
# @Device  : Moss
# --------------------------------------
import sys
import os
sys.path.insert(0, "./")
import argparse
import tqdm
from easydict import EasyDict
import torch.distributed as dist
from Models import build_trainer
from Models.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # mode
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default=None, required=True, help='specify the config for training')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--resume', type=str, default=None, help='the ckp path')
    # save path
    parser.add_argument('--folder', type=str, default='./output', help='save root dir for this experiment')

    args = parser.parse_args()
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)
    args.run_mode = 'train'
    return args, cfgs


def main():
    args, cfgs = parse_config()
    # env
    if args.fix_random_seed:
        seed = 0 if not args.dist_mode else dist.get_rank()
        common_utils.set_random_seed(seed=seed)
    if not os.path.exists(args.folder):
        os.makedirs(args.folder, exist_ok=True)
    args.ckpt_dir = os.path.join(args.folder, 'ckpt')
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir, exist_ok=True)

    # trainer
    model_trainer = build_trainer(args, cfgs)

    tbar = tqdm.trange(model_trainer.last_epoch + 1, model_trainer.total_epochs,
                       desc='epochs', dynamic_ncols=True, bar_format='{l_bar}{bar}{r_bar}\n')
    # train loop
    for current_epoch in tbar:
        model_trainer.train(current_epoch, tbar)
        model_trainer.save_ckpt(current_epoch)
        if current_epoch % cfgs.TRAINER.EVAL_INTERVAL == 0 or current_epoch == model_trainer.total_epochs - 1:
            model_trainer.evaluate(current_epoch)


if __name__ == '__main__':
    main()