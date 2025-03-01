# --------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 7/28/2024 15:32
# @Author  : Ding Yang
# @Project : OpenMedStereo
# @Device  : Moss
# --------------------------------------
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import time
from datetime import datetime
from os import path as osp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Dataset import fetch_dataset
from Models.utils import common_utils
from Models.utils.clip_grad import ClipGrad
from Models.utils.common_utils import color_map_tensorboard, write_tensorboard
from Models.utils.lamb import Lamb
from Models.utils.warmup import LinearWarmup
from tools.metrics import EPE_metric, D1_metric


class TrainerTemplate:
    def __init__(self, args, cfgs, model):
        self.args = args
        self.cfgs = cfgs
        self.logger = self.build_logger()
        self.tb_writer = self.build_tb_writer()
        for key, val in vars(args).items():
            self.logger.info('{:16} {}'.format(key, val))
        self.model = self.build_model(model)

        if self.args.run_mode == 'train':
            self.train_set, self.train_loader = self.build_train_loader()
            self.eval_set, self.eval_loader = self.build_eval_loader()
            self.total_epochs = cfgs.OPTIMIZATION.NUM_EPOCHS
            self.last_epoch = -1

            self.optimizer, self.scheduler = self.build_optimizer_and_scheduler()
            self.scaler = torch.cuda.amp.GradScaler(enabled=cfgs.OPTIMIZATION.AMP)

            if self.args.resume is not None:
                self.resume_ckpt()

            self.warmup_scheduler = self.build_warmup()
            self.clip_gard = self.build_clip_grad()

    def build_logger(self):
        TIMESTAMP = "{0:%Y%m%d%H%M%S}".format(datetime.now())
        logger = logging.getLogger()
        formatter = logging.Formatter('%(name)s %(levelname)s %(message)s')
        logger.setLevel(logging.INFO)
        f_handler = logging.FileHandler(filename=osp.join(self.args.folder, f"{TIMESTAMP}-{self.args.run_mode}.log"),
                                        mode='a')
        s_handler = logging.StreamHandler()
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(formatter)
        s_handler.setLevel(logging.INFO)
        s_handler.setFormatter(formatter)
        logger.addHandler(f_handler)
        logger.addHandler(s_handler)
        return logger

    def build_tb_writer(self):
        TIMESTAMP = "{0:%Y%m%d%H%M%S}".format(datetime.now())
        tb_writer = SummaryWriter(log_dir=osp.join(self.args.folder, TIMESTAMP))
        return tb_writer

    def build_train_loader(self):
        train_set = fetch_dataset(self.cfgs.DATA_CONFIG, mode='train')
        train_loader = DataLoader(train_set, batch_size=self.cfgs.OPTIMIZATION.BATCH_SIZE_PER_GPU, shuffle=True)
        return train_set, train_loader

    def build_eval_loader(self):
        val_set = fetch_dataset(self.cfgs.DATA_CONFIG, mode='val')
        val_loader = DataLoader(val_set, batch_size=self.cfgs.EVALUATOR.BATCH_SIZE_PER_GPU, shuffle=False)
        return val_set, val_loader

    def build_model(self, model):
        if self.cfgs.OPTIMIZATION.get('FREEZE_BN', False):
            model = common_utils.freeze_bn(model)
            self.logger.info('Freeze the batch normalization layers')

        if self.cfgs.OPTIMIZATION.SYNC_BN and self.args.dist_mode:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.logger.info('Convert batch norm to sync batch norm')
        if self.args.dist_mode:
            model = nn.DataParallel(model)
        # load pretrained model
        if self.cfgs.MODEL.PRETRAINED_MODEL:
            self.logger.info('Loading parameters from checkpoint %s' % self.cfgs.MODEL.PRETRAINED_MODEL)
            if not os.path.isfile(self.cfgs.MODEL.PRETRAINED_MODEL):
                raise FileNotFoundError
            common_utils.load_params_from_file(
                model, self.cfgs.MODEL.PRETRAINED_MODEL,
                dist_mode=self.args.dist_mode, logger=self.logger, strict=False)
        model = model.cuda()
        return model

    def build_optimizer_and_scheduler(self):
        if self.cfgs.OPTIMIZATION.OPTIMIZER.NAME == 'Lamb':
            optimizer_cls = Lamb
        else:
            optimizer_cls = getattr(torch.optim, self.cfgs.OPTIMIZATION.OPTIMIZER.NAME)
        valid_arg = common_utils.get_valid_args(optimizer_cls, self.cfgs.OPTIMIZATION.OPTIMIZER, ['name'])
        optimizer = optimizer_cls(params=[p for p in self.model.parameters() if p.requires_grad], **valid_arg)

        self.cfgs.OPTIMIZATION.SCHEDULER.TOTAL_STEPS = self.total_epochs * len(self.train_loader)
        scheduler_cls = getattr(torch.optim.lr_scheduler, self.cfgs.OPTIMIZATION.SCHEDULER.NAME)
        valid_arg = common_utils.get_valid_args(scheduler_cls, self.cfgs.OPTIMIZATION.SCHEDULER, ['name', 'on_epoch'])
        scheduler = scheduler_cls(optimizer, **valid_arg)

        return optimizer, scheduler

    def resume_ckpt(self):
        self.logger.info(f'Resume from ckpt: {self.args.resume}')
        checkpoint = torch.load(self.args.resume)
        self.last_epoch = checkpoint['epoch']
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        if self.args.dist_mode:
            self.model.module.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint['model_state'])

    def build_warmup(self):
        last_step = (self.last_epoch + 1) * len(self.train_loader) - 1
        if 'WARMUP' in self.cfgs.OPTIMIZATION.SCHEDULER:
            warmup_steps = self.cfgs.OPTIMIZATION.SCHEDULER.WARMUP.get('WARM_STEPS', 1)
            warmup_scheduler = LinearWarmup(
                self.optimizer,
                warmup_period=warmup_steps,
                last_step=last_step)
        else:
            warmup_scheduler = LinearWarmup(
                self.optimizer,
                warmup_period=1,
                last_step=last_step)

        return warmup_scheduler

    def build_clip_grad(self):
        clip_gard = None
        if 'CLIP_GRAD' in self.cfgs.OPTIMIZATION:
            clip_type = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('TYPE', None)
            clip_value = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('CLIP_VALUE', 0.1)
            max_norm = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('MAX_NORM', 35)
            norm_type = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('NORM_TYPE', 2)
            clip_gard = ClipGrad(clip_type, clip_value, max_norm, norm_type)
        return clip_gard

    def train(self, current_epoch, tbar):
        self.model.train()
        if self.cfgs.OPTIMIZATION.get('FREEZE_BN', False):
            self.model = common_utils.freeze_bn(self.model)
        self.train_one_epoch(current_epoch=current_epoch, tbar=tbar)
        if self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH:
            self.scheduler.step()
            self.warmup_scheduler.lrs = [group['lr'] for group in self.optimizer.param_groups]

    def evaluate(self, current_epoch):
        self.model.eval()
        self.eval_one_epoch(current_epoch=current_epoch)

    def save_ckpt(self, current_epoch):
        if current_epoch % self.cfgs.TRAINER.CKPT_SAVE_INTERVAL == 0 or current_epoch == self.total_epochs - 1:
            ckpt_name = osp.join(self.args.ckpt_dir, 'checkpoint_epoch_%d.pth' % current_epoch)
            common_utils.save_checkpoint(self.model, self.optimizer, self.scheduler, self.scaler,
                                         self.args.dist_mode, current_epoch, filename=ckpt_name)

    def train_one_epoch(self, current_epoch, tbar):
        start_epoch = self.last_epoch + 1
        logger_iter_interval = self.cfgs.TRAINER.LOGGER_ITER_INTERVAL
        total_loss = 0.0
        loss_func = self.model.module.get_loss if self.args.dist_mode else self.model.get_loss
        train_loader_iter = iter(self.train_loader)
        for i in range(0, len(self.train_loader)):
            self.optimizer.zero_grad()
            lr = self.optimizer.param_groups[0]['lr']

            start_timer = time.time()
            data = next(train_loader_iter)
            for k, v in data.items():
                data[k] = v.cuda() if torch.is_tensor(v) else v
            data_timer = time.time()
            with torch.cuda.amp.autocast(enabled=self.cfgs.OPTIMIZATION.AMP):
                model_pred = self.model(data)
                infer_timer = time.time()
                loss, tb_info = loss_func(model_pred, data['disp'])
            self.logger.info(f"scaler scale: {self.scaler.get_scale()}, loss: {loss.item()}")
            # 不要在autocast下调用, calls backward() on scaled loss to create scaled gradients.
            self.scaler.scale(loss).backward()

  
            # 做梯度剪裁的时候需要先unscale, unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)
            # 梯度剪裁
            if self.clip_gard is not None:
                self.clip_gard(self.model)
            # optimizer's gradients are already unscaled, so scaler.step does not unscale them
            self.scaler.step(self.optimizer)
           
            # Updates the scale for next iteration.
            self.scaler.update()
            # torch.cuda.empty_cache()

            # warmup_scheduler period>1 和 batch_scheduler 不要同时使用
            with self.warmup_scheduler.dampening():
                if not self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH:
                    self.scheduler.step()

            total_loss += loss.item()
            total_iter = current_epoch * len(self.train_loader) + i
            trained_time_past_all = tbar.format_dict['elapsed']
            single_iter_second = trained_time_past_all / (total_iter + 1 - start_epoch * len(self.train_loader))
            remaining_second_all = single_iter_second * (self.total_epochs * len(self.train_loader) - total_iter - 1)
            if total_iter % logger_iter_interval == 0:
                message = ('Training Epoch:{:>2d}/{} Iter:{:>4d}/{} '
                           'Loss:{:#.6g}({:#.6g}) LR:{:.4e} '
                           'DataTime:{:.2f} InferTime:{:.2f}ms '
                           'Time cost: {}/{}'
                           ).format(current_epoch, self.total_epochs, i, len(self.train_loader),
                                    loss.item(), total_loss / (i + 1), lr,
                                    data_timer - start_timer, (infer_timer - data_timer) * 1000,
                                    tbar.format_interval(trained_time_past_all),
                                    tbar.format_interval(remaining_second_all))
                self.logger.info(message)

            if self.cfgs.TRAINER.TRAIN_VISUALIZATION:
                tb_info['image/train/image'] = torch.cat([data['left'][0], data['right'][0]], dim=1) / 256
                tb_info['image/train/disp'] = color_map_tensorboard(data['disp'][0],
                                                                    model_pred['disp_pred'].squeeze(1)[0])

            tb_info.update({'scalar/train/lr': lr})
            if total_iter % logger_iter_interval == 0 and self.tb_writer is not None:
                write_tensorboard(self.tb_writer, tb_info, total_iter)
    
    @torch.no_grad()
    def eval_one_epoch(self, current_epoch):

        metric_func_dict = {
            'epe': epe_metric,
            'd1_all': d1_metric
        }

        evaluator_cfgs = self.cfgs.EVALUATOR

        epoch_metrics = {}
        for k in evaluator_cfgs.METRIC:
            epoch_metrics[k] = Average_item()

        for i, data in enumerate(self.eval_loader):
            for k, v in data.items():
                data[k] = v.cuda() if torch.is_tensor(v) else v
            bz = v.shape[0] 
            with torch.cuda.amp.autocast(enabled=self.cfgs.OPTIMIZATION.AMP):
                infer_start = time.time()
                model_pred = self.model(data)
                infer_time = time.time() - infer_start

            disp_pred = model_pred['disp_pred']
            disp_gt = data["disp"]
            mask = (disp_gt < evaluator_cfgs.MAX_DISP) & (disp_gt > 0)
            if 'occ_mask' in data and evaluator_cfgs.get('APPLY_OCC_MASK', False):
                mask = mask & ~data['occ_mask'].to(torch.bool)

            for m in evaluator_cfgs.METRIC:
                if m not in metric_func_dict:
                    raise ValueError("Unknown metric: {}".format(m))
                metric_func = metric_func_dict[m]
                res = metric_func(disp_pred.squeeze(1), disp_gt, mask)
                # epoch_metrics[m]['indexes'].extend(data['index'].tolist())
                # epoch_metrics[m]['values'].extend(res.tolist())
                epoch_metrics[m].update(res.item(), bz)
            if i % self.cfgs.TRAINER.LOGGER_ITER_INTERVAL == 0:
                message = ('Evaluating Epoch:{:>2d} Iter:{:>4d}/{} InferTime: {:.2f}ms'
                           ).format(current_epoch, i, len(self.eval_loader), infer_time * 1000)
                self.logger.info(message)

                if self.cfgs.TRAINER.EVAL_VISUALIZATION and self.tb_writer is not None:
                    tb_info = {
                        'image/eval/image': torch.cat([data['left'][0], data['right'][0]], dim=1) / 256,
                        'image/eval/disp': color_map_tensorboard(data['disp'][0], model_pred['disp_pred'].squeeze(1)[0])
                    }
                    write_tensorboard(self.tb_writer, tb_info, current_epoch * len(self.eval_loader) + i)

        results = {}
        for k in epoch_metrics.keys():
            # results[k] = torch.tensor(epoch_metrics[k]["values"]).mean()
            results[k] = epoch_metrics[k].mean

        if self.tb_writer is not None:
            tb_info = {}
            for k, v in results.items():
                # tb_info[f'scalar/val/{k}'] = v.item()
                tb_info[f"scalar/val/{k}"] = v

            write_tensorboard(self.tb_writer, tb_info, current_epoch)

        self.logger.info(f"Epoch {current_epoch} metrics: {results}")

class Average_item:
    def __init__(self):
        self.mean = 0
        self.cnt = 0

    def update(self, value, n=1):
        self.cnt += n
        self.mean += (value - self.mean) * n / self.cnt

    def zero(self):
        self.mean = 0
        self.cnt = 0

def check_for_nan(name, tensor):
    if torch.isnan(tensor).any():
        print(f"NaN found in {name}")

def print_params(model, save_path):
        with open(save_path, 'w') as f:
            # 遍历模型的参数
            for name, param in model.named_parameters():
                check_for_nan(name, param)
                # 写入参数名称
                f.write(f'{name}:\n')
                # 写入参数值
                f.write(f'{param.data}\n')
                # 分隔不同参数
                f.write('-' * 40 + '\n')
def print_grad(model,save_path):
    with open(save_path, 'w') as f:
        # 遍历模型的参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                check_for_nan(name, param)
                # 写入参数名称
                f.write(f'{name}\n')
                # 写入参数值
                f.write(f'{param.data}\n')
                # 写入梯度
                f.write(f'{param.grad}\n')
                # 分隔不同参数
                f.write('-' * 40 + '\n')
