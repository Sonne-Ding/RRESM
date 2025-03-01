import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from functools import wraps
from torch.utils.data import DistributedSampler
import torch.nn.functional as F
import numpy as np
import time
from Dataset.base_dataset import ScaredTrainBase
from tools.logger import BaseLogger
from tools.metrics import EPE_metric, D1_metric, Thres_metric, AverageMeterDict
from tools.data_convert import tensor2float
from torch.utils.data import DataLoader
import gc
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import multiprocessing as mp
from Models import LightEndoStereo
from Models.LightEndoStereo import loss as LossFuncs
import timm
import wandb
# from apex import amp
# import cv2

def merge_dicts(dicts_list):
    """
        :param metric_dicts: list[dict, dict, ...]
    """
    keys = dicts_list[0].keys()
    # 初始化一个字典来存储平均值
    avg_dict = {key: 0.0 for key in keys}
    # 计算每个键的值的总和
    for d in dicts_list:
        for key,value in d.items():
            if isinstance(value, list):
                value = np.mean(value)
            avg_dict[key] += value
    # 计算平均值
    for key in keys:
        avg_dict[key] /= len(dicts_list)
    return avg_dict
    
def init_wandb(proj, expname, expinfo):
    wandb.init(project=proj,
               name=expname,
               notes=expinfo)
    wandb.define_metric("train/step")
    wandb.define_metric("val/step")
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("val/loss", step_metric="val/step")
    wandb.define_metric("val/*", step_metric="epoch")

def train(config, model, trainLoader, testLoader, optimizer, lrScheduler, rank, flag):
    # parse arguments, set seeds
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    os.makedirs(config.logdir, exist_ok=True)
    # create summary logger only in the rank 0 process
    if rank == 0:
        msg_logger = BaseLogger(config.logdir)
        msg_logger.info(config.info)
        init_wandb(proj="GwcNet",
                   expname=config.expname,
                   expinfo=config.info)
    # load parameters
    start_epoch = 0
    model.train()
    best_epe = 50.0
    trainSampler = trainLoader.sampler
    lossFunc = getattr(LossFuncs, config.loss)
    global_train_step = 0
    global_val_step = 0
    for epoch_idx in range(start_epoch, config.epochs):
        # training
        trainSampler.set_epoch(epoch_idx)
        for batch_idx, sample in enumerate(trainLoader):
            if flag[0]:
                raise InterruptedError("Interrupted by other process.")
            # global_step = len(trainLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_train_step % config.summary_freq == 0
            loss, scalar_outputs= train_sample(model, optimizer, sample, config.maxdisp, lossFunc,compute_metrics=do_summary)
            # assert not torch.isnan(loss), "loss is nan"
            del scalar_outputs
            if rank == 0:
                msg_logger.info('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, config.epochs, batch_idx, len(trainLoader), loss, time.time() - start_time))
                wandb.log({
                    "train/step": global_train_step,
                    "train/loss": loss,
                })
            global_train_step += 1
        # saving checkpoints
        if (epoch_idx + 1) % config.save_freq == 0 and rank == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.module.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(config.logdir, epoch_idx))
        lrScheduler.step()
        if (epoch_idx + 1) % config.val_freq == 0:
            avg_test_scalars = AverageMeterDict()
            for batch_idx, sample in enumerate(testLoader):
                if flag[0]:
                    raise InterruptedError("Interrupted by other process.")
                start_time = time.time()
                do_summary = global_val_step % config.summary_freq == 0
                loss, scalar_outputs = test_sample(model, sample,config.maxdisp, lossFunc,compute_metrics=do_summary)
                avg_test_scalars.update(scalar_outputs)
                del scalar_outputs
                if rank==0:
                    msg_logger.info('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, config.epochs,batch_idx, len(testLoader), loss, time.time() - start_time))
                    wandb.log({
                        "val/step": global_val_step,
                        "val/loss": loss,  # 使用 "test_loss" 来区分训练和测试阶段的损失
                    })
                global_val_step += 1
            avg_test_scalars = avg_test_scalars.mean()
            if rank == 0:
                gather_test_scalers = [dict() for _ in range(dist.get_world_size())]
                dist.gather_object(avg_test_scalars, gather_test_scalers, dst=0)
            else:
                dist.gather_object(avg_test_scalars, None, dst=0)
            dist.barrier()
            if rank==0:
                avg_test_scalars = merge_dicts(gather_test_scalers)
                msg_logger.info(f"avg_test_scalars: {avg_test_scalars}")
                metric_wandb = {}
                for k,v in avg_test_scalars.items():
                    metric_wandb[f"val/{k}"] = v
                metric_wandb["epoch"] = epoch_idx
                wandb.log(metric_wandb)
                test_epe_mean = np.mean(avg_test_scalars["EPE"])
                if test_epe_mean < best_epe:
                    best_epe = test_epe_mean
                    checkpoint_data = {'epoch': epoch_idx, 'model': model.module.state_dict(), 'optimizer': optimizer.state_dict()}
                    torch.save(checkpoint_data, "{}/checkpoint_bestEpe.ckpt".format(config.logdir))
            gc.collect()
    if rank==0:
        msg_logger.info(f"The best epe={best_epe}")
        wandb.finish()
    # 如果时正常退出，则等待所有进程完成训练
    dist.barrier()


# train one sample
def train_sample(model, optimizer, sample, maxdisp, lossFunc,compute_metrics=False):
    model.train()
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disp']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.squeeze(1).cuda()
    optimizer.zero_grad()
    disp_ests = model(imgL, imgR)
    assert not torch.any(torch.isnan(disp_ests[0])), "disp ests has nan"
    mask = (disp_gt < maxdisp) & (disp_gt > 0)
    loss = lossFunc(disp_ests, disp_gt, mask)
    scalar_outputs = {"loss": loss}
    # image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            # image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, disp_gt) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    loss.backward()
    optimizer.step()
    dist.barrier()
    return tensor2float(loss), tensor2float(scalar_outputs)

@torch.no_grad()
def test_sample(model, sample, maxdisp, lossFunc,compute_metrics=True):
    model.eval()
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disp']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.squeeze(1).cuda()
    mask = (disp_gt < maxdisp) & (disp_gt > 0)
    disp_ests = model(imgL, imgR)
    loss = lossFunc(disp_ests, disp_gt, mask)
    scalar_outputs = {"loss": loss}
    # image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    # image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, disp_gt, mask, 'disp') for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    return tensor2float(loss), tensor2float(scalar_outputs)

def setup_DDP(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size,)
    torch.cuda.set_device(rank)
    dist.barrier()

def cleanup():
    if dist.is_initialized():
        dist.barrier()  
        dist.destroy_process_group()

def safe_procs(func):
    # 保证出现错误时，也能调用cleanup清理资源，防止Ubuntu中出现文件占用或内存泄漏的问题。
    @wraps(func)
    def wrapper(*args, **kwargs):
        rank = args[0]
        flag = args[-1]
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(f"Processing {rank} caught an {e}")
        # KeyboardInterrupt属于BaseException
        except BaseException as e:
            print(f"Processing {rank} caught an {e}")
        finally:
            print(f"Processing {rank} is ready to clean.")
            flag[0] = True # Interrupt other processing
            cleanup()
    return wrapper

def get_trainloader(config):
    train_dataset = ScaredTrainBase(config, mode='train')
    # With setup_DDP, the sampler will retrive the rank and world_size from the environment. So we don't need to pass them to the sampler.
    sampler = DistributedSampler(train_dataset)
    TrainImgLoader = DataLoader(train_dataset, config.batch_size, shuffle=False, num_workers=config.num_workers, drop_last=True, sampler=sampler)
    return TrainImgLoader

def get_testloader(config):
    test_dataset = ScaredTrainBase(config, mode='val')
    sampler = DistributedSampler(test_dataset, shuffle=False)
    TestImgLoader = DataLoader(test_dataset, config.batch_size, shuffle=False, num_workers=config.num_workers, drop_last=False, sampler=sampler)
    return TestImgLoader

def group_params(model):
    feature_group = []
    other_group = []
    for name, param in model.named_parameters():
        if name.startswith("feature_extraction"):
            feature_group.append(param)
        else:
            other_group.append(param)
    return feature_group, other_group

def get_model_optimizer_lrScheduler(model_config, exp_config):
    # model = timm.create_model(model_config.model, maxdisp=model_config.maxdisp,
                            #   use_concat_volume=model_config.use_concat_volume,
                            #   featureNet=model_config.featureNet)
    model = timm.create_model(model_config.model, **model_config)
    if model_config.syncBN:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(torch.cuda.current_device())
    else:
        model = model.to(torch.cuda.current_device())
    # optimizer = getattr(optim, exp_config.optimizer.name)(params=model.parameters(),
    #                                                       lr=exp_config.optimizer.lr,
    #                                                       betas=tuple(exp_config.optimizer.betas))
    # fetch optimizer
    feature_group, other_group = group_params(model)
    optimizer = getattr(optim, exp_config.optimizer.name)([
        {"params":feature_group, "lr": exp_config.optimizer.lr*0.3, "betas": tuple(exp_config.optimizer.betas)}, 
        {"params":other_group, "lr": exp_config.optimizer.lr, "betas": tuple(exp_config.optimizer.betas)}
        ])
    lrScheduler = getattr(optim.lr_scheduler, exp_config.lr_scheduler.name)(optimizer,milestones=exp_config.lr_scheduler.milestones,gamma=exp_config.lr_scheduler.gamma)
    # Wrap model with DDP
    model = DDP(model, device_ids=[torch.cuda.current_device()],)
    return model, optimizer, lrScheduler

@safe_procs
def worker(rank, config, flag):
    setup_DDP(rank, config.exp_config.world_size, config.exp_config.port)
    trainloader = get_trainloader(config.dataset_config.trainSet)
    testloader = get_testloader(config.dataset_config.valSet)
    model, optimizer, lrScheduler = get_model_optimizer_lrScheduler(config.model_config, config.exp_config)
    train(config.exp_config, model, trainloader, testloader, optimizer, lrScheduler, rank, flag) 
     