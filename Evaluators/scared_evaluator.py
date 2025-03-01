from __future__ import print_function, division
from Dataset.img_reader import rgb_reader, tiff_reader
import torchvision
import timm
import multiprocessing as mp
import os
from os import path as osp
import torch
import yaml
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
from torch.utils.data import DataLoader
from tools.metrics import D1_metric, EPE_metric, Thres_metric,AverageMeterDict
from tools.visualization import save_images_local, img_rainbow_func, errormap_rainbow_func, img_only_rainbow_func
from tools.data_convert import tensor2float
import cv2
from Models import LightEndoStereo
cudnn.benchmark = True
from tqdm import tqdm
from Dataset.base_dataset import ScaredDatasetTest
from torchvision.transforms import functional as F

def scale_tensor_img(tensor_img, mask=None):
    if mask is not None:
        tensor_img[torch.logical_not(mask)] = 0
    tensor_img = tensor_img.clip(min=0.0, max=250.0)
    return tensor_img

@torch.no_grad()
def test_sample(config, model, sample, Q, device):
    """
        :return: scalar_outputs, image_outputs
    """
    imgL, imgR, disp_gt =  sample['left'], sample['right'], sample['disp']
    imgL = imgL.to(device)
    imgR = imgR.to(device)
    disp_gt = disp_gt.squeeze(1).to(device)
    disp_est = model(imgL, imgR)[-1]
    mask = (disp_gt > 0) & (disp_gt<config.maxdisp)
    if config.depth:
        depth_gt = disp2depth(disp_gt, Q)
        depth_est = disp2depth(disp_est, Q)
    else:
        depth_gt = disp_gt
        depth_est = disp_est
    
    scalar_outputs = {}
    scalar_outputs["D1"] = D1_metric(depth_est, depth_gt, mask)
    scalar_outputs["EPE"] = EPE_metric(depth_est, depth_gt, mask)
    scalar_outputs["Thres1"] = Thres_metric(depth_est, depth_gt, mask, 1.0)
    scalar_outputs["Thres2"] = Thres_metric(depth_est, depth_gt, mask, 2.0)
    scalar_outputs["Thres3"] = Thres_metric(depth_est, depth_gt, mask, 3.0)
    depth_est = scale_tensor_img(depth_est)
    depth_gt = scale_tensor_img(depth_gt, mask)
    image_outputs = {
        "depth_est": img_only_rainbow_func.apply(depth_est, "depth"),
        "depth_gt": img_rainbow_func.apply(depth_gt,mask, "depth"),
        "abs_errormap": errormap_rainbow_func.apply(depth_est, depth_gt, mask, 'depth'),
        "rel_errormap": errormap_rainbow_func.apply(depth_est, depth_gt, mask, 'rel'),
        "imgL": imgL,
        "imgR": imgR
        }
    return tensor2float(scalar_outputs), image_outputs

def disp2depth(disp, Q):
    if disp.is_cuda:
        disp_np = disp.squeeze().cpu().numpy()
    else:
        disp_np = disp.squeeze().numpy()
    depth_np = cv2.reprojectImageTo3D(disp_np, Q)[:, :, -1]
    depth_np = np.nan_to_num(depth_np, posinf=0, neginf=0, nan=0)
    depth_tensor = torch.tensor(depth_np, device=disp.device, dtype=disp.dtype).unsqueeze(0)
    return depth_tensor

def test_on_dataset(config, testDataloader, model, tbar,Q, savepath, device):
    """
        :param dataset: Choices['dataset_8', 'dataset_9']
    """
    model.eval()
    avg_test_scalars = AverageMeterDict()
    for batch_idx, sample in enumerate(testDataloader):
        scalar_outputs, image_outputs = test_sample(config, model, sample, Q, device)
        if scalar_outputs['EPE'] != 0:
            avg_test_scalars.update(scalar_outputs)
        # else:
        #     print(f"Abandon sample {batch_idx}")
        if batch_idx < config.savefig:
            sample_name = osp.basename(sample['disp_filename'][0]).split('.')[0]
            img_savepath = osp.join(savepath, sample_name)
            save_images_local(img_savepath, image_outputs)
        tbar.update(1)
    tbar.close()
    avg_test_scalars = avg_test_scalars.mean()
    return avg_test_scalars

def fetch_model(config, device):
    model = timm.create_model(config.model,**config)
    if config.syncBN:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    return model

    
def fetch_dataloader(config, ds, kf):
    dataset = ScaredDatasetTest(config, ds, kf)
    Q = dataset.get_Q()
    test_dataloader = DataLoader(dataset, shuffle=False, batch_size=config.batch_size, num_workers=config.num_workers)
    return test_dataloader, Q
    
def worker(config, dataset, keyframe):
    """
        :param dataset: choices['dataset_8', 'dataset_9']
    """
    idx = int(mp.current_process().name.split('-')[-1])
    # idx = 1
    device = torch.device("cuda",config.scared_test.cuda[idx-1])
    print(f"Processing {os.getpid()} is using GPU {device}")
    model = fetch_model(config.model_config, device)
    # load parameters
    print("loading model {}".format(config.scared_test.loadckpt))
    state_dict = torch.load(config.scared_test.loadckpt, map_location=device)
    try:
        model.load_state_dict(state_dict['model'])
    except RuntimeError:
        state_dict = module_remove(state_dict["model"])
        model.load_state_dict(state_dict)
    save_folder = osp.join(config.scared_test.savedir,dataset,keyframe)
    test_dataloader, Q = fetch_dataloader(config.dataset_config.testSet, dataset, keyframe)
    os.makedirs(save_folder, exist_ok=True)
    tbar = tqdm(total=len(test_dataloader), position=idx-1, desc=f"{dataset}/{keyframe}")
    test_result = test_on_dataset(config.scared_test, test_dataloader, model, tbar, Q, save_folder, device)
    output = {
        dataset:{keyframe:test_result}
    }
    return output


def module_remove(state_dict):
    from collections import OrderedDict  
    nsd = OrderedDict()
    for k, v in state_dict.items():
        nsd[k.replace("module.", "")] = v
    return nsd

def concat_result_dicts(results):
    output = {
        "dataset_8":{},
        "dataset_9":{}
    }
    for result in results:
        for ds, kf_dict in result.items():
            output[ds].update(kf_dict)
    return output

@torch.no_grad()
def inference_time(config):
    import time
    from tqdm import tqdm
    from thop import profile
    device = "cuda:0"
    model = fetch_model(config.model_config, device)
    
    # load parameters
    print("loading model {}".format(config.scared_test.loadckpt))
    state_dict = torch.load(config.scared_test.loadckpt, map_location=device)
    try:
        model.load_state_dict(state_dict['model'])
    except RuntimeError:
        state_dict = module_remove(state_dict["model"])
        model.load_state_dict(state_dict)
    test_dataloader, Q = fetch_dataloader(config.dataset_config.testSet, 'dataset_8', 'keyframe_0')
    i = 0
    timelist = []
    model.eval()
    for batch_idx, sample in tqdm(enumerate(test_dataloader), total=100):
        if batch_idx==101:
            break
        imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disp']
        imgL = imgL.to(device)
        imgR = imgR.to(device)
        start_time = time.time()  # 记录开始时间
        disp_est = model(imgL, imgR)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算经过时间
        timelist.append(elapsed_time)
        i += 1
    # 去除第一项，启动的时间
    timelist.pop(0)
    print("len timelist:", len(timelist))
    print(timelist[:10])
    flops, params = profile(model=model, inputs=(imgL, imgR), verbose=False)
    print(f"Total number of parameters: {params / 1e6:.4f} M")
    print(f"Total FLOPs: {flops / 1e12:.4f} T")
    print(f"Total samples {i-1}, total time={np.sum(timelist)}s, mean time={np.mean(timelist)*1000}")


# def run(config):
#     param_queue = [(config, "dataset_8", "keyframe_0"),(config, "dataset_8", "keyframe_1"),
#                    (config, "dataset_8", "keyframe_2"),(config, "dataset_8", "keyframe_3"),
#                    (config, "dataset_8", "keyframe_4"),(config, "dataset_9", "keyframe_0"),
#                    (config, "dataset_9", "keyframe_1"),(config, "dataset_9", "keyframe_2"),
#                    (config, "dataset_9", "keyframe_3"),(config, "dataset_9", "keyframe_4")
#                    ]
#     with mp.Pool(processes=config.scared_test.workers) as pool:
#         results = pool.starmap(worker, param_queue)
#     results = concat_result_dicts(results)
#     print(results)
#     with open(osp.join(config.scared_test.savedir, 'results.yaml'), mode='w') as wf:
#         yaml.dump(results, wf)
@torch.no_grad()
def viz_all_outputs(config, savepath):
    device = torch.device("cuda",0) 
    imgl = "/home/dingyang/data/scaredDisp/dataset_8/keyframe_4/data/left_rectified/000000.png"
    imgr = "/home/dingyang/data/scaredDisp/dataset_8/keyframe_4/data/right_rectified/000000.png" 
    displ = "/home/dingyang/data/scaredDisp/dataset_8/keyframe_4/data/disparity/000000.tiff"
    imgL = F.to_tensor(rgb_reader(imgl))
    imgR = F.to_tensor(rgb_reader(imgr))
    disp_gt = F.to_tensor(tiff_reader(displ))
    model = fetch_model(config.model_config, device)
    # load parameters
    print("loading model {}".format(config.scared_test.loadckpt))
    state_dict = torch.load(config.scared_test.loadckpt, map_location=device)
    try:
        model.load_state_dict(state_dict['model'])
    except RuntimeError:
        state_dict = module_remove(state_dict["model"])
        model.load_state_dict(state_dict)
    model.train()
    imgL = imgL.to(device).unsqueeze(0)
    imgR = imgR.to(device).unsqueeze(0)
    disp_ests = model(imgL, imgR)
    mask = (disp_gt > 0) & (disp_gt < config.exp_config.maxdisp)

    image_outputs = {
        "depth_est0": img_rainbow_func.apply(disp_ests[0],mask, "disp"),
        "depth_est1": img_rainbow_func.apply(disp_ests[1],mask, "disp"),
        "depth_est2": img_rainbow_func.apply(disp_ests[2],mask, "disp"),
        "depth_gt": img_rainbow_func.apply(disp_gt,mask, "disp"),
    }
    save_images_local(savepath, image_outputs)    
