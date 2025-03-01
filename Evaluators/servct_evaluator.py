import timm
import yaml
import torch
from torch import nn
from Dataset.img_reader import rgb_reader
from torchvision.transforms import Resize
from Dataset.data_transform import ToTensor
from torch.utils.data import Dataset
from os import path as osp
import os
from Models.msdesis import multitask_light, multitask_resnet32
from tools.metrics import D1_metric, EPE_metric, Thres_metric,AverageMeterDict
from tools.visualization import save_images_local, img_rainbow_func, errormap_rainbow_func
from tools.data_convert import tensor2float
import cv2
from pathlib import Path
import numpy as np
from glob import glob
import errno
def scale_tensor_img(tensor_img, mask):
    tensor_img[torch.logical_not(mask)] = 0
    tensor_img = tensor_img.clip(min=0.0, max=250.0)
    return tensor_img
def load_subpix_png(path, scale_factor=256.0):
    if not Path(path).is_file():
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), str(path))
    disparity = cv2.imread(str(path), -1)
    disparity_type = disparity.dtype
    disparity = disparity.astype(np.float32)
    if disparity_type == np.uint16:
        disparity = disparity / scale_factor
    return disparity

class SERV(Dataset):
    def __init__(self, root="/home/dingyang/data/SERV-CT-disp",expand=False) -> None:
        super().__init__()
        self.expand = expand
        self.left_imgs = sorted(glob(osp.join(root, "left", "*.png")))
        self.right_imgs = sorted(glob(osp.join(root, "right", "*.png")))
        self.disp_imgs = sorted(glob(osp.join(root, "disparity", "*.png")))
        self.img_reader = rgb_reader
        self.disp_reader = load_subpix_png
        if expand:
            self.rz = Resize((576,736))
        self.transforms = ToTensor("hello")
    
    def __getitem__(self, index):
        sample = {}
        sample["left"] = self.img_reader(self.left_imgs[index])
        sample["right"] = self.img_reader(self.right_imgs[index])
        sample["disp"] = self.disp_reader(self.disp_imgs[index])
        sample["disp_filename"] = self.disp_imgs[index]
        sample = self.transforms(sample)
        if self.expand:
            sample['left'] = self.rz(sample['left'])
            sample['right'] = self.rz(sample['right'])
            sample['disp'] = self.rz(sample['disp'])
        return sample

    def __len__(self,):
        return len(self.disp_imgs)
    
def module_remove(state_dict):
    from collections import OrderedDict  
    nsd = OrderedDict()
    for k, v in state_dict.items():
        nsd[k.replace("module.", "")] = v
    return nsd
def fetch_model_ms(config, device):
    if config.model_config.model=="light":
        model = multitask_light(320, 1, 'disparity')
        ckp = torch.load("checkpoints/msdesis_light.pt")
        model.load_state_dict(ckp)
    elif config.model_config.model=="resnet":
        model = multitask_resnet32(320, 1, 'disparity')
        ckp = torch.load("checkpoints/msdesis_resnet34.pt")
        model.load_state_dict(ckp)
    else:
        raise NameError("Wrong model type")
    model = model.to(device)
    return model
def fetch_model(config, device):
    model = timm.create_model(config.model_config.model,**config.model_config)
    if config.model_config.syncBN:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    state_dict = torch.load(config.scared_test.loadckpt, map_location=device)
    try:
        model.load_state_dict(state_dict['model'])
    except RuntimeError:
        state_dict = module_remove(state_dict["model"])
        model.load_state_dict(state_dict)
    return model

@torch.no_grad()
def test_sample(model, sample, device, maxdisp=192.0):
    """
        :return: scalar_outputs, image_outputs
    """
    imgL, imgR, disp_gt =  sample['left'], sample['right'], sample['disp']
    imgL = imgL.to(device).unsqueeze(0)
    imgR = imgR.to(device).unsqueeze(0)
    disp_gt = disp_gt.unsqueeze(0).squeeze(1).to(device)
    disp_est = model(imgL, imgR)[-1]
    # print(disp_est.shape)
    mask = (disp_gt > 0) & (disp_gt<maxdisp)
    depth_est = disp_est
    depth_gt = disp_gt 
    scalar_outputs = {}
    scalar_outputs["D1"] = D1_metric(depth_est, depth_gt, mask)
    scalar_outputs["EPE"] = EPE_metric(depth_est, depth_gt, mask)
    scalar_outputs["Thres1"] = Thres_metric(depth_est, depth_gt, mask, 1.0)
    scalar_outputs["Thres2"] = Thres_metric(depth_est, depth_gt, mask, 2.0)
    scalar_outputs["Thres3"] = Thres_metric(depth_est, depth_gt, mask, 3.0)
    depth_est = scale_tensor_img(depth_est, mask)
    depth_gt = scale_tensor_img(depth_gt, mask)
    image_outputs = {
        "depth_est": img_rainbow_func.apply(depth_est,mask, "disp"),
        "depth_gt": img_rainbow_func.apply(depth_gt,mask, "disp"),
        "abs_errormap": errormap_rainbow_func.apply(depth_est, depth_gt, mask, 'disp'),
        "rel_errormap": errormap_rainbow_func.apply(depth_est, depth_gt, mask, 'rel'),
        "imgL": imgL,
        "imgR": imgR
        }
    return tensor2float(scalar_outputs), image_outputs

def test_on_dataset(dataset,model ,savepath, device):
    """
        :param dataset: Choices['dataset_8', 'dataset_9']
    """
    model.eval()
    avg_test_scalars = AverageMeterDict()
    for i in range(len(dataset)):
        print(i)
        sample = dataset[i]
        scalar_outputs, image_outputs = test_sample(model, sample, device)
        print(scalar_outputs['EPE'])
        if scalar_outputs['EPE'] != 0:
            avg_test_scalars.update(scalar_outputs)
        # else:
        #     print(f"Abandon sample {batch_idx}")
        sample_name = osp.basename(sample['disp_filename']).split('.')[0]
        img_savepath = osp.join(savepath, sample_name)
        save_images_local(img_savepath, image_outputs)
    avg_test_scalars = avg_test_scalars.mean()
    return avg_test_scalars
        

def worker(config):
    device = torch.device("cuda", 0)
    # model = fetch_model(config, device)
    model = fetch_model_ms(config, device)
    dataset = SERV(expand=True)
    test_results = test_on_dataset(dataset,model, "results/serv_ms_resnet",device)
    with open(osp.join("results/serv_ms_resnet", 'best_epe.yaml'), mode='w') as wf:
        yaml.dump(test_results, wf)
        print("saved ")

