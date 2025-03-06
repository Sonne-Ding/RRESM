# --------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 12/12/2024 12:14
# @Author  : Ding Yang
# @Project : OpenMedStereo
# @Device  : Moss
# --------------------------------------
from __future__ import print_function
import torchvision.utils as vutils
import numpy as np
import torch
import torch.utils.data
from torch.autograd import Function
from tools.data_convert import tensor2numpy
from matplotlib import pyplot as plt
from PIL import Image
from os import path as osp
import os


def save_images_local(path, images_dict):
    os.makedirs(path, exist_ok=True)
    images_dict = tensor2numpy(images_dict)
    for tag, values in images_dict.items():
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = [values]
        for idx, value in enumerate(values):
            if len(value.shape) == 3:
                value = value[:, np.newaxis, :, :]
            value = value[:1]
            value = torch.from_numpy(value)
            image_name = '{}/{}.png'.format(path, tag)
            vutils.save_image(value, image_name, padding=0, nrow=1, normalize=True, scale_each=True)


class ColormapPainter:
    def __init__(self, colormap_path, max_clip):
        self.colormap = np.load(colormap_path).astype(np.float32)
        self.maxClip = max_clip
        self.stages = self.colormap.shape[0]

    def paint(self, img):
        clipped_img = np.clip(img, a_min=0, a_max=self.maxClip)
        color_index_map = ((self.stages - 1) * (clipped_img / self.maxClip)).astype(np.int32)
        colored_img = self.colormap[color_index_map]
        return colored_img

class img_only_rainbow_func(Function):
    @staticmethod
    def forward(self, img, map_type):
        if map_type == "disp":
            colorpainter = ColormapPainter("tools/rainbow_colormap.npy", 98.0)
        elif map_type == "depth":
            colorpainter = ColormapPainter("tools/rainbow_colormap.npy", 116.0)
        else:
            colorpainter = ColormapPainter("tools/rainbow_colormap.npy", 100.0)
        img_np = img.detach().cpu().numpy()
        color_image = colorpainter.paint(img_np)
        return torch.from_numpy(np.ascontiguousarray(color_image.transpose([0, 3, 1, 2])))

    @staticmethod
    def backward(self, grad_output):
        return None

class img_rainbow_func(Function):
    @staticmethod
    def forward(self, img, mask, map_type):
        if map_type == "disp":
            colorpainter = ColormapPainter("tools/rainbow_colormap.npy", 98.0)
        elif map_type == "depth":
            colorpainter = ColormapPainter("tools/rainbow_colormap.npy", 116.0)
        else:
            colorpainter = ColormapPainter("tools/rainbow_colormap.npy", 100.0)
        img_np = img.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()
        img_np[np.logical_not(mask_np)] = 0.0
        color_image = colorpainter.paint(img_np)
        return torch.from_numpy(np.ascontiguousarray(color_image.transpose([0, 3, 1, 2])))

    @staticmethod
    def backward(self, grad_output):
        return None


class errormap_rainbow_func(Function):
    @staticmethod
    def forward(self, D_est_tensor, D_gt_tensor, mask, error_map_type):
        """
            :type D_est_tensor: torch.Tensor
            :type D_gt_tensor: torch.Tensor

            :param error_map_type: choice [disp, rel, depth]
            :return: Error map
            :rtype: torch.Tensor, on cpu
        """
        D_gt_np = D_gt_tensor.detach().cpu().numpy()
        D_est_np = D_est_tensor.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        error = np.abs(D_gt_np - D_est_np)
        if error_map_type == 'rel':
            error /= (D_gt_np + 1e-6)
            colorpainter = ColormapPainter("tools/rainbow_colormap2.npy", 0.1)
        elif error_map_type == "abs":
            colorpainter = ColormapPainter("tools/rainbow_colormap2.npy", 8.0)
        else:
            colorpainter = ColormapPainter("tools/rainbow_colormap2.npy", 10.0)
        error[np.logical_not(mask)] = 0
        error_image = colorpainter.paint(error)
        return torch.from_numpy(np.ascontiguousarray(error_image.transpose([0, 3, 1, 2])))

    @staticmethod
    def backward(self, grad_output):
        return None


def plot_dataset_examples(result_root):
    datasets = ['dataset_8', 'dataset_9']
    keyframes = ['keyframe_2', 'keyframe_3', 'keyframe_4']
    for dataset in datasets:
        fig, axs = plt.subplots(3, 5, figsize=(15, 9))
        for ax in axs.flat:
            ax.axis("off")
        imgl_list = []
        pred_list = []
        gt_list = []
        abs_list = []
        rel_list = []
        for kf in keyframes:
            imgl_list.append(np.array(Image.open(osp.join(result_root, dataset, kf, '000000/imgL.png'))))
            pred_list.append(np.array(Image.open(osp.join(result_root, dataset, kf, '000000/depth_est.png'))))
            gt_list.append(np.array(Image.open(osp.join(result_root, dataset, kf, '000000/depth_gt.png'))))
            abs_list.append(np.array(Image.open(osp.join(result_root, dataset, kf, '000000/abs_errormap.png'))))
            rel_list.append(np.array(Image.open(osp.join(result_root, dataset, kf, '000000/rel_errormap.png'))))
        for i in range(3):
            axs[i, 0].imshow(imgl_list[i])
            axs[i, 1].imshow(gt_list[i])
            axs[i, 2].imshow(pred_list[i])
            axs[i, 3].imshow(abs_list[i])
            axs[i, 4].imshow(rel_list[i])
        axs[0, 0].set_title("Img L")
        axs[0, 1].set_title("Depth Gt")
        axs[0, 2].set_title("Depth est")
        axs[0, 3].set_title("Error Abs")
        axs[0, 4].set_title("Error Rel")
        fig.tight_layout()
        fig.savefig(osp.join(result_root, f"{dataset}_examples.png"))


if __name__ == "__main__":
    pass
