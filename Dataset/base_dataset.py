# --------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 7/28/2024 15:32
# @Author  : Ding Yang
# @Project : OpenMedStereo
# @Device  : Moss
# --------------------------------------
from torch.utils.data import Dataset
from os import path as osp
import yaml
import numpy as np
import random
import json
from Dataset.tools import build_reader, build_transform_by_cfg

class ScaredTrainBase(Dataset):
    def __init__(self, config, mode='train'):
        """
        
        :param config: config.dataset_config
        :param mode: str in ['train', 'val']
        """
        if mode=="train":
            self.mode = "train"
        elif mode=="val":
            self.mode = "val"
        else:
            raise KeyError("Wrong mode in ScaredTrainBase.")
        self.config = config
        self.transforms = build_transform_by_cfg(self.config.transform)
        self.img_reader = build_reader(self.config.imgReader)
        self.disp_reader = build_reader(self.config.dispReader)
        self.dataset = {}
        self.load_items(self.config.root, self.config.catalog)

    def load_items(self, data_root, data_split_file):
        """
        :param data_root:
        :param data_split_file:{train:{img_l, img_r, disp_l}, test, val}
        :return:
        """
        with open(data_split_file, mode='r') as rf:
            datasets = yaml.safe_load(rf)
        tag_set = datasets.get(self.mode)
        assert tag_set is not None, f"{self.mode} set is none!"
        self.dataset['left'] = [osp.join(data_root, p) for p in tag_set['img_l']]
        self.dataset['right'] = [osp.join(data_root, p) for p in tag_set['img_r']]
        self.dataset['disp'] = [osp.join(data_root, p) for p in tag_set['disparity']]
    
    def __getitem__(self, index):
        sample = {}
        sample["left"] = self.img_reader(self.dataset["left"][index])
        sample["right"] = self.img_reader(self.dataset["right"][index])
        sample["disp"] = self.disp_reader(self.dataset["disp"][index])
        sample = self.transforms(sample)
        return sample
    
    def __len__(self):
        return len(self.dataset['disp'])


class ScaredDatasetTest(Dataset):
    def __init__(self, config, ds, kf):
        """
        
        :param config: config.dataset_config
        :param mode: str in ['train', 'val']
        """
        self.mode = "test"
        self.config = config
        self.kf = kf
        self.ds = ds
        self.transforms = build_transform_by_cfg(self.config.transform)
        self.img_reader = build_reader(self.config.imgReader)
        self.disp_reader = build_reader(self.config.dispReader)
        self.dataset = {}
        self.load_items(self.config.root, self.config.catalog)
    
    def get_Q(self):
        with open(osp.join(self.config.root, self.ds, self.kf, 'stereo_calib.json'), mode='r') as rf:
            calib = json.load(rf)
        Q = np.array(calib['Q']['data'], dtype=np.float64).reshape(4,4)
        return Q

    def load_items(self, data_root, data_split_file):
        """
        :param data_root:
        :param data_split_file:{train:{img_l, img_r, disp_l}, test, val}
        :return:
        """
        with open(data_split_file, mode='r') as rf:
            datasets = yaml.safe_load(rf)
        tag_set = datasets[self.ds][self.kf]
        assert tag_set is not None, f"{self.mode} set is none!"
        self.dataset['left'] = [osp.join(data_root, p) for p in tag_set['img_l']]
        self.dataset['right'] = [osp.join(data_root, p) for p in tag_set['img_r']]
        self.dataset['disp'] = [osp.join(data_root, p) for p in tag_set['disp_l']]
    
    def __getitem__(self, index):
        sample = {}
        sample["left"] = self.img_reader(self.dataset["left"][index])
        sample["right"] = self.img_reader(self.dataset["right"][index])
        sample["disp"] = self.disp_reader(self.dataset["disp"][index])
        sample["disp_filename"] = self.dataset['disp'][index]
        sample = self.transforms(sample)
        return sample
    def __len__(self):
        return len(self.dataset['disp'])
  
    