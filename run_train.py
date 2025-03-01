from Trainers.scared_trainer import worker
import torch
import yaml
# import tomlkit
from torch import multiprocessing as mp
import os
from tools.exp_container import ConfigDataContainer
import os

def load_config(config_fp):
    with open(config_fp, mode='r') as rf:
        config = yaml.safe_load(rf)
    logdir = config['exp_config']['logdir']
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "config.yaml"), mode='w') as wf:
        yaml.dump(config, wf)
    config = ConfigDataContainer(**config)
    return config

if __name__=="__main__":
    config_file = "configs/GwcNet/gwcdynet_abl20.yaml"
    config = load_config(config_file)
    # flag==True，终止进程，用于解决单个进程异常，但是其他进程正常时，无法正常退出的情况
    flag = torch.tensor([False], dtype=torch.bool).share_memory_()
    mp.spawn(
        worker,
        args=(config,flag,),
        nprocs=config.exp_config.world_size,
    )
    