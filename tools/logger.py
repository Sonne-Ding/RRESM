import logging
from os import path as osp

class BaseLogger:
    def __init__(self, root):
        level = logging.INFO
        self.logger = logging.getLogger(osp.basename(root))
        self.logger.setLevel(level)
        fmt = logging.Formatter("[%(asctime)s %(filename)s:%(lineno)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        f_handler = logging.FileHandler(osp.join(root, 'training.log'), 'a')
        s_handler = logging.StreamHandler()
        f_handler.setFormatter(fmt)
        s_handler.setFormatter(fmt)
        self.logger.addHandler(f_handler)  
        self.logger.addHandler(s_handler)  

    def info(self, msg):
        self.logger.info(msg)
    
