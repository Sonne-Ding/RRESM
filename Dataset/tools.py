# --------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 9/4/2024 20:31
# @Author  : Ding Yang
# @Project : OpenMedStereo
# @Device  : Moss
# --------------------------------------
from Dataset import data_transform
from Dataset import img_reader

def build_transform_by_cfg(transform_config):
    """
        :param transform_config: config.dataset_config.**Set.transform
    """
    transform_compose = []
    for cur_cfg in transform_config:
        cur_augmentor = getattr(data_transform, cur_cfg.name)(config=cur_cfg)
        transform_compose.append(cur_augmentor)
    return data_transform.Compose(transform_compose)

def build_reader(reader_type):
    """
        :param reader_type: Options ["rgb_reader", "tiff_reader"]
    """
    reader = getattr(img_reader, reader_type)
    return reader