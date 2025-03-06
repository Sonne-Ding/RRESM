# --------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 12/12/2024 13:14
# @Author  : Ding Yang
# @Project : OpenMedStereo
# @Device  : Moss
# --------------------------------------
import torch
import numpy as np
from collections import namedtuple

def yaml2toml(yaml_fp, toml_fp):
    """

    :param yaml_fp: the path of the yaml file
    :type yaml_fp: str
    :param toml_fp: the path of the toml file
    :type toml_fp: str
    """
    import yaml
    import tomlkit
    with open(yaml_fp, mode='r') as rf:
        config = yaml.safe_load(rf)
    print(config)
    with open(toml_fp, mode='w') as wf:
        tomlkit.dump(config, wf)  
    print("Convert yaml to toml successfully, saved in {}".format(toml_fp))


def dict_to_namedtuple(data, typename):
    """

    :param data: dict to convert to namedtuple
    :type data: dict only
    :param typename: name of the namedtuple
    :type typename: str
    :return: namedtuple
    :rtype: namedtuple
    """
    if isinstance(data, dict):
        fields = data.keys()
        values = [dict_to_namedtuple(data[field], field.capitalize()) for field in fields]
        return namedtuple(typename, fields)(*values)
    elif isinstance(data, list):
        return [dict_to_namedtuple(d, typename) for d in data]
    return data



def make_iterative_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper

@make_iterative_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type for tensor2float")

@make_iterative_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.cpu().numpy()
    else:
        raise NotImplementedError("invalid input type for tensor2numpy")

@make_iterative_func
def check_allfloat(vars):
    assert isinstance(vars, float)