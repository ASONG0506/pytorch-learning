# -*-coding:utf-8 -*-
# --------------------
# author: cjs
# time: 2019-11-12
# usage: parse the cmd line configurations
# --------------------
def parse_model_config(path):
    """
    从模型定义的文件中读取配置参数，从而构建模型
    :param path:
    :return:
    """
    file = open(path, "r")
    lines = file.read().split("\n")
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.strip().lstrip() for x in lines] # 去除掉左边的缩进
    module_defs = []

    for line in lines:
        if line.startswith('['):
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.rstrip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


def parse_data_config(path):
    """
    读取并解析数据的配置
    :param path:
    :return:
    """
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == "" or line.startswith("#"):
            continue
        key, value = line.split("=")
        options[key.strip()] = value.strip()

    return options