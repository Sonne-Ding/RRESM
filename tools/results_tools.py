import yaml
import csv
from os import path as osp
import statistics

def yaml2rst(yaml_path):
    with open(yaml_path, 'r') as rf:
        data = yaml.safe_load(rf)
    # 计算均值
    # means = {metric: sum(values) / len(values) for metric, values in data.items()}
    # 生成.rst表格
    rst_table = """
+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
|        |                 dataset8                   |                 dataset9                   |        |
+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
| Metric |  K0    | K1     | K2     | K3     | K4     | K0     | K1     | K2     | K3     | K4     |  Mean  |
+========+========+========+========+========+========+========+========+========+========+========+========+
"""
    # 遍历指标和关键帧，填充表格数据
    # 提取数据并计算均值
    metrics = ['D1', 'EPE', 'Thres1', 'Thres2', 'Thres3']
    for metric in metrics:
        values_dataset_8 = [data['dataset_8'][f'keyframe_{i}'][metric] for i in range(5)]
        values_dataset_9 = [data['dataset_9'][f'keyframe_{i}'][metric] for i in range(5)]
        mean_value = sum(values_dataset_8 + values_dataset_9) / 10
        rst_table += f"|{metric:8}|{values_dataset_8[0]:8.4f}|{values_dataset_8[1]:8.4f}|{values_dataset_8[2]:8.4f}|{values_dataset_8[3]:8.4f}|{values_dataset_8[4]:8.4f}|{values_dataset_9[0]:8.4f}|{values_dataset_9[1]:8.4f}|{values_dataset_9[2]:8.4f}|{values_dataset_9[3]:8.4f}|{values_dataset_9[4]:8.4f}|{mean_value:8.4f}|\n"
        rst_table += "+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+\n"

    # 保存RST表格到文件
    with open(osp.join(osp.dirname(yaml_path),'data_table.rst'), 'w') as file:
        file.write(rst_table)

def yaml2csv(yaml_path, save_name=None):
    # 将YAML内容加载为Python字典
    with open(yaml_path, 'r') as rf:
        data = yaml.safe_load(rf)
    # CSV文件的列名
    with open(osp.join(osp.dirname(yaml_path),'results.csv' if save_name is None else save_name), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow([ '','dataset8', '', '', '', '', 'dataset9', '', '', ''])
        writer.writerow(['Metric', 'K0', 'K1', 'K2', 'K3', 'K4', 'K0', 'K1', 'K2', 'K3', 'K4', 'mean'])
        # 写入数据
        metrics = ['EPE', 'D1', 'Thres1', 'Thres2', 'Thres3']
        for metric in metrics:
            # 提取dataset_8和dataset_9的值
            values_8 = [data['dataset_8']['keyframe_' + str(i)].get(metric) for i in range(5)]
            values_9 = [data['dataset_9']['keyframe_' + str(i)].get(metric) for i in range(5)]
            # 计算均值
            mean_total = statistics.mean(values_8 + values_9)
            formatted_mean_total = f"{mean_total:.4g}"
            # 保留四位有效数字
            formatted_values_8 = [f"{value:.4g}" for value in values_8]
            formatted_values_9 = [f"{value:.4g}" for value in values_9]
            # 写入CSV
            writer.writerow([metric] + formatted_values_8 + formatted_values_9 + [formatted_mean_total])
    print("CSV文件已生成。")