from Evaluators.scared_evaluator import worker,inference_time, concat_result_dicts,viz_all_outputs

# from Evaluators.msdesis_evaluator import worker, viz_all_outputs,inference_time,concat_result_dicts
from os import path as osp 
import multiprocessing as mp
from tools.exp_container import ConfigDataContainer
import yaml
from tools.results_tools import yaml2csv
import wandb

def wandb_log_result(proj, expname, data_dict):
    run = wandb.init(
        project=proj,
        name=expname,
    )
    columns = ["dataset_keyframe", "D1", "EPE", "Thres1", "Thres2", "Thres3"]
    my_table = wandb.Table(columns=columns)
    sums = {col: 0 for col in columns[1:]}
    count = 0
    # 遍历字典并添加数据到表格
    for dataset, keyframes in data_dict.items():
        for keyframe, metrics in keyframes.items():
            # 构造自定义索引
            custom_index = f"{dataset}_{keyframe}"
            # 构造一行数据
            row_data = [custom_index] + [metrics[col] for col in columns[1:]]
            # 添加到表格
            my_table.add_data(*row_data)
            # 累加用于计算均值
            for col in columns[1:]:
                sums[col] += metrics[col]
            count += 1
    # 计算均值
    means = {col: sums[col] / count for col in columns[1:]}
    # 添加均值行到表格
    mean_row = ["Mean"] + [means[col] for col in columns[1:]]
    my_table.add_data(*mean_row)
    # 记录表格到Wandb
    run.log({"Results on SCARED": my_table})
    run.finish()


if __name__=="__main__":
    config_file = "configs/GwcNet/gwcdynet_abl11.yaml"
    # config_file = "configs/GwcNet/ms_eval.yaml"
    with open(config_file, mode='r') as rf:
        config = yaml.safe_load(rf)
    config = ConfigDataContainer(**config)
    # inference_time(config)
    viz_all_outputs(config,"results/ablaton11/viz_folder")
    # param_queue = [(config, "dataset_8", "keyframe_1"),]
    # # param_queue = [(config, "dataset_8", "keyframe_0"),(config, "dataset_8", "keyframe_1"),
    # #                (config, "dataset_8", "keyframe_2"),(config, "dataset_8", "keyframe_3"),
    # #                (config, "dataset_8", "keyframe_4"),(config, "dataset_9", "keyframe_0"),
    # #                (config, "dataset_9", "keyframe_1"),(config, "dataset_9", "keyframe_2"),
    # #                (config, "dataset_9", "keyframe_3"),(config, "dataset_9", "keyframe_4")
    # #                ]
    # # worker(config, "dataset_8", "keyframe_0")
    # with mp.Pool(processes=config.scared_test.workers) as pool:
    #     results = pool.starmap(worker, param_queue)

    # results = concat_result_dicts(results)
    # # wandb_log_result("GwcNet", config.scared_test.expname, results)
    # with open(osp.join(config.scared_test.savedir, 'best_epe.yaml'), mode='w') as wf:
    #     yaml.dump(results, wf)
    # yaml2csv(osp.join(config.scared_test.savedir, 'best_epe.yaml')) 