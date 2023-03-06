import pandas as pd
import numpy as np
import os

ind2name = {47: 'ctl_cifar100_wo_connect_only_ance_fs512_512_sp03_more_data_Mar3',
            48: 'ctl_cifar100_wo_connect_only_ance_fs512_256_sp03_more_data_Mar3',
            49: 'ctl_cifar100_wo_connect_only_ance_fs512_128_sp03_more_data_Mar3',
            50: 'ctl_cifar100_wo_connect_only_ance_fs512_64_sp03_more_data_Mar3',
            51: 'ctl_cifar100_wo_connect_full_fs512_512_sp03_more_data_Mar3',
            52: 'ctl_cifar100_wo_connect_full_fs512_256_sp03_more_data_Mar3',
            53: 'ctl_cifar100_wo_connect_full_fs512_128_sp03_more_data_Mar3',
            54: 'ctl_cifar100_wo_connect_full_fs512_64_sp03_more_data_Mar3',
            55: 'ctl_cifar100_connect_fs512_512_sp03_more_data_Mar3',
            56: 'ctl_cifar100_connect_fs512_256_sp03_more_data_Mar3',
            57: 'ctl_cifar100_connect_fs512_128_sp03_more_data_Mar3',
            58: 'ctl_cifar100_connect_fs512_64_sp03_more_data_Mar3',
            59: 'ctl_cifar100_connect_fs256_256_sp03_more_data_Mar3',
            60: 'ctl_cifar100_connect_fs256_128_sp03_more_data_Mar3',
            61: 'ctl_cifar100_connect_fs256_64_sp03_more_data_Mar3',
            62: 'ctl_cifar100_wo_connect_only_ance_random_order_fs512_512_sp03_more_data_Mar3',
            63: 'ctl_cifar100_wo_connect_full_random_order_fs512_512_sp03_more_data_Mar3',
            64: 'ctl_imagenet100_wo_connect_only_ance_fs512_512_512_more_data_Mar3',
            65: 'ctl_imagenet100_wo_connect_only_ance_fs512_256_128_more_data_Mar3',
            66: 'ctl_imagenet100_wo_connect_only_ance_fs512_128_64_more_data_Mar3',
            67: 'ctl_imagenet100_wo_connect_only_ance_fs512_256_64_more_data_Mar3',
            68: 'ctl_imagenet100_wo_connect_full_fs512_512_512_more_data_Mar3',
            69: 'ctl_imagenet100_wo_connect_full_fs512_256_128_more_data_Mar3',
            70: 'ctl_imagenet100_wo_connect_full_fs512_128_64_more_data_Mar3',
            71: 'ctl_imagenet100_wo_connect_full_fs512_256_64_more_data_Mar3',
            72: 'ctl_imagenet100_connect_fs512_512_512_more_data_Mar3',
            73: 'ctl_imagenet100_connect_fs512_256_128_more_data_Mar3',
            74: 'ctl_imagenet100_connect_fs512_128_64_more_data_Mar3',
            75: 'ctl_imagenet100_connect_fs512_256_64_more_data_Mar3',
            76: 'ctl_imagenet100_wo_connect_only_ance_random_order_fs512_512_512_more_data_Mar3',
            77: 'ctl_imagenet100_wo_connect_full_random_order_fs512_512_512_more_data_Mar3',
            78: 'ctl_cifar100_der_baseline_fs512_512_sp03_more_data_Mar3',
            79: 'ctl_cifar100_der_baseline_fs512_256_sp03_more_data_Mar3',
            80: 'ctl_cifar100_der_baseline_fs512_128_sp03_more_data_Mar3',
            81: 'ctl_cifar100_der_baseline_fs512_64_sp03_more_data_Mar3',
            82: 'ctl_cifar100_der_baseline_fs256_256_sp03_more_data_Mar3',
            83: 'ctl_cifar100_der_baseline_fs256_128_sp03_more_data_Mar3',
            84: 'ctl_cifar100_der_baseline_fs256_64_sp03_more_data_Mar3',
            85: 'ctl_imagenet100_der_baseline_fs512_512_512_more_data_Mar3',
            86: 'ctl_imagenet100_der_baseline_fs512_256_128_more_data_Mar3',
            87: 'ctl_imagenet100_der_baseline_fs512_128_64_more_data_Mar3',
            88: 'ctl_imagenet100_der_baseline_fs512_256_64_more_data_Mar3',
            89: 'ctl_cifar100_der_baseline_random_order_fs512_512_sp03_more_data_Mar3',
            90: 'ctl_imagenet100_der_baseline_random_order_fs512_512_512_more_data_Mar3',
            }

def load_logs(log_path, restart=False):
    if not os.path.exists(log_path):
        return None
    with open(log_path, 'r') as f:
        info = f.readlines()

    info_filtered = []
    record_list = []
    for i in info:
        if i != '\n' and i != '':
            info_filtered.append(i)
            if 'Evaluation eval_after_decouple' in i:
                finest_info = i.split('finest avg: ')[1].split(' classes, coarse avg:')[0].split(' with ')
                if int(finest_info[1]) != 0:
                    record_list.append(float(finest_info[0]))
    if restart:
        record_list = record_list[1:]
    return record_list


res_dict = {}
model_name_list = []
final_acc_list = []
incremental_acc_list = []
model_setting_list = []
fs_list = []
dataset_list = []
date_list = []
curr_task_list = []

for job_ind in ind2name:
    model_name_i = ind2name[job_ind]
    model_name_split = model_name_i.split('_')
    dataset = model_name_split[1]
    fs_split = model_name_i.split('_fs')
    model_setting = '_'.join(fs_split[0].split('_')[2:])

    if dataset == 'cifar100':
        fs = [int(i) for i in fs_split[1].split('_')[:2]]
        addition_info = fs_split[1].split('_')[2:-1]
    else:
        fs = [int(i) for i in fs_split[1].split('_')[:3]]
        addition_info = fs_split[1].split('_')[3:-1]
    date = model_name_split[-1]
    model_setting_list.append(model_setting)
    fs_list.append(fs)
    dataset_list.append(dataset)
    date_list.append(date)


    log_path = f'/datasets/{dataset}_results/{model_name_i}/train/logs'
    # log_path = f'/Users/chenyuzhao/Downloads/train_Mar3.log'

    record_list = load_logs(f'{log_path}/train.log')
    while len(record_list) != 20:
        record_list_tmp = load_logs(f'{log_path}/train_retrain_retrain_from_step{len(record_list)}.log', True)
        if record_list_tmp:
            record_list.extend(record_list_tmp)
        else:
            break

    model_name_list.append(model_name_i)

    if len(record_list) == 20:

        final_acc_list.append(record_list[-1])
        incremental_acc_list.append(np.round(np.mean(record_list), 3))
        curr_task_list.append('finish')
    else:
        curr_task_list.append(len(record_list))
        final_acc_list.append(-1)
        incremental_acc_list.append(-1)

df = pd.DataFrame(
    {'model name': model_name_list, 'dataset': dataset_list, 'model_setting': model_setting_list, 'fs': fs_list,
     'final acc': final_acc_list, 'incremental acc': incremental_acc_list, 'date': date_list,
     'curr_task': curr_task_list}, index=list(ind2name.keys()))
# df.to_csv('/Users/chenyuzhao/Downloads/ctl_acc_summary_Mar3.csv')
df.to_csv('ctl_acc_summary_Mar3.csv')
