import pandas as pd
import numpy as np
import os

ind2name_2 = {1: 'ctl_imagenet100_wo_connect_only_ance_3layer_DFS_fs512_512_512_Mar11',
              2: 'ctl_imagenet100_wo_connect_only_ance_2layer_fs512_512_512_Mar11',
              3: 'ctl_imagenet100_wo_connect_only_ance_4layer_fs512_512_512_Mar11',
              4: 'ctl_imagenet100_wo_connect_only_ance_3layers_random_nodes_fs512_512_512_Mar11',
              5: 'ctl_imagenet100_wo_connect_only_ance_3layer_fs512_256_128_Mar11',
              6: 'ctl_imagenet100_wo_connect_only_ance_3layer_fs512_256_64_Mar11',
              7: 'ctl_imagenet100_wo_connect_only_ance_3layer_fs512_128_64_Mar11',
              8: 'ctl_imagenet100_wo_connect_only_prev_3layer_fs512_512_512_Mar11',
              9: 'ctl_imagenet100_wo_connect_only_self_3layer_fs512_512_512_Mar11',
              10: 'ctl_imagenet100_der_baseline_3layer_DFS_fs512_512_512_Mar11',
              11: 'ctl_imagenet100_der_baseline_2layer_fs512_512_512_Mar11',
              12: 'ctl_imagenet100_der_baseline_4layer_fs512_512_512_Mar11',
              13: 'ctl_imagenet100_der_baseline_3layers_random_nodes_fs512_512_512_Mar11',
              14: 'ctl_imagenet100_wo_connect_only_ance_3layer_BFS_fs512_512_512_Mar11',
              15: 'ctl_imagenet100_der_baseline_3layers_random_nodes_change_coarse_fea_fs512_512_512_Mar12',
              16: 'ctl_imagenet100_der_baseline_3layer_DFS_change_coarse_fea_fs512_512_512_Mar12',
              17: 'ctl_imagenet100_der_baseline_3layers_random_nodes_1_change_coarse_fea_fs512_512_512_Mar12',
              18: 'ctl_imagenet100_der_baseline_3layers_random_nodes_2_change_coarse_fea_fs512_512_512_Mar12',
              19: 'ctl_cifar100_wo_connect_only_ance_fs512_512_Mar11',
              20: 'ctl_cifar100_connect_only_ance_fs512_512_Mar11',
              21: 'ctl_imagenet100_wo_connect_only_ance_3layers_random_nodes_1_fs512_512_512_Mar11',
              22: 'ctl_imagenet100_wo_connect_only_ance_3layers_random_nodes_2_fs512_512_512_Mar11',
              23: 'ctl_imagenet100_der_baseline_3layers_random_nodes_1_fs512_512_512_Mar11',
              24: 'ctl_imagenet100_der_baseline_3layers_random_nodes_2_fs512_512_512_Mar11',
              25: 'ctl_cifar100_icarl_wo_connect_only_ance_Mar12', 26: 'ctl_cifar100_icarl_wo_connect_full_Mar12',
              27: 'ctl_cifar100_der_baseline_fs512_512_Mar12', 28: 'ctl_cifar100_der_baseline_fs512_256_Mar12',
              29: 'ctl_cifar100_der_baseline_fs512_128_Mar12', 30: 'ctl_cifar100_der_baseline_fs512_64_Mar12',
              31: 'ctl_cifar100_wo_connect_only_self_with_joint_ce_fs512_512_Mar13',
              32: 'ctl_cifar100_wo_connect_only_self_wo_joint_ce_fs512_512_Mar13',
              33: 'ctl_cifar100_wo_connect_only_prev_with_joint_ce_fs512_512_Mar13',
              34: 'ctl_cifar100_wo_connect_only_prev_wo_joint_ce_fs512_512_Mar13',
              35: 'ctl_cifar100_wo_connect_only_ance_wo_joint_ce_fs512_512_Mar13',
              36: 'ctl_cifar100_icarl_wo_connect_only_ance_with_nloss_Mar13',
              37: 'ctl_cifar100_icarl_wo_connect_full_with_nloss_Mar13',
              38: 'ctl_cifar100_icarl_wo_connect_only_ance_with_nloss_with_fe_init_Mar13',
              39: 'ctl_cifar100_icarl_wo_connect_full_with_nloss_with_fe_init_Mar13',
              40: 'ctl_cifar100_icarl_wo_connect_only_ance_with_fe_init_Mar13',
              41: 'ctl_cifar100_icarl_wo_connect_full_with_fe_init_Mar13',
              42: 'ctl_cifar100_icarl_wo_connect_only_ance_with_nloss_with_fe_init_seed_1993_Mar13',
              43: 'ctl_cifar100_icarl_wo_connect_full_with_nloss_with_fe_init_seed_1993_Mar13',
              44: 'ctl_cifar100_icarl_wo_connect_only_ance_with_nloss_with_fe_init_seed_600_Mar13',
              45: 'ctl_cifar100_icarl_wo_connect_full_with_nloss_with_fe_init_seed_600_Mar13'}
ind2name = {}
for i in ind2name_2:
    if i in list(range(14, 20)):
        ind2name[i] = ind2name_2[i]


# ind2name = {21: 'ctl_imagenet100_wo_connect_only_ance_3layers_random_nodes_1_fs512_512_512_Mar11'}
def load_logs(log_path, restart=False):
    if not os.path.exists(log_path):
        return None, None
    with open(log_path, 'r') as f:
        info = f.readlines()

    info_filtered = []
    record_list = []
    curr_finest_classes = 0
    curr_task = 0
    for i in info:
        if 'Begin task ' in i:
            curr_task_i = int(i.strip().strip('\n').split('Begin task ')[1])
        elif 'Begin step ' in i:
            curr_task_i = int(i.strip().strip('\n').split('Begin step ')[1])
        if i != '\n' and i != '':
            info_filtered.append(i)
            if 'Evaluation eval_after_decouple' in i:
                if curr_task != curr_task_i:
                    curr_task = curr_task_i
                finest_info = i.split('finest avg: ')[1].split(' classes, coarse avg:')[0].split(' with ')
                if int(finest_info[1]) != 0 and int(finest_info[1]) != curr_finest_classes:
                    record_list.append(float(finest_info[0]))
                curr_finest_classes = int(finest_info[1])
    if restart:
        record_list = record_list[1:]
    return record_list, curr_task


res_dict = {}
model_name_list = []
final_acc_list = []
incremental_acc_list = []
model_setting_list = []
fs_list = []
dataset_list = []
curr_task_list = []

for job_ind in ind2name:
    model_name_i = ind2name[job_ind]
    model_name_split = model_name_i.split('_')
    dataset = model_name_split[1]
    fs_split = model_name_i.split('_fs')
    model_setting = '_'.join(fs_split[0].split('_')[2:])

    model_setting_list.append(model_setting)
    dataset_list.append(dataset)

    log_path = f'/datasets/{dataset}_results/{model_name_i}/train/logs'
    # log_path = f'/Users/chenyuzhao/Downloads/train_Mar3.log'

    record_list, curr_task = load_logs(f'{log_path}/train.log')
    while len(record_list) != 20:
        record_list_tmp, curr_task = load_logs(f'{log_path}/train_retrain_from_step{int(curr_task) + 1}.log', True)

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
        final_acc_list.append(record_list[-1])
        incremental_acc_list.append(-1)

df = pd.DataFrame(
    {'model name': model_name_list, 'dataset': dataset_list, 'model_setting': model_setting_list,
     'final acc': final_acc_list, 'incremental acc': incremental_acc_list,
     'curr_task': curr_task_list}, index=list(ind2name.keys()))

# df.to_csv('/Users/chenyuzhao/Downloads/ctl_acc_summary_Mar3.csv')
df.to_csv('ctl_acc_summary_Mar12_test2.csv')
