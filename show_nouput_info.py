import pandas as pd
import numpy as np


def get_nout_mean(file_path, task_i, save=False):
    nout_csv_path = f'{file_path}/nout_details_task{task_i}.csv'
    df_nout = pd.read_csv(nout_csv_path)

    nout_mean_dict = {}
    col_nout_pair_list = []

    for col_i in list(df_nout.columns):
        if col_i != 'targets':
            mean_col_i = np.mean(df_nout.loc[:, col_i])
            nout_mean_dict[col_i] = mean_col_i
            col_nout_pair_list.append((col_i, mean_col_i))

    col_nout_pair_list.sort(key=lambda x:abs(x[1]), reverse=True)



    if save:
        df_nout_mean = pd.DataFrame(nout_mean_dict, index=[0])
        df_nout_mean.to_csv(f'{file_path}/nout_mean.csv', index=False)


    return nout_mean_dict, col_nout_pair_list

def show_trend_diff_tasks(base_path, task_i_list, file_name_i):

    df_max = pd.read_csv(
        f'{base_path}/nout_details_task{max(task_i_list)}.csv')

    total_col = list(df_max.columns)
    total_col.remove('targets')

    nout_mean_summary_dict = {}
    for col_i in total_col:
        nout_mean_summary_dict[col_i] = []

    file_path_i = file_path_dict[file_name_i]
    for task_i in task_i_list:
        nout_mean_dict_i, col_nout_pair_list_i = get_nout_mean(file_path_i, task_i, save=False)
        for col_i in total_col:
            if col_i not in nout_mean_dict_i:
                nout_mean_summary_dict[col_i].append(None)
            else:
                nout_mean_summary_dict[col_i].append(nout_mean_dict_i[col_i])

    df_nout_mean_summary = pd.DataFrame(nout_mean_summary_dict, index=[f'{file_name_i}_task{task_i}' for task_i in task_i_list])
    df_nout_mean_summary.to_csv(f'{base_path}/nout_mean_summary_for_{file_name_i}.csv')

def show_nout_trend_diff_datasets(file_path_dict, task_i, save_path):
    nout_mean_summary_dict = {}

    tmp_nout_mean_dict = {}

    for file_name_i in file_path_dict:
        file_path_i = file_path_dict[file_name_i]
        nout_mean_dict_i, col_nout_pair_list_i = get_nout_mean(file_path_i, task_i, save=False)
        tmp_nout_mean_dict[file_name_i] = [nout_mean_dict_i, col_nout_pair_list_i]

    total_col = []
    for nout_file_i in tmp_nout_mean_dict.values():
        for col_j in nout_file_i:
            if col_j not in total_col:
                total_col.append(col_j)

    for task_i in tmp_nout_mean_dict:
        nout_mean_dict_i, col_nout_pair_list_i = tmp_nout_mean_dict[task_i][0], tmp_nout_mean_dict[task_i][1]
        for col_i in total_col:
            if col_i not in nout_mean_dict_i:
                nout_mean_summary_dict[col_i].append(None)
            else:
                nout_mean_summary_dict[col_i].append(nout_mean_dict_i[col_i])


    df_nout_mean_summary = pd.DataFrame(nout_mean_summary_dict,
                                        index=[f'{file_name_i}_task{task_i}' for file_name_i in file_path_dict])
    df_nout_mean_summary.to_csv(f'{save_path}/nout_mean_summary.csv')

def show_top_nlayer_result(taxonomy_tree, csv_path, n=1):
    if n == 1:
        df = pd.read_csv(csv_path)
        top_1_columns = taxonomy_tree.depth_dict[0]
        df['max_top_1_index'] = df[[f'p_root_c_{i}' for i in top_1_columns]].apply(
            lambda x: np.argmax(x), axis=1)
        pred_list = np.array(df['max_index'])

        targets_list = list(df['targets'])

        gt_list = []
        index2child = taxonomy_tree.nodes.get('root').children
        child2index = {index2child[i]: i for i in index2child}
        for i in targets_list:
            gt_list.append(child2index[taxonomy_tree.get_parent_n_layer(node_label_list=[i], n_layer=1)[0][0]])
        gt_list = np.array(gt_list)
        total_acc = np.sum(gt_list == pred_list) / len(pred_list)

        partial_list = {}
        for i in range(4):
            index = np.where(gt_list == i)
            partial_res_i = np.sum(pred_list[index] == i) / len(index[0])
            partial_list[f'i_{index2child[i]}'] = partial_res_i

        return total_acc, partial_list
    elif n==2:
        pass
    else:
        raise('n layers not available')

if __name__ == '__main__':

    file_path_dict = {
                        'BFS_imagnet': '/Users/chenyuzhao/Downloads/ctl_rtc_imagenet100_trial3_BFS_seed500_with_imagenet_config_check_noutput',
                        'BFS_imagenet_joint_ce_sp12': '/Users/chenyuzhao/Downloads/ctl_rtc_imagenet100_trial3_BFS_seed500_add_joint_ce_loss_sp0p1_0p2_imagenet_config_check_noutput',
                        'DFS_imagenet': '/Users/chenyuzhao/Downloads/ctl_rtc_imagenet100_trial3_DFS_seed500_retrain_from_task13_check_nouput'
                      }

    base_path = '/Users/chenyuzhao/Downloads/ctl_rtc_imagenet100_trial3_BFS_seed500_with_imagenet_config_check_noutput'

    taxonmy_tree_BFS = None
    taxonmy_tree_DFS = None