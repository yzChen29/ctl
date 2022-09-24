import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

test_path = '/Users/chenyuzhao/Downloads/imagenet100_trial3_BFS_seed1993_reuslt/eval_after_decouple_total'

task_class_dir = {}
class_i_curr = []
csv_name_list = os.listdir(test_path)

for i in range(25):
    for j in csv_name_list:
        if f'_task_{i}' in j and '.csv' in j:
            df = pd.read_csv(f'{test_path}/{j}')
            break

    task_i_class = list(df['class_index'])
    task_i_class.remove('epoch_0')

    task_i_class = [int(i) for i in task_i_class if '-' not in i]

    try:
        for k in class_i_curr:
            task_i_class.remove(k)
    except:
        print(task_i_class)
        print(k)
        raise('Error')

    task_class_dir[i] = task_i_class
    class_i_curr += task_i_class



print(task_class_dir)


# class trend over task
for i in range(5):
    task_class = task_class_dir[5]
    class_i = task_class[i]
    acc_trend = []
    for i in range(21):
        for j in csv_name_list:
            if f'_task_{i}' in j and '.csv' in j:
                df = pd.read_csv(f'{test_path}/{j}')
            break
        index_i = df[df['class_index'] == class_i].index
        avg_acc = df.iloc[index_i, 1]
        try:
            acc_trend.append((i, float(avg_acc)))
        except Exception:
            pass
    print(acc_trend)
    plt.plot([i[0] for i in acc_trend], [i[1] for i in acc_trend])
    plt.title(f'avg acc trend for class {class_i}')
    plt.xlabel('#task')
    plt.ylabel('avg acc')
    plt.show()

# incremental acc over task
#
# res_acc_avg_dict = {}
# res_acc_avg_dict_task = {}
# task_class = task_class_dir[6]
# class_i = task_class[2]

# for task_i in range(1, 21):
#     task_class = task_class_dir[task_i]
#     acc_trend_task = []
#     for class_i_index in range(5):
#         class_i = task_class[class_i_index]
#         acc_trend = []
#         for i in range(21):
#             df = pd.read_csv(f'{test_path}/_task_{i}.csv')
#             index_i = df[df['class_index'] == class_i].index
#             avg_acc = df.iloc[index_i, 1]
#             try:
#                 acc_trend.append((i, float(avg_acc)))
#             except Exception:
#                 pass
#
#         res_acc_avg_dict[class_i] = np.round(np.average([i[1] for i in acc_trend]), 3)
#         acc_trend_task.append(np.round(np.average([i[1] for i in acc_trend]), 3))
#     res_acc_avg_dict_task[task_i] = np.round(np.average(acc_trend_task), 3)
#
# print(res_acc_avg_dict_task)
# print(res_acc_avg_dict)
# print(np.round(np.average(list(res_acc_avg_dict.values())), 3))


# print('trial 6')
# a = '''{0: [], 1: ['36', '50', '65', '74', '80'], 2: ['2', '11', '35', '46', '98'], 3: ['26', '45', '77', '79', '99'], 4: ['4', '30', '55', '72', '95'], 5: ['1', '32', '67', '73', '91'], 6: ['8', '13', '48', '58', '90'], 7: ['47', '52', '56', '59', '96'], 8: ['12', '17', '37', '68', '76'], 9: ['9', '10', '16', '28', '61'], 10: ['15', '19', '21', '31', '38'], 11: ['54', '62', '70', '82', '92'], 12: ['23', '33', '49', '60', '71'], 13: ['27', '29', '44', '78', '93'], 14: ['5', '20', '25', '84', '94'], 15: ['0', '51', '53', '57', '83'], 16: ['3', '42', '43', '88', '97'], 17: ['41', '69', '81', '85', '89'], 18: ['6', '7', '14', '18', '24'], 19: ['22', '39', '40', '86', '87'], 20: ['34', '63', '64', '66', '75']}
# [(4, 0.56), (5, 0.52), (6, 0.54), (7, 0.54), (8, 0.54), (9, 0.56), (10, 0.5), (11, 0.5), (12, 0.5), (13, 0.44), (14, 0.46), (15, 0.46), (16, 0.38), (17, 0.38), (18, 0.34), (19, 0.36), (20, 0.38)]
# {1: 0.604, 2: 0.505, 3: 0.649, 4: 0.581, 5: 0.64, 6: 0.827, 7: 0.712, 8: 0.793, 9: 0.74, 10: 0.708, 11: 0.714, 12: 0.744, 13: 0.605, 14: 0.759, 15: 0.772, 16: 0.748, 17: 0.811, 18: 0.656, 19: 0.718, 20: 0.792}
# {'36': 0.814, '50': 0.527, '65': 0.538, '74': 0.553, '80': 0.59, '2': 0.608, '11': 0.44, '35': 0.433, '46': 0.559, '98': 0.487, '26': 0.543, '45': 0.631, '77': 0.662, '79': 0.726, '99': 0.684, '4': 0.473, '30': 0.747, '55': 0.468, '72': 0.478, '95': 0.738, '1': 0.751, '32': 0.524, '67': 0.555, '73': 0.704, '91': 0.665, '8': 0.924, '13': 0.753, '48': 0.868, '58': 0.884, '90': 0.704, '47': 0.544, '52': 0.789, '56': 0.954, '59': 0.639, '96': 0.634, '12': 0.731, '17': 0.818, '37': 0.708, '68': 0.898, '76': 0.812, '9': 0.82, '10': 0.61, '16': 0.787, '28': 0.793, '61': 0.692, '15': 0.684, '19': 0.747, '21': 0.818, '31': 0.724, '38': 0.565, '54': 0.608, '62': 0.81, '70': 0.67, '82': 0.85, '92': 0.63, '23': 0.751, '33': 0.758, '49': 0.818, '60': 0.691, '71': 0.704, '27': 0.648, '29': 0.605, '44': 0.44, '78': 0.71, '93': 0.62, '5': 0.76, '20': 0.863, '25': 0.517, '84': 0.74, '94': 0.917, '0': 0.803, '51': 0.787, '53': 0.89, '57': 0.683, '83': 0.697, '3': 0.556, '42': 0.744, '43': 0.788, '88': 0.884, '97': 0.768, '41': 0.795, '69': 0.86, '81': 0.79, '85': 0.9, '89': 0.71, '6': 0.693, '7': 0.513, '14': 0.693, '18': 0.6, '24': 0.78, '22': 0.7, '39': 0.83, '40': 0.58, '86': 0.72, '87': 0.76, '34': 0.72, '63': 0.88, '64': 0.74, '66': 0.72, '75': 0.9}
# 0.704'''
# print(a)


#
# trial_6_dict = {0: [], 1: ['36', '50', '65', '74', '80'], 2: ['2', '11', '35', '46', '98'], 3: ['26', '45', '77', '79', '99'], 4: ['4', '30', '55', '72', '95'], 5: ['1', '32', '67', '73', '91'], 6: ['8', '13', '48', '58', '90'], 7: ['47', '52', '56', '59', '96'], 8: ['12', '17', '37', '68', '76'], 9: ['9', '10', '16', '28', '61'], 10: ['15', '19', '21', '31', '38'], 11: ['54', '62', '70', '82', '92'], 12: ['23', '33', '49', '60', '71'], 13: ['27', '29', '44', '78', '93'], 14: ['5', '20', '25', '84', '94'], 15: ['0', '51', '53', '57', '83'], 16: ['3', '42', '43', '88', '97'], 17: ['41', '69', '81', '85', '89'], 18: ['6', '7', '14', '18', '24'], 19: ['22', '39', '40', '86', '87'], 20: ['34', '63', '64', '66', '75']}
# trial_6_dict_res = {1: 0.604, 2: 0.505, 3: 0.649, 4: 0.581, 5: 0.64, 6: 0.827, 7: 0.712, 8: 0.793, 9: 0.74, 10: 0.708, 11: 0.714, 12: 0.744, 13: 0.605, 14: 0.759, 15: 0.772, 16: 0.748, 17: 0.811, 18: 0.656, 19: 0.718, 20: 0.792}
#
# dict5_2_dict6 = {}
# for i in range(1, 21):
#     for j in range(1, 21):
#         if trial_6_dict[j] == task_class_dir[i]:
#             dict5_2_dict6[i] = j
# print(dict5_2_dict6)
# res_dict = {i: (i, res_acc_avg_dict_task[i], dict5_2_dict6[i], trial_6_dict_res[dict5_2_dict6[i]])for i in range(1, 21)}
#
#
# for i in range(1, 21):
#     print(res_dict[i])




# # task i performance in the last task
#
# task_class_dir = {}
# class_i_curr = []
# for i in range(21):
#     df = pd.read_csv(f'{test_path}/after_train_task_{i}.csv')
#     task_i_class = list(df['class_index'])
#     task_i_class.remove('epoch_0')
#     task_i_class = [i for i in task_i_class if '-' not in i]
#     for j in class_i_curr:
#         task_i_class.remove(j)
#     task_class_dir[i] = task_i_class
#     class_i_curr += task_i_class
#
# df_last = pd.read_csv(f'{test_path}/after_train_task_20.csv')
# task_i_avg_acc = []
# for i in range(1, 21):
#     task_i_acc_list = []
#     for j in task_class_dir[i]:
#         index_i = df_last[df_last['class_index'] == j].index
#         avg_acc = df_last.iloc[index_i, 2]
#         task_i_acc_list.append(avg_acc)
#     task_i_avg_acc.append(np.mean(task_i_acc_list))
#
# plt.plot(list(range(1, 21)), task_i_avg_acc)
# plt.xlabel('task_i')
# plt.ylabel('avg acc in the final task')
# # plt.savefig('/Users/chenyuchao/Downloads/test_acc/finaltest_performance_for_all_task.png')
# plt.show()
