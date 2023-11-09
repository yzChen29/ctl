import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

csv_path = '/datasets/iNat_datasets/iNat_1010_max600/train_img_info_600.csv'

df = pd.read_csv(csv_path)
cate_id_dict = {}
tax_dict = {}
tax_idx_dict = {}

rename_dict = {}
label_list = []
label_ordered_list = []
random_task_order_list = []


cate_name_list = list(set(df['name']))
print(cate_name_list)


for i in cate_name_list:
    sup_cate = list(df[df['name']==i]['supercategory'])[0]
    cate_id = list(df[df['name']==i]['category_id'])[0]
    if sup_cate not in tax_dict:
        tax_dict[sup_cate] = {i:{}}
    else:
        tax_dict[sup_cate][i] = {}
    
    cate_id_dict[i] = cate_id


count = 0

for i in tax_dict:
    print(i)
    for j in tax_dict[i]:
        tax_idx_dict[j] = count
        count += 1

['Aves', 'Amphibia', 'Mollusca', 'Reptilia', 'Animalia', 'Mammalia', 'Fungi', 'Arachnida', 'Plantae', 'Insecta'], 

# Random Tax   
for i in tax_dict:
    label_list.append(list(tax_dict[i].keys()))
np.random.shuffle(label_list)
label_list.insert(0, list(tax_dict.keys()))

# Ordered Tax
for i in ['Aves', 'Plantae', 'Insecta', 'Reptilia', 'Mammalia', 'Amphibia', 'Mollusca', 'Fungi', 'Animalia', 'Arachnida']:
    label_ordered_list.append(list(tax_dict[i].keys()))

label_ordered_list.insert(0, list(tax_dict.keys()))

# Der Ori Random
random_task_order_list_tmp = []
for i in tax_dict:
    random_task_order_list_tmp+=list(tax_dict[i].keys())
np.random.shuffle(random_task_order_list_tmp)
for i in range(10):
    random_task_order_list.append(random_task_order_list_tmp[i*10:(i+1)*10])



print(tax_dict)
print(tax_idx_dict)
print(label_ordered_list)
print(random_task_order_list)

# img_path = '/datasets/iNat_datasets/iNat_1010_max600/train_val_images/Amphibia/Plethodon cinereus/0349d45dc469dad477bc1e257e362712.jpg'
# img = plt.imread(img_path)
# plt.imshow(img)
# plt.savefig('./test_111.png')

df.drop(df[df['file_name']=='train_val_images/Aves/Ardea herodias/7496a76e41786c2798dfc2947d666978.jpg'].index, inplace=True)

df_new = df[['file_name', 'name']]
df_new.rename(columns={'file_name': 'images', 'name': 'label'}, inplace=True)

df_new['train_flag'] = [1]*df.shape[0]

for cate_i in cate_name_list:
    index_i = df_new[df_new['label'] == cate_i].index
    test_index_i = pd.Index(np.random.choice(index_i, int(1/6*len(index_i)), replace=False))
    # test_index_i = pd.Index(np.random.choice(index_i, int(1/30*len(index_i)), replace=False))
    df_new['train_flag'][test_index_i]=[0]*len(test_index_i)


df_new.to_csv('/datasets/iNat_datasets/iNat_1010_max600/train_img_info_600_processed.csv', index=False)
df_new.to_csv('./train_img_info_600_processed.csv', index=False)

# df_new.to_csv('/datasets/iNat_datasets/iNat_1010_max600/train_img_info_600_processed_only_for_debug.csv', index=False)
# df_new.to_csv('./train_img_info_600_processed_for_debug.csv', index=False)
print()