import pandas as pd
import numpy as np

np.random.seed(0)

ori_csv_path = './inclearn/datasets/hab_in_situ_v2_c34_workshop2019.csv'
df = pd.read_csv(ori_csv_path)

# classes_list = list(set(df['label']))
cate_rename_dict = {'Thalassionema or Thalassiothrix chain': 'Thalassionema_or_Thalassiothrix_chain','Prorocentrum gracile':'Prorocentrum_gracile','Chaetoceros socialis':'Chaetoceros_socialis',
                    'Pseudo-nitzschia chain':'Pseudo_nitzschia_chain', 'Prorocentrum micans':'Prorocentrum_micans', 'Lingulodinium polyedra':'Lingulodinium_polyedra',
                    'Ceratium falcatiforme or fusus':'Ceratium_falcatiforme_or_fusus', 'Ceratium other':'Ceratium_Other', 'Unknown pennate diatom':'Unknown_pennate_diatom',
                    'Ceratium furca':'Ceratium_furca','diatom chain':'diatom_chain','Unknown dinoflagellates elongated':'Unknown_dinoflagellates_elongated','Kelp Fragment':'Kelp_Fragment'}





print(df.shape)
# print(len(cate_list))
df = df[df['label']!='Marine Lashes']
df.reindex(list(range(df.shape[0])))
for i in cate_rename_dict:

    df.replace(i, cate_rename_dict[i], inplace=True)

# cate_list = list(set(df['label']))
cate_list = ['Aggregate', 'Gyrodinium', 'Acantharea', 'Kelp_Fragment', 'Chaetoceros_socialis', 'Unknown_dinoflagellates_elongated', 'Ciliates', 'Pseudo_nitzschia_chain', 
             'Thalassionema_or_Thalassiothrix_chain', 'Cochlodinium', 'Torodinium', 'Polykrikos', 'Prorocentrum_gracile', 'Protoperidinium', 
             'Sand', 'Ceratium_Other', 'Ceratium_falcatiforme_or_fusus', 'Ceratium_furca', 'Akashiwo', 'Unknown_pennate_diatom', 'Bubble', 
             'Eucampia', 'BadImageSegmentation', 'Prorocentrum_micans', 'Lingulodinium_polyedra', 'Nauplii', 'diatom_chain', 'Chattonella', 'Dinophysis']

df['train_flag'] = [1]*df.shape[0]

for cate_i in cate_list:
    index_i = df[df['label'] == cate_i].index
    test_index_i = pd.Index(np.random.choice(index_i, int(0.1*len(index_i)), replace=False))
    df['train_flag'][test_index_i]=[0]*len(test_index_i)



# for cate_i in cate_list:
#     index_i = df[df['label'] == cate_i].index
#     test_number = np.sum(df['train_flag'][index_i]==0)
#     print(f'cate:{cate_i}, total{len(index_i)}, test{test_number}, rate:{test_number/len(index_i)}')

df['images'] = df['images'].str.replace('/data6/phytoplankton-db/hab_in_situ/images/workshop2019_v2', 'workshop2019_v2')
df['images'] = df['images'].str.replace('Kelp Fragment', 'Kelp_Fragment')
df.to_csv('./inclearn/datasets/workshop2019_processed.csv', index=False)


# df = df[df['label']!='Mashi']