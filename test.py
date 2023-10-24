import tarfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# root_path = '/opt/app-root/src/summer2019.tar'
# # curr_path = './plankton_analysis.tar'

# # # # with tarfile.open(curr_path, 'r') as tar:
# # # #     for member in tar.getmembers():
# # # #         print(member.name)

# # # #     print()

# with tarfile.open(root_path, 'r') as tar:
#     # 解压到当前文件夹
#     tar.extractall(path='/datasets/plankton_datasets')

# # import matplotlib.pyplot as plt
# # import tifffile

# # img = tifffile.imread('/datasets/plankton_datasets/hab_in_vitro/images/20190516/001/SPCBench-1488567949-003254-003-612-1876-80-88-.tif')
# # plt.imshow(img)
# # plt.savefig('test.png', cmap='gray')
# # plt.show()

# import os

# def count_files_in_directory(directory):
#     return sum([len(files) for subdir, dirs, files in os.walk(directory)])

# directory_path = '/datasets/plankton_datasets/summer2019'  # 替换为你的文件夹路径
# print(f"There are {count_files_in_directory(directory_path)} files in the directory.")

# classes = ['Acantharea', 'Aggregate', 'Akashiwo', 'BadImageSegmentation', 'Bubble', 'Ceratium_falcatiforme_or_fusus', 'Ceratium furca', 'Ceratium other', 'Chaetoceros socialis', 'Chattonella', 'Ciliates', 'Cochlodinium', 'Dinophysis', 'Eucampia', 'Gyrodinium', 'Kelp Fragment', 'Lingulodinium polyedra', 'Marine Lashes', 'Nauplii', 'Polykrikos', 'Prorocentrum gracile', 'Prorocentrum micans', 'Protoperidinium', 'Pseudo-nitzschia chain', 'Sand', 'Thalassionema or Thalassiothrix chain', 'Torodinium', 'Unknown dinoflagellates elongated', 'Unknown pennate diatom', 'diatom chain']

# print(len(classes))


# import numpy as np

# tmp = np.load('./hab_classes.npy')
# print()

# import nltk
# nltk.download('wordnet')


# df_1 = pd.read_csv('/datasets/plankton_datasets/hab_in_situ_v2_c34_workshop2019.csv')

# tmp = df_1[df_1['label']=='Kelp Fragment']
# print(tmp)
# print(list(tmp['images'])[0])

# print()


# label = list(df_1['label'])
# print(len(set(label)))



# import torch
# from torchvision import transforms
# from torchvision.transforms import functional as F
# from PIL import Image


# img_path = './SPCP2-1492428219-125125-000-1816-944-104-64.jpg'



# # 创建一个示例的图像
# image = Image.open(img_path)



# m, n = image.size
# offset = abs(n-m)
# left_offset, top_offset, right_offset, bottom_offset = 0, 0, 0, 0
# if m>n:
#     top_offset = int(offset/2)
#     bottom_offset = offset-top_offset
# elif m<n:
#     left_offset = int(offset/2)
#     right_offset = offset-left_offset
# else:
#     pass
# padding = (left_offset, top_offset, right_offset, bottom_offset)
# padded_image = F.pad(image, padding)


# # 使用torch.nn.functional.pad进行填充
# padded_image = F.pad(image, padding)
# # padded_image_array = np.array(padded_image)
# print(padded_image.size)
# plt.imshow(padded_image)
# # plt.show()
# plt.savefig('./test_padding.png')
# plt.close()
# print()

# 可以将填充后的图像传递给其他transforms进行预处理
# preprocess = transforms.Compose([
#     transforms.Resize((256, 256)),  # 调整图像大小
#     transforms.ToTensor(),           # 将图像转换为张量
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
# ])

# preprocessed_image = preprocess(padded_image)

# df = pd.read_csv('/datasets/plankton_datasets/hab_in_situ_v2_c34_workshop2019.csv')
# rename_dict = {'BadImageSegmentation': 'Bad_Image_Segmentation','Ceratium falcatiforme or fusus':'Ceratium_falcatiforme_or_fusus', 
#                'Ceratium furca': 'Ceratium_furca', 'Ceratium other': 'Ceratium_other', 'Chaetoceros socialis':'Chaetoceros_socialis', 
#                'Kelp Fragment':'Kelp_Fragment', 'Lingulodinium polyedra':'Lingulodinium_polyedra', 
#                'Marine Lashes':'Marine_Lashes', 'Prorocentrum gracile':'Prorocentrum_gracile', 'Prorocentrum micans': 'Prorocentrum_micans',
#                'Pseudo-nitzschia chain': 'Pseudo_nitzschia_chain', 'Thalassionema or Thalassiothrix chain': 'Thalassionema_or_Thalassiothrix_chain', 
#                 'Unknown dinoflagellates elongated': 'Unknown_dinoflagellates_elongated', 'Unknown pennate diatom': 'Unknown_pennate_diatom',
#                 'diatom chain': 'diatom_chain'
# }
# for i in rename_dict:
#     df.replace(i, rename_dict[i], inplace=True)
# df.to_csv('/datasets/plankton_datasets/workshop2019.csv', index=False)

class_list = ['Torodinium', 'Ceratium_falcatiforme_or_fusus', 'Chaetoceros_socialis', 'Akashiwo', 'Bad_Image_Segmentation', 'Ceratium_furca', 'Gyrodinium', 'Ceratium_other', 'Lingulodinium_polyedra', 'Kelp_Fragment', 'Prorocentrum_gracile', 'Dinophysis', 'Marine_Lashes', 'Prorocentrum_micans', 'Thalassionema_or_Thalassiothrix_chain', 'Sand', 'Unknown_pennate_diatom', 'Nauplii', 'Unknown_dinoflagellates_elongated', 'Ciliates', 'Eucampia', 'Chattonella', 'Protoperidinium', 'Aggregate', 'Polykrikos', 'Cochlodinium', 'Acantharea', 'Pseudo_nitzschia_chain', 'diatom_chain', 'Bubble']
df = pd.read_csv('/datasets/plankton_datasets/workshop2019.csv')
class_index_dict = {}
for i in class_list:
    tmp_i = df[df['label']==i]
    class_index_dict[i] = tmp_i.index
    flag_1, flag_2 = False, False
    for j in tmp_i['images']:
        if not flag_1 and 'part1' in j:
            print(f'label {i}, part1 sample path: \n {j}')
            flag_1 = True
        if not flag_2 and 'part2' in j:
            print(f'label {i}, part2 sample path: \n {j}')
            flag_2 = True
        if flag_1 and flag_2:
            print(f'label {i}, part12 all exist')
            break
    if flag_1 and not flag_2:
        print(f'label {i}, part2 not exist')
    elif not flag_1 and flag_2:
        print(f'label {i}, part1 not exist')
    print('\n\n\n\n\n')



# # tmp = df.head(5)
# print(tmp)
# print(tmp['images'][0])




# import zipfile
# zip_file_path = '/datasets/workshop2019_v2.zip'
# extracted_folder_path = '/datasets/workshop2019_v2_datasets'
# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     # 解压ZIP文件到目标文件夹
#     zip_ref.extractall(extracted_folder_path)