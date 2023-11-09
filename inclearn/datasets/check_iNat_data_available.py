import cv2
import pandas as pd

df = pd.read_csv('/datasets/iNat_datasets/iNat_1010_max600/train_img_info_600_processed.csv')

img_path_list = list(df['images'])

# img_path_list = ['/datasets/iNat_datasets/iNat_1010_max600/train_val_images/Aves/Ardea herodias/7547f1a979588a9630d157270bacd3bf.jpg', '/datasets/iNat_datasets/iNat_1010_max600/train_val_images/Aves/Ardea herodias/7496a76e41786c2798dfc2947d666978.jpg', '/datasets/iNat_datasets/iNat_1010_max600/train_val_images/Aves/Ardea herodias/7547f1a979588a9630d157270bacd3bf.jpg']
error_list = []
# ['train_val_images/Aves/Ardea herodias/7496a76e41786c2798dfc2947d666978.jpg']
for i in img_path_list:
    
    img = cv2.imread(f'/datasets/iNat_datasets/iNat_1010_max600/{i}')
    if img is None:
        error_list.append(i)

error_file_csv = pd.DataFrame({'file_name':error_list}, index=list(range(len(error_list))))
error_file_csv.to_csv('./error_file.csv', index=False)
print(error_list)
