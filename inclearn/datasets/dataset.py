import os.path as osp
import numpy as np
import pandas as pd

import albumentations as A
from inclearn.deeprtc.libs import Tree
from albumentations.pytorch import ToTensorV2

from torchvision import datasets, transforms
from collections import defaultdict, OrderedDict


def get_datasets(dataset_names):
    return [get_dataset(dataset_name) for dataset_name in dataset_names.split("-")]


def get_dataset(dataset_name):
    # here returns an object of the specific class (no instance yet!)
    if dataset_name == "cifar10":
        return iCIFAR10
    elif dataset_name == "cifar100":
        return iCIFAR100
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


class DataHandler:
    base_dataset = None
    train_transforms = []
    common_transforms = [ToTensorV2()]
    class_order = None


class iCIFAR10(DataHandler):
    base_dataset_cls = datasets.cifar.CIFAR10
    transform_type = 'torchvision'
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def __init__(self, data_folder, train, device, is_fine_label=False):
        self.base_dataset = self.base_dataset_cls(data_folder, train=train, download=True)
        self.data = self.base_dataset.data
        self.targets = self.base_dataset.targets
        self.n_cls = 10

    @property
    def is_proc_inc_data(self):
        return False

    @classmethod
    def class_order(cls, trial_i):
        return [4, 0, 2, 5, 8, 3, 1, 6, 9, 7]


class iCIFAR100(iCIFAR10):
    base_dataset_cls = datasets.cifar.CIFAR100
    transform_type = 'torchvision'
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    # data_name_hier_dict = OrderedDict({
    #     'vehicles_1': {'motorcycle': {}, 'bus': {}, 'train': {}, 'bicycle': {}, 'pickup_truck': {}},
    #     'trees': {'palm_tree': {}, 'willow_tree': {}, 'maple_tree': {}, 'oak_tree': {}, 'pine_tree': {}},
    #     'large_man-made_outdoor_things': {'bridge': {}, 'road': {}, 'skyscraper': {}, 'house': {}, 'castle': {}},
    #     'food_containers': {'can': {}, 'cup': {}, 'plate': {}, 'bowl': {}, 'bottle': {}},
    #     'small_mammals': {'hamster': {}, 'mouse': {}, 'shrew': {}, 'rabbit': {}, 'squirrel': {}},
    #     'large_omnivores_and_herbivores': {'cattle': {}, 'camel': {}, 'chimpanzee': {}, 'kangaroo': {}, 'elephant': {}},
    #     'flowers': {'rose': {}, 'tulip': {}, 'poppy': {}, 'orchid': {}, 'sunflower': {}},
    #     'large_natural_outdoor_scenes': {'forest': {}, 'plain': {}, 'cloud': {}, 'mountain': {}, 'sea': {}},
    #     'reptiles': {'turtle': {}, 'crocodile': {}, 'dinosaur': {}, 'lizard': {}, 'snake': {}},
    #     'household_furniture': {'wardrobe': {}, 'bed': {}, 'couch': {}, 'chair': {}, 'table': {}},
    #     'fruit_and_vegetables': {'apple': {}, 'pear': {}, 'mushroom': {}, 'sweet_pepper': {}, 'orange': {}},
    #     'large_carnivores': {'bear': {}, 'leopard': {}, 'tiger': {}, 'wolf': {}, 'lion': {}},
    #     'vehicles_2': {'streetcar': {}, 'tractor': {}, 'tank': {}, 'lawn_mower': {}, 'rocket': {}},
    #     'people': {'man': {}, 'boy': {}, 'girl': {}, 'baby': {}, 'woman': {}},
    #     'insects': {'butterfly': {}, 'bee': {}, 'beetle': {}, 'caterpillar': {}, 'cockroach': {}},
    #     'household_electrical_devices': {'lamp': {}, 'television': {}, 'telephone': {}, 'keyboard': {}, 'clock': {}},
    #     'non-insect_invertebrates': {'crab': {}, 'snail': {}, 'lobster': {}, 'worm': {}, 'spider': {}},
    #     'aquatic_mammals': {'dolphin': {}, 'whale': {}, 'otter': {}, 'seal': {}, 'beaver': {}},
    #     'fish': {'aquarium_fish': {}, 'flatfish': {}, 'ray': {}, 'trout': {}, 'shark': {}},
    #     'medium_mammals': {'raccoon': {}, 'fox': {}, 'porcupine': {}, 'skunk': {}, 'possum': {}}
    # })

    data_name_hier_dict = OrderedDict({
        'medium_mammals': {'raccoon': {}, 'fox': {}, 'porcupine': {}, 'skunk': {}, 'possum': {}},
        'fish': {'aquarium_fish': {}, 'flatfish': {}, 'ray': {}, 'trout': {}, 'shark': {}},
        'aquatic_mammals': {'dolphin': {}, 'whale': {}, 'otter': {}, 'seal': {}, 'beaver': {}},
        'non-insect_invertebrates': {'crab': {}, 'snail': {}, 'lobster': {}, 'worm': {}, 'spider': {}},
        'household_electrical_devices': {'lamp': {}, 'television': {}, 'telephone': {}, 'keyboard': {}, 'clock': {}},
        'insects': {'butterfly': {}, 'bee': {}, 'beetle': {}, 'caterpillar': {}, 'cockroach': {}},
        'people': {'man': {}, 'boy': {}, 'girl': {}, 'baby': {}, 'woman': {}},
        'vehicles_2': {'streetcar': {}, 'tractor': {}, 'tank': {}, 'lawn_mower': {}, 'rocket': {}},
        'large_carnivores': {'bear': {}, 'leopard': {}, 'tiger': {}, 'wolf': {}, 'lion': {}},
        'fruit_and_vegetables': {'apple': {}, 'pear': {}, 'mushroom': {}, 'sweet_pepper': {}, 'orange': {}},
        'household_furniture': {'wardrobe': {}, 'bed': {}, 'couch': {}, 'chair': {}, 'table': {}},
        'reptiles': {'turtle': {}, 'crocodile': {}, 'dinosaur': {}, 'lizard': {}, 'snake': {}},
        'large_natural_outdoor_scenes': {'forest': {}, 'plain': {}, 'cloud': {}, 'mountain': {}, 'sea': {}},
        'flowers': {'rose': {}, 'tulip': {}, 'poppy': {}, 'orchid': {}, 'sunflower': {}},
        'large_omnivores_and_herbivores': {'cattle': {}, 'camel': {}, 'chimpanzee': {}, 'kangaroo': {}, 'elephant': {}},
        'small_mammals': {'hamster': {}, 'mouse': {}, 'shrew': {}, 'rabbit': {}, 'squirrel': {}},
        'food_containers': {'can': {}, 'cup': {}, 'plate': {}, 'bowl': {}, 'bottle': {}},
        'large_man-made_outdoor_things': {'bridge': {}, 'road': {}, 'skyscraper': {}, 'house': {}, 'castle': {}},
        'trees': {'palm_tree': {}, 'willow_tree': {}, 'maple_tree': {}, 'oak_tree': {}, 'pine_tree': {}},
        'vehicles_1': {'motorcycle': {}, 'bus': {}, 'train': {}, 'bicycle': {}, 'pickup_truck': {}}
    })

    data_label_index_dict = {
        'medium_mammals': -20, 'fish': -19, 'aquatic_mammals': -18, 'non-insect_invertebrates': -17,
        'household_electrical_devices': -16, 'insects': -15, 'people': -14, 'vehicles_2': -13, 'large_carnivores': -12,
        'fruit_and_vegetables': -11, 'household_furniture': -10, 'reptiles': -9, 'large_natural_outdoor_scenes': -8,
        'flowers': -7, 'large_omnivores_and_herbivores': -6, 'small_mammals': -5, 'food_containers': -4,
        'large_man-made_outdoor_things': -3, 'trees': -2, 'vehicles_1': -1, 'apple': 0, 'aquarium_fish': 1, 'baby': 2,
        'bear': 3, 'beaver': 4, 'bed': 5, 'bee': 6, 'beetle': 7, 'bicycle': 8, 'bottle': 9, 'bowl': 10, 'boy': 11,
        'bridge': 12, 'bus': 13, 'butterfly': 14, 'camel': 15, 'can': 16, 'castle': 17, 'caterpillar': 18, 'cattle': 19,
        'chair': 20, 'chimpanzee': 21, 'clock': 22, 'cloud': 23, 'cockroach': 24, 'couch': 25, 'crab': 26,
        'crocodile': 27, 'cup': 28, 'dinosaur': 29, 'dolphin': 30, 'elephant': 31, 'flatfish': 32, 'forest': 33,
        'fox': 34, 'girl': 35, 'hamster': 36, 'house': 37, 'kangaroo': 38, 'keyboard': 39, 'lamp': 40,
        'lawn_mower': 41, 'leopard': 42, 'lion': 43, 'lizard': 44, 'lobster': 45, 'man': 46, 'maple_tree': 47,
        'motorcycle': 48, 'mountain': 49, 'mouse': 50, 'mushroom': 51, 'oak_tree': 52, 'orange': 53, 'orchid': 54,
        'otter': 55, 'palm_tree': 56, 'pear': 57, 'pickup_truck': 58, 'pine_tree': 59, 'plain': 60, 'plate': 61,
        'poppy': 62, 'porcupine': 63, 'possum': 64, 'rabbit': 65, 'raccoon': 66, 'ray': 67, 'road': 68, 'rocket': 69,
        'rose': 70, 'sea': 71, 'seal': 72, 'shark': 73, 'shrew': 74, 'skunk': 75, 'skyscraper': 76, 'snail': 77,
        'snake': 78, 'spider': 79, 'squirrel': 80, 'streetcar': 81, 'sunflower': 82, 'sweet_pepper': 83, 'table': 84,
        'tank': 85, 'telephone': 86, 'television': 87, 'tiger': 88, 'tractor': 89, 'train': 90, 'trout': 91,
        'tulip': 92, 'turtle': 93, 'wardrobe': 94, 'whale': 95, 'willow_tree': 96, 'wolf': 97, 'woman': 98, 'worm': 99}

    taxonomy_tree = Tree('cifar100', data_name_hier_dict, data_label_index_dict)
    used_nodes, leaf_id, node_labels = taxonomy_tree.prepro()

    def __init__(self, data_folder, train, device, is_fine_label=False, debug=False, df_name=''):
        super().__init__(data_folder, train, is_fine_label)
        self.base_dataset = self.base_dataset_cls(data_folder, train=train, download=True)
        self.data = self.base_dataset.data
        self.targets = self.base_dataset.targets
        self.n_cls = 100
        self.transform_type = 'torchvision'

    @property
    def is_proc_inc_data(self):
        return False

    @classmethod
    def class_order(cls, trial_i):
        idx_to_label = {cls.data_label_index_dict[x]: x for x in cls.data_label_index_dict}
        if trial_i == 0:
            return [
                62, 54, 84, 20, 94, 22, 40, 29, 78, 27, 26, 79, 17, 76, 68, 88, 3, 19, 31, 21, 33, 60, 24, 14, 6, 10,
                16, 82, 70, 92, 25, 5, 28, 9, 61, 36, 50, 90, 8, 48, 47, 56, 11, 98, 35, 93, 44, 64, 75, 66, 15, 38, 97,
                42, 43, 12, 37, 55, 72, 95, 18, 7, 23, 71, 49, 53, 57, 86, 39, 87, 34, 63, 81, 89, 69, 46, 2, 1, 73, 32,
                67, 91, 0, 51, 83, 13, 58, 80, 74, 65, 4, 30, 45, 77, 99, 85, 41, 96, 59, 52
            ]
        elif trial_i == 1:
            return cls.lt_convert([
                [68, 56, 78, 8, 23],
                [84, 90, 65, 74, 76],
                [40, 89, 3, 92, 55],
                [9, 26, 80, 43, 38],
                [58, 70, 77, 1, 85],
                [19, 17, 50, 28, 53],
                [13, 81, 45, 82, 6],
                [59, 83, 16, 15, 44],
                [91, 41, 72, 60, 79],
                [52, 20, 10, 31, 54],
                [37, 95, 14, 71, 96],
                [98, 97, 2, 64, 66],
                [42, 22, 35, 86, 24],
                [34, 87, 21, 99, 0],
                [88, 27, 18, 94, 11],
                [12, 47, 25, 30, 46],
                [62, 69, 36, 61, 7],
                [63, 75, 5, 32, 4],
                [51, 48, 73, 93, 39],
                [67, 29, 49, 57, 33]], idx_to_label)
        elif trial_i == 2:  # PODNet
            return [
                87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45,
                88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6,
                46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76,
                40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39
            ]
        elif trial_i == 3:  # PODNet
            return [
                58, 30, 93, 69, 21, 77, 3, 78, 12, 71, 65, 40, 16, 49, 89, 46, 24, 66, 19, 41, 5, 29, 15, 73, 11, 70,
                90, 63, 67, 25, 59, 72, 80, 94, 54, 33, 18, 96, 2, 10, 43, 9, 57, 81, 76, 50, 32, 6, 37, 7, 68, 91, 88,
                95, 85, 4, 60, 36, 22, 27, 39, 42, 34, 51, 55, 28, 53, 48, 38, 17, 83, 86, 56, 35, 45, 79, 99, 84, 97,
                82, 98, 26, 47, 44, 62, 13, 31, 0, 75, 14, 52, 74, 8, 20, 1, 92, 87, 23, 64, 61
            ]
        elif trial_i == 4:  # PODNet
            return [
                71, 54, 45, 32, 4, 8, 48, 66, 1, 91, 28, 82, 29, 22, 80, 27, 86, 23, 37, 47, 55, 9, 14, 68, 25, 96, 36,
                90, 58, 21, 57, 81, 12, 26, 16, 89, 79, 49, 31, 38, 46, 20, 92, 88, 40, 39, 98, 94, 19, 95, 72, 24, 64,
                18, 60, 50, 63, 61, 83, 76, 69, 35, 0, 52, 7, 65, 42, 73, 74, 30, 41, 3, 6, 53, 13, 56, 70, 77, 34, 97,
                75, 2, 17, 93, 33, 84, 99, 51, 62, 87, 5, 15, 10, 78, 67, 44, 59, 85, 43, 11
            ]
        elif trial_i == 5:
            label_list = [
                ['medium_mammals', 'fish', 'aquatic_mammals', 'non-insect_invertebrates',
                 'household_electrical_devices', 'insects', 'people', 'vehicles_2', 'large_carnivores',
                 'fruit_and_vegetables', 'household_furniture', 'reptiles', 'large_natural_outdoor_scenes',
                 'flowers', 'large_omnivores_and_herbivores', 'small_mammals', 'food_containers',
                 'large_man-made_outdoor_things', 'trees', 'vehicles_1'],
                #           ['vehicles_1', 'trees', 'large_man-made_outdoor_things', 'food_containers', 'small_mammals',
                #            'large_omnivores_and_herbivores', 'flowers', 'large_natural_outdoor_scenes', 'reptiles',
                #            'household_furniture', 'fruit_and_vegetables', 'large_carnivores', 'vehicles_2', 'people',
                #            'insects', 'household_electrical_devices', 'non-insect_invertebrates', 'aquatic_mammals',
                #            'fish', 'medium_mammals'],
                # ['household_furniture', 'large_man-made_outdoor_things', 'medium_mammals',
                #  'food_containers', 'reptiles', 'vehicles_2', 'large_natural_outdoor_scenes',
                #  'large_omnivores_and_herbivores', 'flowers', 'small_mammals', 'trees',
                #  'vehicles_1', 'large_carnivores', 'people', 'aquatic_mammals', 'fish',
                #  'insects', 'fruit_and_vegetables', 'non-insect_invertebrates',
                #  'household_electrical_devices'],
                ['motorcycle', 'bus', 'train', 'bicycle', 'pickup_truck'],
                ['palm_tree', 'willow_tree', 'maple_tree', 'oak_tree', 'pine_tree'],
                ['bridge', 'road', 'skyscraper', 'house', 'castle'],
                ['can', 'cup', 'plate', 'bowl', 'bottle'],
                ['hamster', 'mouse', 'shrew', 'rabbit', 'squirrel'],  #
                ['cattle', 'camel', 'chimpanzee', 'kangaroo', 'elephant'],
                ['rose', 'tulip', 'poppy', 'orchid', 'sunflower'],
                ['forest', 'plain', 'cloud', 'mountain', 'sea'],
                ['turtle', 'crocodile', 'dinosaur', 'lizard', 'snake'],
                ['wardrobe', 'bed', 'couch', 'chair', 'table'],
                ['apple', 'pear', 'mushroom', 'sweet_pepper', 'orange'],
                ['bear', 'leopard', 'tiger', 'wolf', 'lion'],
                ['streetcar', 'tractor', 'tank', 'lawn_mower', 'rocket'],
                ['man', 'boy', 'girl', 'baby', 'woman'],  #
                ['butterfly', 'bee', 'beetle', 'caterpillar', 'cockroach'],
                ['lamp', 'television', 'telephone', 'keyboard', 'clock'],
                ['crab', 'snail', 'lobster', 'worm', 'spider'],  #
                ['dolphin', 'whale', 'otter', 'seal', 'beaver'],  #
                ['aquarium_fish', 'flatfish', 'ray', 'trout', 'shark'],  #
                ['raccoon', 'fox', 'porcupine', 'skunk', 'possum']]
            return label_list

        elif trial_i == 6:
            label_list = [['medium_mammals', 'fish', 'aquatic_mammals', 'non-insect_invertebrates',
                           'household_electrical_devices', 'insects', 'people', 'vehicles_2', 'large_carnivores',
                           'fruit_and_vegetables', 'household_furniture', 'reptiles', 'large_natural_outdoor_scenes',
                           'flowers', 'large_omnivores_and_herbivores', 'small_mammals', 'food_containers',
                           'large_man-made_outdoor_things', 'trees', 'vehicles_1'],
                          ['hamster', 'mouse', 'shrew', 'rabbit', 'squirrel'],
                          ['man', 'boy', 'girl', 'baby', 'woman'],
                          ['crab', 'snail', 'lobster', 'worm', 'spider'],
                          ['dolphin', 'whale', 'otter', 'seal', 'beaver'],  #
                          ['aquarium_fish', 'flatfish', 'ray', 'trout', 'shark'],
                          ['motorcycle', 'bus', 'train', 'bicycle', 'pickup_truck'],
                          ['palm_tree', 'willow_tree', 'maple_tree', 'oak_tree', 'pine_tree'],
                          ['bridge', 'road', 'skyscraper', 'house', 'castle'],
                          ['can', 'cup', 'plate', 'bowl', 'bottle'],
                          ['cattle', 'camel', 'chimpanzee', 'kangaroo', 'elephant'],
                          ['rose', 'tulip', 'poppy', 'orchid', 'sunflower'],
                          ['forest', 'plain', 'cloud', 'mountain', 'sea'],
                          ['turtle', 'crocodile', 'dinosaur', 'lizard', 'snake'],  #
                          ['wardrobe', 'bed', 'couch', 'chair', 'table'],
                          ['apple', 'pear', 'mushroom', 'sweet_pepper', 'orange'],
                          ['bear', 'leopard', 'tiger', 'wolf', 'lion'],
                          ['streetcar', 'tractor', 'tank', 'lawn_mower', 'rocket'],
                          ['butterfly', 'bee', 'beetle', 'caterpillar', 'cockroach'],  #
                          ['lamp', 'television', 'telephone', 'keyboard', 'clock'],
                          ['raccoon', 'fox', 'porcupine', 'skunk', 'possum']]
            return label_list

        elif trial_i == -1:
            label_list = [
                # ['hamster', 'mouse', 'shrew', 'rabbit', 'squirrel']
                # ['poppy', 'orchid', 'table', 'chair', 'wardrobe']
                ['people'],
                ['man', 'boy', 'girl', 'baby', 'woman']
                # 'crab', 'snail', 'lobster', 'worm', 'spider',
                # 'dolphin', 'whale', 'otter', 'seal', 'beaver',  #
                # 'aquarium_fish', 'flatfish', 'ray', 'trout', 'shark',
                # 'motorcycle', 'bus', 'train', 'bicycle', 'pickup_truck',
                # 'palm_tree', 'willow_tree', 'maple_tree', 'oak_tree', 'pine_tree',
                # 'bridge', 'road', 'skyscraper', 'house', 'castle',
                # 'can', 'cup', 'plate', 'bowl', 'bottle',
                # 'cattle', 'camel', 'chimpanzee', 'kangaroo', 'elephant',
                # 'rose', 'tulip', 'poppy', 'orchid', 'sunflower',
                # 'forest', 'plain', 'cloud', 'mountain', 'sea',
                # 'turtle', 'crocodile', 'dinosaur', 'lizard', 'snake',  #
                # 'wardrobe', 'bed', 'couch', 'chair', 'table',
                # 'apple', 'pear', 'mushroom', 'sweet_pepper', 'orange',
                # 'bear', 'leopard', 'tiger', 'wolf', 'lion',
                # 'streetcar', 'tractor', 'tank', 'lawn_mower', 'rocket',
                # 'butterfly', 'bee', 'beetle', 'caterpillar', 'cockroach',  #
                # 'lamp', 'television', 'telephone', 'keyboard', 'clock',
                # 'raccoon', 'fox', 'porcupine', 'skunk', 'possum'
            ]
            return label_list

    @classmethod
    def label_to_target(cls, class_order_list):
        class_id_list = []
        for task_label_list in class_order_list:
            task_id_list = []
            for task_label in task_label_list:
                task_id_list.append(cls.data_label_index_dict[task_label])
            class_id_list.append(task_id_list)
        return class_id_list

    @staticmethod
    def lt_convert(ori_list, mapping):
        new_list = []
        for ori_sub_list in ori_list:
            new_sub_list = []
            for task_label in ori_sub_list:
                new_sub_list.append(mapping[task_label])
            new_list.append(new_sub_list)
        return new_list


class DataHandler:
    base_dataset = None
    train_transforms = []
    common_transforms = [ToTensorV2()]
    class_order = None