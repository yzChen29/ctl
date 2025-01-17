import random
import cv2
import numpy as np
import os.path as osp
from PIL import Image
import multiprocessing as mp
import albumentations as A
import random
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.transforms import functional as F

from .dataset import get_dataset
from inclearn.deeprtc.libs import Tree
from inclearn.tools.data_utils import construct_balanced_subset
from inclearn.tools.utils import set_feature_size
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)


def get_data_folder(data_folder, dataset_name):
    return osp.join(data_folder, dataset_name)


class IncrementalDataset:
    def __init__(self, trial_i, dataset_name, is_distributed=False, random_order=False, shuffle=True, workers=10,
                 device=None, batch_size=128, seed=1, sample_rate=[0.1, 0.2, 0.3], increment=10, validation_split=0.0,
                 resampling=False, data_folder="./data", start_class=0, mode_train=True, taxonomy=None, connect_fs=512, debug=False, 
                 full_connect=True, df_name=''):
        # The info about incremental split
        self.trial_i = trial_i
        self.start_class = start_class
        self.mode_train = mode_train
        # the number of classes for each step in incremental stage
        self.task_size = increment
        self.is_distributed = is_distributed
        self.increments = []
        self.random_order = random_order
        self.validation_split = validation_split
        self._device = device

        self._seed = seed
        self.connect_fs = connect_fs
        self.full_connect = full_connect
        # self._s_rate = sample_rate
        self._s_rate = sample_rate
        self._workers = workers
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._resampling = resampling
        # -------------------------------------
        # Dataset Info
        # -------------------------------------
        # self.data_folder = get_data_folder(data_folder, dataset_name)
        if 'imagenet' in dataset_name:
            ds_name = 'imagenet-ilsvrc2012'
        elif 'plankton' in dataset_name:
            ds_name = 'workshop2019_v2_datasets'

        elif 'iNat' in dataset_name:
            if dataset_name == 'iNat100':
                ds_name = 'iNat_1010_max600'
            else:
                raise ('check dataset name')
        else:
            ds_name = dataset_name
        self.data_folder = get_data_folder(data_folder, ds_name)
        self.dataset_name = dataset_name
        self.train_dataset = None
        self.test_dataset = None
        self.n_tot_cls = -1
        # datasets is the object
        dataset_class = get_dataset(dataset_name)

        self.debug = debug
        self.df_name = df_name
        self._setup_data(dataset_class)

        # Currently, don't support multiple datasets
        self.train_transforms = dataset_class.train_transforms
        self.test_transforms = dataset_class.test_transforms
        # torchvision or albumentations
        self.transform_type = dataset_class.transform_type

        # Taxonomy setting
        self.taxonomy = taxonomy
        self.curriculum = None
        self._setup_curriculum(dataset_class)
        self._current_task = 0
        self.taxonomy_tree = dataset_class.taxonomy_tree
        self.cur_part_tree = Tree(self.dataset_name)
        self.current_ordered_dict = OrderedDict()

        # memory Mt
        # self.data_memory = None
        #         # self.targets_memory = []
        self.memory_dict = {}
        # Incoming data D_t
        self.data_cur, self.targets_cur = None, None
        self.targets_cur_unique = []
        self.targets_all_unique = []
        # Available data \tilde{D}_t = D_t \cup M_t
        self.data_inc, self.targets_inc = None, None  # Cur task data + memory
        self.data_test_inc, self.targets_test_inc = [], []
        # self.targets_ori = None
        # Available data stored in cpu memory.
        self.shared_data_inc, self.shared_test_data = None, None
        self.task_info = pd.DataFrame()

    @property
    def n_tasks(self):
        return len(self.curriculum)

    def new_task(self):
        x_train, y_train, x_val, y_val, x_test, y_test = self._get_cur_data_for_all_children()

        # for correct avg acc start
        fine_level_index = np.where(y_test>=0)
        if fine_level_index[0].shape[0] != 0:
            x_test, y_test = x_test[fine_level_index], y_test[fine_level_index]
        # for correct avg acc end 
        
        self.data_cur, self.targets_cur = x_train, y_train
        self.targets_cur_unique = list(OrderedDict.fromkeys(self.targets_cur))
        print(self.targets_cur_unique)
        # self.targets_cur_unique = sorted(list(set(self.targets_cur)))
        self.targets_all_unique += self.targets_cur_unique
        if self._current_task >= len(self.curriculum):
            raise Exception("No more tasks.")

        # important
        if not self.debug:
            if self.mode_train:
                if self._current_task > 0:
                    self._update_memory_for_new_task(self.curriculum[self._current_task])

                # if self.data_memory is not None:
                    data_memory, targets_memory = self.gen_memory_array_from_dict()
                    print("Set memory of size: {}.".format(len(data_memory)))
                    if len(data_memory) != 0:
                        x_train = np.concatenate((x_train, data_memory))
                        y_train = np.concatenate((y_train, targets_memory))

        self.data_inc, self.targets_inc = x_train, y_train
        self.data_test_inc, self.targets_test_inc = x_test, y_test

        train_loader = self._get_loader(x_train, y_train, mode="train")
        val_loader = self._get_loader(x_val, y_val, shuffle=False, mode="test")
        # print(val_loader.sampler)
        test_loader = self._get_loader(x_test, y_test, shuffle=False, mode="test")

        # old method
        # task_until_now = self.curriculum[:self._current_task + 1]
        # cur_parent_node = self.taxonomy_tree.get_task_parent(self.curriculum[self._current_task])
        # self.current_ordered_dict[cur_parent_node] = self.curriculum[self._current_task]
        # self.cur_part_tree = self.taxonomy_tree.gen_partial_tree(task_until_now)

        # new method
        self.taxonomy_tree.expand_tree(self.cur_part_tree, self.curriculum[self._current_task])
        self.cur_part_tree.reset_params()
        # self.cur_part_tree = self.taxonomy_tree.reset_params_2(self.cur_part_tree)

        new_task_info = self.get_new_task_info(len(x_train), len(y_train))
        self.task_info= pd.concat([self.task_info, new_task_info])
        self._current_task += 1
        return self.task_info, train_loader, val_loader, test_loader
    
    def get_new_task_info(self, xlen, ylen):
        cur_classes = self.curriculum[self._current_task]
        pnode_name = self.taxonomy_tree.get_parent(cur_classes[0])
        depth = self.taxonomy_tree.nodes.get(pnode_name).depth
        feature_size = set_feature_size(depth, self.connect_fs)
        if self.full_connect and len(self.task_info) > 0:
            ancestors = list(self.task_info['parent_node'])
        else: 
            ancestors = self.taxonomy_tree.get_ancestor_list(pnode_name)
        new_task_info = {}
        new_task_info["task_order"] = [self._current_task]
        new_task_info["child_nodes"] = [cur_classes]
        new_task_info["parent_node"] = [pnode_name]
        new_task_info["depth"] = [depth]
        new_task_info["feature_size"] = [feature_size]
        new_task_info["base_nf"] = [int(feature_size / 8)]
        new_task_info["ancestor_tasks"] = [ancestors]
        new_task_info["task_size"] = [len(cur_classes)]
        new_task_info["n_train_data"] = [xlen]
        new_task_info["n_test_data"] = [ylen]
        new_task_info["part_tree"] = [self.cur_part_tree]
        return pd.DataFrame(new_task_info)


    def _update_memory_for_new_task(self, labels):
        # delete the memory data with parent labels that have been replaced by finer labels
        if self.taxonomy:
            parent_names = set([self.taxonomy_tree.nodes[x].parent for x in labels])
            parent_labels = set([self.taxonomy_tree.nodes[x].label_index for x in parent_names])
            for lb in parent_labels:
                self.memory_dict.pop(lb, -1)

    def gen_memory_array_from_dict(self):
        data_memory = []
        target_memory = []
        for i in self.memory_dict:
            data_memory += [self.memory_dict[i]]
            target_memory += [i] * self.memory_dict[i].shape[0]
        return np.concatenate(data_memory), np.array(target_memory)

    def _get_cur_data_for_all_children(self):
        label_map_train = self._gen_label_map(self.curriculum[self._current_task])
        label_map_test = self._gen_label_map(list(np.concatenate(self.curriculum[:self._current_task + 1]).flatten()))
        x_train, y_train = self._select_from_idx(self.dict_train, label_map_train, train=True)
        x_val, y_val = self._select_from_idx(self.dict_val, label_map_test, train=False)
        x_test, y_test = self._select_from_idx(self.dict_test, label_map_test, train=False)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _gen_label_map(self, name_coarse):
        label_map = {}
        for nc in name_coarse:
            lc = self.taxonomy_tree.nodes.get(nc).label_index
            name_map_single = self.taxonomy_tree.get_finest(nc)
            # name_map_fine += name_map_single
            for nf in name_map_single:
                lf = self.taxonomy_tree.nodes.get(nf).label_index
                # position 0: coarse label; position 1: leaf node depth; position 2: parent node depth
                label_map[lf] = [lc, self.taxonomy_tree.nodes.get(nf).depth, self.taxonomy_tree.nodes.get(nc).depth]
        return label_map

    def _select_from_idx(self, data_dict, label_map, train=True):
        x_selected = np.empty([0, 32, 32, 3], dtype=np.uint8)
        if self.dataset_name in ['imagenet100', 'plankton29', 'iNat100']:
            x_selected = np.empty([0], dtype='<U74')
        y_selected = np.empty([0], dtype=np.uint8)
        if train:
            for lf in label_map:
                lfx_all = data_dict[lf]
                lfy_all = np.array([label_map[lf][0]] * len(lfx_all))  # position 0: coarse label
                lfx_used_idx = self.dict_train_used[lf]
                idx_available = np.where(lfx_used_idx == 0)[0]
                # position 1: leaf node depth; position 2: parent node depth
                # if coarse node, select by a fraction; if leaf node, select all remaining
                data_frac = self._sample_rate(label_map[lf][1], label_map[lf][2])

                if self._device.type == 'cuda':
                    if data_frac > 0 and self.taxonomy:
                        sel_ind = random.sample(list(idx_available), round(data_frac * len(lfx_all)))
                        
                    else:
                        sel_ind = idx_available
                        
                else:
                    if not self.debug:
                        sel_ind = idx_available
                    else:
                        sel_ind = random.sample(list(idx_available), 2)
                        

                x_selected = np.concatenate((x_selected, lfx_all[sel_ind]))
                y_selected = np.concatenate((y_selected, lfy_all[sel_ind]))
                self.dict_train_used[lf][sel_ind] = 1
                print(f'task{self._current_task}, finest class: {self.taxonomy_tree.nodes.get(self.taxonomy_tree.label2name[lf]).name}, sample rate: {data_frac}')
        else:
            for lf in label_map:
                lfx_all = data_dict[lf]
                lfy_all = np.array([label_map[lf][0]] * len(lfx_all))  # position 0: coarse label
                if not self.debug:
                    x_selected = np.concatenate((x_selected, lfx_all))
                    y_selected = np.concatenate((y_selected, lfy_all))
                else:

                    idx_available = np.arange(len(lfx_all))
                    sel_ind = list(idx_available)[:10]
                    x_selected = np.concatenate((x_selected, lfx_all[sel_ind]))
                    y_selected = np.concatenate((y_selected, lfy_all[sel_ind]))
        return x_selected, y_selected

    def _sample_rate(self, leaf_depth, parent_depth):
        assert leaf_depth >= parent_depth

        if isinstance(self._s_rate[0], float):
            # balance tree
            if leaf_depth == parent_depth:
                return -1
            else:
                return self._s_rate[parent_depth-1]
        else:
            # imbalance tree
            if leaf_depth == parent_depth:
                return -1
            else:
                return self._s_rate[leaf_depth-2][parent_depth-1]
        



    def _get_cur_step_data_for_raw_data(self, ):
        min_class = sum(self.increments[:self._current_task])
        max_class = sum(self.increments[:self._current_task + 1])

        x_train, y_train = self._select(self.data_train, self.targets_train, low_range=min_class, high_range=max_class)
        x_test, y_test = self._select(self.data_test, self.targets_test, low_range=0, high_range=max_class)
        return min_class, max_class, x_train, y_train, x_test, y_test

    # --------------------------------
    #           Data Setup
    # --------------------------------
    def _setup_data(self, dataset):
        self.data_train, self.targets_train = [], []
        self.data_test, self.targets_test = [], []
        self.data_val, self.targets_val = [], []
        self.dict_val, self.dict_train, self.dict_test = {}, {}, {}
        # current_class_idx = 0  # When using multiple datasets
        self.train_dataset = dataset(self.data_folder, train=True, device=self._device, debug=self.debug, df_name=self.df_name)
        self.test_dataset = dataset(self.data_folder, train=False, device=self._device, debug=self.debug, df_name=self.df_name)
        if self.dataset_name == 'imagenet100':
            train_idx = np.isin(self.train_dataset.targets, self.train_dataset.index_list)
            test_idx = np.isin(self.test_dataset.targets, self.test_dataset.index_list)
            self.train_dataset.data = self.train_dataset.data[train_idx]
            self.train_dataset.targets = self.train_dataset.targets[train_idx]
            self.test_dataset.data = self.test_dataset.data[test_idx]
            self.test_dataset.targets = self.test_dataset.targets[test_idx]
        self.n_tot_cls = self.train_dataset.n_cls  # number of classes in whole dataset

        self._setup_data_for_raw_data(self.train_dataset, self.test_dataset)
        # !list
        self.data_train = np.concatenate(self.data_train)
        self.targets_train = np.concatenate(self.targets_train)
        self.data_val = np.concatenate(self.data_val)
        self.targets_val = np.concatenate(self.targets_val)
        self.data_test = np.concatenate(self.data_test)
        self.targets_test = np.concatenate(self.targets_test)
        self.dict_train_used = {y: np.zeros(len(self.dict_train[y])) for y in self.dict_train}

    @staticmethod
    def get_true_targets(n_array):
        return 1

    def _setup_data_for_raw_data(self, train_dataset, test_dataset):
        x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
        x_val, y_val, x_train, y_train, dict_val, dict_train = \
            self._split_per_class(x_train, y_train, self.validation_split, self._seed)
        x_test, y_test = test_dataset.data, np.array(test_dataset.targets)
        _, _, x_test, y_test, _, dict_test = self._split_per_class(x_test, y_test, 0, self._seed)
        self.data_train.append(x_train)
        self.targets_train.append(y_train)
        self.data_val.append(x_val)
        self.targets_val.append(y_val)
        self.data_test.append(x_test)
        self.targets_test.append(y_test)
        self.dict_val.update(dict_val)
        self.dict_train.update(dict_train)
        self.dict_test.update(dict_test)

    def _setup_curriculum(self, dataset):
        # Get Class Order
        order = [i for i in range(len(np.unique(self.targets_train)))]
        if self.random_order:
            random.seed(self._seed)  # Ensure that following order is determined by seed:
            random.shuffle(order)
        elif dataset.class_order(self.trial_i) is not None:
            order = dataset.class_order(self.trial_i)
        self.curriculum = order

    @staticmethod
    def _split_per_class(x, y, validation_split=0.0, seed=0):
        """Splits train data for a subset of validation data.
        Split is done so that each class has equal amount of data.
        """
        # np.random.seed(seed)
        shuffled_indexes = np.random.permutation(x.shape[0])

        x = x[shuffled_indexes]
        y = y[shuffled_indexes]
        x_val, y_val = [], []
        x_train, y_train = [], []
        dict_val, dict_train = {}, {}

        for class_id in np.unique(y):
            class_indexes = np.where(y == class_id)[0]
            nb_val_elts = int(class_indexes.shape[0] * validation_split)

            val_indexes = class_indexes[:nb_val_elts]
            train_indexes = class_indexes[nb_val_elts:]

            x_val.append(x[val_indexes])
            y_val.append(y[val_indexes])
            x_train.append(x[train_indexes])
            y_train.append(y[train_indexes])
            dict_val[class_id] = x[val_indexes]
            dict_train[class_id] = x[train_indexes]

        # !list
        x_val, y_val = np.concatenate(x_val), np.concatenate(y_val)
        x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)

        return x_val, y_val, x_train, y_train, dict_val, dict_train

    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: order.index(x), y)))

    def _select(self, x, y, low_range=0, high_range=0):
        idxes = sorted(np.where(np.logical_and(y >= low_range, y < high_range))[0])
        if isinstance(x, list):
            selected_x = [x[idx] for idx in idxes]
        else:
            selected_x = x[idxes]
        return selected_x, y[idxes]

    # --------------------------------
    #           Get Loader
    # --------------------------------
    def get_datainc_loader(self, mode='train'):
        train_loader = self._get_loader(self.data_inc, self.targets_inc, mode=mode)
        return train_loader

    def get_custom_loader_from_memory(self, class_indexes, mode="test"):
        if not isinstance(class_indexes, list):
            class_indexes = [class_indexes]
        data, targets = [], []
        for class_index in class_indexes:
            class_data, class_targets = self._select(self.data_memory,
                                                     self.targets_memory,
                                                     low_range=class_index,
                                                     high_range=class_index + 1)
            data.append(class_data)
            targets.append(class_targets)

        data = np.concatenate(data)
        targets = np.concatenate(targets)

        return data, targets, self._get_loader(data, targets, shuffle=False, mode=mode)

    def _get_loader(self, x, y, share_memory=None, shuffle=True, mode="train", batch_size=None, resample=None):
        if x.shape[0] == 0 or len(y)==0:
            return None
        if "balanced" in mode:
            x, y = construct_balanced_subset(x, y)

        batch_size = batch_size if batch_size is not None else self._batch_size

        if "train" in mode:
            trsf = self.train_transforms
            resample_ = self._resampling if resample is None else True
            if resample_ is False:
                sampler = None
            else:
                sampler = get_weighted_random_sampler(y)

            shuffle = False if resample_ is True else True
        elif "test" in mode:
            trsf = self.test_transforms
            sampler = None
        elif mode == "flip":
            if "imagenet" in self.dataset_name:
                trsf = A.Compose([A.HorizontalFlip(p=1.0), *self.test_transforms.transforms])
            else:
                trsf = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), *self.test_transforms.transforms])
            sampler = None
        else:
            raise NotImplementedError("Unknown mode {}.".format(mode))
        # TODO: fix sampler
        dataset = DummyDataset(x, y, trsf, trsf_type=self.transform_type, share_memory_=share_memory,
                               dataset_name=self.dataset_name)
        if self.is_distributed and 'train' in mode:
            # TODO: fix the hardcode 4
            sampler = DistributedSampler(dataset, num_replicas=4, drop_last=True)
        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=(sampler is None),
                          num_workers=self._workers,
                          sampler=sampler,
                          pin_memory=False)

    def get_custom_loader(self, class_indexes, mode="test", data_source="train", imgs=None, tgts=None):
        """Returns a custom loader.

        :param class_indexes: A list of class indexes that we want.
        :param mode: Various mode for the transformations applied on it.
        :param data_source: Whether to fetch from the train, val, or test set.
        :return: The raw data and a loader.
        """
        if not isinstance(class_indexes, list):  # TODO: deprecated, should always give a list
            class_indexes = [class_indexes]

        if data_source == "train":
            x, y = self.data_inc, self.targets_inc
        elif data_source == "val":
            x, y = self.data_val, self.targets_val
        elif data_source == "test":
            x, y = self.data_test, self.targets_test
        elif data_source == 'specified' and imgs is not None and tgts is not None:
            x, y = imgs, tgts
        else:
            raise ValueError("Unknown data source <{}>.".format(data_source))

        data, targets = [], []
        for class_index in class_indexes:
            class_data, class_targets, = self._select(x, y, low_range=class_index, high_range=class_index + 1)
            data.append(class_data)
            targets.append(class_targets)

        data = np.concatenate(data)
        targets = np.concatenate(targets)

        return data, targets, self._get_loader(data, targets, shuffle=False, mode=mode)


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, trsf, trsf_type, share_memory_=None, dataset_name=None):
        self.dataset_name = dataset_name
        self.x, self.y = x, y
        self.trsf = trsf
        self.trsf_type = trsf_type
        self.manager = mp.Manager()
        self.buffer_size = 4000000
        if share_memory_ is None:
            if self.x.shape[0] > self.buffer_size:
                self.share_memory = self.manager.list([None for i in range(self.buffer_size)])
            else:
                self.share_memory = self.manager.list([None for i in range(len(x))])
        else:
            self.share_memory = share_memory_

    def __len__(self):
        if isinstance(self.x, list):
            return len(self.x)
        else:
            return self.x.shape[0]

    def __getitem__(self, idx):
        x, y, = self.x[idx], self.y[idx]

        if isinstance(x, np.ndarray):
            # assume cifar
            x = Image.fromarray(np.uint8(x))
            # add padding + resize here

        else:
            # Assume the dataset is ImageNet
            x_tmp = x
            x = cv2.imread(x)
            x = x[:, :, ::-1]

        if 'plankton' in self.dataset_name:
            # add padding   
            m, n, _ = x.shape
            offset = abs(n-m)
            left_offset, top_offset, right_offset, bottom_offset = 0, 0, 0, 0
            if m<n:
                top_offset = int(offset/2)
                bottom_offset = offset-top_offset
            elif m>n:
                left_offset = int(offset/2)
                right_offset = offset-left_offset
            else:
                pass
            padding = (left_offset, top_offset, right_offset, bottom_offset)

            x = F.pad(Image.fromarray(x), padding)

        if 'torch' in self.trsf_type:
            x = self.trsf(x)
            x = np.array(x)
        else:
            x = self.trsf(image=x)['image']
        return x, y


def tgt_to_aux_tgt(targets, targets_unique_list, device):
    aux_targets = targets.clone()
    t_list = torch.tensor(targets_unique_list)
    # set the labels that are not in current task (i.e. in memory) to 0
    old_idx = np.logical_not(np.isin(aux_targets.cpu(), t_list.cpu()))
    aux_targets[old_idx] = 0
    for index_i in range(len(t_list)):
        aux_targets[aux_targets == t_list[index_i]] = index_i + 1
    aux_targets = aux_targets.type(torch.LongTensor)

    if device.type == 'cuda':
        aux_targets = aux_targets.cuda()
    return aux_targets


def aux_tgt_to_tgt(aux_targets, targets_unique_list, device):
    new_idx_pos = (aux_targets != 0)
    new_idx = aux_targets[new_idx_pos]
    targets_ori = torch.tensor(targets_unique_list)
    if device.type == 'cuda':
        targets_ori = targets_ori.cuda()
    # map the non-zero targets into original ones and keep the zeros
    aux_targets[new_idx_pos] = targets_ori[new_idx - 1].long()
    return aux_targets


def tgt_to_tgt0(targets, leaf_id, device):
    targets0 = []
    for target_i in list(np.array(targets.cpu())):
        if target_i in leaf_id.keys():
            targets0.append(leaf_id[target_i])
    targets0 = torch.tensor(targets0)
    if device.type == 'cuda':
        return targets0.cuda()
    else:
        return targets0


def tgt0_to_tgt(targets0, leaf_id):
    targets = []
    leaf_inv = {leaf_id[i]: i for i in leaf_id}
    for i in range(targets0.shape[0]):
        targets.append(leaf_inv[int(targets0[i])])
    return np.array(targets)


def tgt_to_tgt0_no_tax(targets, targets_unique_list, device):
    targets0 = targets.clone()
    map_dict = {targets_unique_list[x]: x for x in range(len(targets_unique_list))}
    for k in map_dict.keys():
        targets0[targets == k] = map_dict[k]
    if device.type == 'cuda':
        return targets0.cuda()
    else:
        return targets0
