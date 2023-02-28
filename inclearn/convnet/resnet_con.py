import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import copy

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, remove_last_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.remove_last_relu = remove_last_relu

    def forward(self, x_list):
        ext_x, x = x_list
        # try: 
        #     cx = torch.cat((ext_x, x), dim=1)
        # except:
        #     cx = x
        cx = torch.cat((ext_x, x), dim=1)
        out = self.conv1(cx)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out += identity
        if not self.remove_last_relu:
            out = self.relu(out)
        return out


class ModuleGroup(nn.Module):
    def __init__(self, module_type, task_info, connect=True, zero_init_residual=True, bn_reset_running=True, bn_no_tracking=True):
        super(ModuleGroup, self).__init__()
        self.module_type = module_type
        self.module_list = nn.ModuleList()
        self.task_info = task_info
        self.connect = connect
        self.zero_init_residual = zero_init_residual
        self.bn_reset_running = bn_reset_running
        self.bn_no_tracking = bn_no_tracking

    def module_initialize(self, module, use='last'):
        assert isinstance(module, nn.Module)
        if self.module_type in ['Conv', 'BasicBlock']:
            if use == 'last' and len(self.module_list) > 0:  # use last module for initialization
                cur_fs = self.task_info.iloc[-1]['feature_size']
                prev_fs = self.task_info.iloc[-2]['feature_size']
                cur_dep = self.task_info.iloc[-1]['depth']
                prev_dep = self.task_info.iloc[-2]['depth']
                if ((self.connect) and (cur_dep == prev_dep)) or ((not self.connect) and (cur_fs == prev_fs)):
                    module.load_state_dict(self.module_list[-1].state_dict())
                    if self.bn_reset_running:
                        for m in module.modules():
                            if isinstance(m, nn.BatchNorm2d):
                                m.reset_running_stats()
                else:
                    print('Feature size mismatch, state dict not loaded!')

                if self.bn_no_tracking:
                    for m in self.module_list[-1].modules():
                        if isinstance(m, nn.BatchNorm2d):
                            m.track_running_stats = False

            else:  # the first module
                for m in module.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        # important
                        # m.weight = nn.Parameter(torch.ones_like(m.weight))

                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

                # Zero-initialize the last BN in each residual branch,
                # so that the residual branch starts with zeros, and each residual block behaves like an identity.
                # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
                if isinstance(module, BasicBlock) and self.zero_init_residual:
                    nn.init.constant_(module.bn2.weight, 0)


    def expand(self, mod_info):
        new_module = get_module(self.module_type, mod_info).cuda()
        self.module_initialize(new_module)
        self.module_list.append(new_module)

    def update_task(self, task_info):
        self.task_info = task_info

    def forward(self, x_list):
        assert len(self.module_list) == len(x_list)
        out = []
        for k in range(len(self.module_list)):
            module = self.module_list[k]
            
            if isinstance(module, BasicBlock):
                if self.connect:
                    x_anc = feature_cat(x_list, self.task_info, t_num=k)
                else:
                    x_anc = torch.tensor([]).cuda()  # switch
                x_needed = [x_anc, x_list[k]]

                #TODO: put module on cuda
                m1 = module.cuda()
                out.append(m1(x_needed))
            else:
                m1 = module.cuda()
                out.append(m1(x_list[k]))
        return out

    def set_train(self):
        for k in range(len(self.module_list) - 1):
            module = self.module_list[k]
            self.set_status(module, 'eval')
        module = self.module_list[-1]
        self.set_status(module, 'train')

    @staticmethod
    def set_status(module, status):
        assert status in ['train', 'eval']
        if status == 'train':
            for p in module.parameters():
                p.requires_grad = True

        else:
            for p in module.parameters():
                p.requires_grad = False


class MultiModuleGroup(nn.Module):
    '''This should correspond to _make_layer() function'''
    def __init__(self, module_type, length, task_info, connect=True, zero_init_residual=True, bn_reset_running=True, bn_no_tracking=True):
        super(MultiModuleGroup, self).__init__()
        self.module_type = module_type
        self.length = length
        self.task_info = task_info

        # use this since nn.Sequential.append() is not found
        groups = []      
        for _ in range(self.length):
            groups.append(ModuleGroup(self.module_type, task_info, connect, zero_init_residual, bn_reset_running, bn_no_tracking))
        self.groups = nn.Sequential(*groups)

    def expand(self, args):
        for k in range(self.length):
            self.groups[k].expand(args[k])

    def update_task(self, task_info):
        for group in self.groups:
            group.update_task(task_info)

    def forward(self, x):
        return self.groups(x)

    def set_train(self):
        for group in self.groups:
            group.set_train()


class ResConnect(nn.Module):
    def __init__(self, block, layer_num, at_info=None, dataset='cifar100', remove_last_relu=False, connect=True, zero_init_residual=True, bn_reset_running=True, bn_no_tracking=True):
        super(ResConnect, self).__init__()
        self.block = block
        self.layer_num = layer_num
        self.at_info = at_info
        self.dataset = dataset
        self.remove_last_relu = remove_last_relu
        self.connect = connect
        self.net_groups = nn.Sequential(
            ModuleGroup('Conv', self.at_info, connect, zero_init_residual, bn_reset_running, bn_no_tracking), 
            MultiModuleGroup(block, layer_num[0], self.at_info, connect, zero_init_residual, bn_reset_running, bn_no_tracking), 
            MultiModuleGroup(block, layer_num[1], self.at_info, connect, zero_init_residual, bn_reset_running, bn_no_tracking), 
            MultiModuleGroup(block, layer_num[2], self.at_info, connect, zero_init_residual, bn_reset_running, bn_no_tracking), 
            MultiModuleGroup(block, layer_num[3], self.at_info, connect, zero_init_residual, bn_reset_running, bn_no_tracking), 
            ModuleGroup('AvgPool', self.at_info, connect), 
        )

    def to_device(self, device):
        self.net_groups.to(device)

    def update_task_info(self, at_info):
        self.at_info = at_info
        for groups in self.net_groups:
            groups.update_task(at_info)
    
    def expand(self, base_nf):
        layer_args = [
            {'nf': base_nf, 'dataset': self.dataset},   # conv params
            self.multigroup_expand_args(base_nf, 0, self.layer_num[0]), 
            self.multigroup_expand_args(base_nf, 1, self.layer_num[1]), 
            self.multigroup_expand_args(base_nf, 2, self.layer_num[2]), 
            self.multigroup_expand_args(base_nf, 3, self.layer_num[3]), 
            {}  # last avgpool layer, no params
        ]
        assert len(layer_args) == len(self.net_groups)
        for k in range(len(layer_args)):
            groups = self.net_groups[k]
            args = layer_args[k]
            groups.expand(args)


    def get_anc_dims(self, t_num=-1):
        ancestors = self.at_info.iloc[t_num]["ancestor_tasks"]

        # problem
        if self.connect:
            anc_tasks = self.at_info.loc[self.at_info['parent_node'].isin(ancestors)]
            anc_nf = sum(anc_tasks["base_nf"])
        else:
            anc_nf = 0
        

        return anc_nf


    def multigroup_expand_args(self, base_nf, stage, blocks, t_num=-1, block_name='BasicBlock', rm_last_relu=False):
        assert stage in [0, 1, 2, 3]
        stage_prev = pow(2, max(stage-1, 0))
        stage_cur = pow(2, stage)

        out_nf_prev = base_nf * stage_prev
        out_nf = base_nf * stage_cur  # 64, 128, 256, 512, refers to out dim
        # problem
        anc_nf = self.get_anc_dims(t_num)
        in_nf_1 = (anc_nf + base_nf) * stage_prev
        in_nf_n = (anc_nf + base_nf) * stage_cur

        # inplanes = self.get_anc_dims(t_num)
        stride = 1 if stage == 0 else 2
        if block_name == 'BasicBlock':
            expansion = 1
        elif block_name == 'BottleNeck':
            expansion = 4

        args = []
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                conv1x1(out_nf_prev, out_nf, stride),
                nn.BatchNorm2d(out_nf),
            )

        basic_params = {"planes": out_nf, "stride": stride}
        layer_dsp, layer_normal, layer_rmrelu = [basic_params.copy(), basic_params.copy(), basic_params.copy()]
        layer_dsp.update({"inplanes": in_nf_1, "downsample": downsample, "rmrelu": False})
        layer_normal.update({"stride": 1, "inplanes": in_nf_n, "downsample": None, "rmrelu": False})
        layer_rmrelu.update({"stride": 1, "inplanes": in_nf_n, "downsample": None, "rmrelu": True})

        args.append(layer_dsp)
        for _ in range(1, blocks):
            args.append(layer_normal)
        if rm_last_relu:
            args[-1] = layer_rmrelu
        return args
        

    def forward(self, x):
        x = [x] * len(self.at_info)
        x = self.net_groups(x)
        return [y.view(y.size(0), -1) for y in x]

    def set_train(self):
        for groups in self.net_groups:
            groups.set_train()
            

def feature_cat(x, task_info, t_num=-1):
    x_list = torch.tensor([]).cuda()
    ancestors = task_info.iloc[t_num]["ancestor_tasks"]
    for k in range(len(x)):
        if task_info.iloc[k]["parent_node"] in ancestors:
            x_list = torch.cat((x_list, x[k]), dim=1)
    return x_list

def get_module(m_type, mod_info):
    if m_type == 'BasicBlock':
        inplanes = mod_info["inplanes"]
        planes = mod_info["planes"]
        stride = mod_info["stride"]
        downsample = mod_info["downsample"]
        rmrelu = mod_info["rmrelu"]
        return BasicBlock(inplanes, planes, stride, downsample, rmrelu)
    elif m_type == 'Conv':
        nf = mod_info["nf"]
        dataset = mod_info["dataset"]
        if dataset == 'cifar100':
            return nn.Sequential(nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=False),
                                           nn.BatchNorm2d(nf), nn.ReLU(inplace=True)) 
        elif dataset == 'imagenet100':
            return nn.Sequential(
                nn.Conv2d(3, nf, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(nf),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            raise ValueError('dataset not applicable!')
            
    elif m_type == 'AvgPool':
        return nn.AdaptiveAvgPool2d((1, 1))


def resconnect18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model."""
    model = ResConnect('BasicBlock', [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
