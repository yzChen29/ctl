import numpy as np
import torch
from torch import nn


class HierNetExp(nn.Module):
    """Module of hierarchical classifier"""
    def __init__(self, input_size=0, nodes=None, reuse=False, task_info=None, feature_mode='full'):
        super(HierNetExp, self).__init__()
        self.input_size = input_size
        self.nodes = nodes
        self.num_nodes = 0
        # self.cur_task = 1 + int((input_size-512) / 128)
        self.cur_task = 0
        self.reuse_old = reuse
        self.task_info = task_info
        self.feature_mode = feature_mode

    def update_task_info(self, nodes, task_info):
        assert len(nodes) == len(task_info)
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.task_info = task_info
        self.cur_task = len(task_info)

        

    def set_train(self):
        self.cls_freeze = []
        self.cls_zeros = []
        self.cls_train = []

        for i in range(self.num_nodes):
            use_features = self.filter_used_features(i)
            for j in range(self.cur_task):
                # fc_name = self.nodes[i].name + f'_TF{j}'
                ct_info = self.task_info.iloc[i]
                fc_name = ct_info['parent_node'] + f'_TF{j}'

                if self.feature_mode == 'add_zero_only_ance':
                    if i < j:
                        self.cls_zeros.append(fc_name)
                    elif i < self.num_nodes - 1:  
                        if j in use_features:
                            self.cls_freeze.append(fc_name)
                        else:
                            self.cls_zeros.append(fc_name)

                    elif i == self.num_nodes - 1:   # latest network, set to train
                        if j in use_features:
                            self.cls_train.append(fc_name)
                        else:
                            self.cls_zeros.append(fc_name)

                    else:
                        raise NotImplementedError('xxx')

                elif self.feature_mode ==  'full':
                    
                    if i == self.num_nodes - 1: 
                        self.cls_train.append(fc_name)
                    else:
                        if j == self.cur_task-1:
                            self.cls_train.append(fc_name)
                        else:
                            self.cls_freeze.append(fc_name)
                elif self.feature_mode ==  'add_zero_only_prev':
                    
                    if i != self.num_nodes - 1:
                        if i < j:
                            self.cls_zeros.append(fc_name)
                        else:
                            self.cls_freeze.append(fc_name)
                    else:
                        self.cls_train.append(fc_name)

                elif self.feature_mode == 'add_zero_only_self':
                    if i == j:
                        if i == self.num_nodes - 1:
                            self.cls_train.append(fc_name)
                        else:
                            self.cls_freeze.append(fc_name)
                    else:
                        self.cls_zeros.append(fc_name)
                else:
                    raise('not impletement')                    
                    
        
        for fc_name in self.cls_train:
            fc_layers = getattr(self, fc_name)
            for p in fc_layers.parameters():
                p.requires_grad = True
        
        for fc_name in self.cls_freeze:
            fc_layers = getattr(self, fc_name)
            for p in fc_layers.parameters():
                p.requires_grad = False

        for fc_name in self.cls_zeros:
            fc_layers = getattr(self, fc_name)
            fc_layers.weight.data = torch.zeros_like(fc_layers.weight.data)
            fc_layers.bias.data = torch.zeros_like(fc_layers.bias.data)
            for p in fc_layers.parameters():
                p.requires_grad = False
        
        c = 9


    def expand(self):
        assert self.cur_task == self.num_nodes
        # one new classifier for each previous task(node)
        # input feature size is determined by current task
        # output feature size is determined by number of child nodes of the corresponding parent node
        for i in range(self.num_nodes - 1):
            ct_info = self.task_info.iloc[i]
            in_fs = self.task_info.iloc[-1]['feature_size']
            out_fs = ct_info['task_size']
            fc_name = ct_info['parent_node'] + f'_TF{self.cur_task - 1}'
            self.add_module(fc_name, nn.Linear(in_fs, out_fs).cuda())
            

            #important
            # self._modules[fc_name].bias = torch.nn.Parameter(torch.zeros_like(self._modules[fc_name].bias))
            # # if self._modules[fc_name].weight.shape[0] == 20:
            # if self._modules[fc_name].weight.shape[0] == 4:
            
            #     a = np.load('classifier_para.npy')
            #     # self._modules[fc_name].weight = torch.nn.Parameter(torch.ones_like(self._modules[fc_name].weight))
            #     self._modules[fc_name].weight = torch.nn.Parameter(torch.from_numpy(a))
            # else:
            #     b = np.load('classifier_para_2.npy')
            #     self._modules[fc_name].weight = torch.nn.Parameter(torch.from_numpy(b))

            

        # n new classifiers for latest node
        # input feature size is determined by corresponding task
        # output feature size is determined by number of child nodes of current node
        for j in range(self.cur_task):
            ct_info = self.task_info.iloc[-1]
            in_fs = self.task_info.iloc[j]['feature_size']
            out_fs = ct_info['task_size']
            fc_name = ct_info['parent_node'] + f'_TF{j}'
            self.add_module(fc_name, nn.Linear(in_fs, out_fs).cuda())

            #important
            # self._modules[fc_name].bias = torch.nn.Parameter(torch.zeros_like(self._modules[fc_name].bias))
            # if self._modules[fc_name].weight.shape[0] == 20:
            #     a = np.load('classifier_para.npy')
            #     # self._modules[fc_name].weight = torch.nn.Parameter(torch.ones_like(self._modules[fc_name].weight))
            #     self._modules[fc_name].weight = torch.nn.Parameter(torch.from_numpy(a))
            # elif self._modules[fc_name].weight.shape[0] == 5:
            #     b = np.load('classifier_para_2.npy')
            #     self._modules[fc_name].weight = torch.nn.Parameter(torch.from_numpy(b))
            # elif self._modules[fc_name].weight.shape[0] == 4:
            #     c = np.load('classifier_para_shape4.npy')
            #     self._modules[fc_name].weight = torch.nn.Parameter(torch.from_numpy(c))
            # else:
            #     raise('classifier weight not match')




    def forward(self, x_list, gate=None, pred=False, thres=0):
        if pred is False:
            # for training

            # sanity check
            assert len(x_list) == self.num_nodes
            cls_dims = sum(self.task_info['feature_size'])
            fea_dims = sum([x.size(1) for x in x_list])
            assert cls_dims == fea_dims

            # compute outputs
            nout = []
            
            for i in range(self.num_nodes):
                prod = 0.0
                node_name = self.nodes[i].name
                task_num = self.task_info.loc[self.task_info['parent_node'] == node_name].loc[0, 'task_order']
                use_features = self.filter_used_features(task_num)  # list
                for j in use_features:
                    # ct_info = self.task_info.iloc[i]
                    

                    # fc_name = ct_info['parent_node'] + f'_TF{j}'
                    fc_name = node_name + f'_TF{j}'
                    fc_layers = getattr(self, fc_name)

                    
                    prod += fc_layers(x_list[j])
                    

                nout.append(prod / 5)

            outs = []
            out_masks = []
            # root node (no dependency to other nodes)
            # cw = torch.from_numpy(self.nodes[0].codeword).float().to(nout[0].device)
            # outs.append(torch.matmul(nout[0], cw) * gate[:, 0].view(-1, 1))
            # other internal nodes
            bs = x_list[0].size(0)
            for i in range(self.num_nodes):
                cw = torch.from_numpy(self.nodes[i].codeword).float().to(nout[i].device)
                cond = self.nodes[i].cond
                cond_gate = torch.ones([bs, 1]).to(nout[i].device)
                while cond:
                    parent, _ = cond
                    cond_gate = torch.mul(cond_gate, gate[:, parent].view(-1, 1))
                    cond = self.nodes[parent].cond
                gate = gate.to(cond_gate.device)
                outs.append(torch.matmul(nout[i], cw) * cond_gate * gate[:, i].view(-1, 1))
                mask = torch.clamp(torch.from_numpy(self.nodes[i].mask).float().to(nout[i].device), 1e-17, 1)
                out_masks.append(torch.log(mask) * (1 - gate[:, i].view(-1, 1)))


            self.output = torch.sum(torch.stack(outs[:]), 0)
            out_mask = torch.eq(torch.sum(torch.stack(out_masks), 0), 0).float()
            self.sfmx_base = torch.sum(torch.exp(self.output) * out_mask, 1)

            return self.output, nout, self.sfmx_base

        else:
            # for testing
            bs = x_list[0].size(0)
            if gate is None:
                gate = torch.zeros([x.size(0), self.num_nodes]).to(x.device)
                nout = []
                for i in range(self.num_nodes):
                    fc_layers = getattr(self, 'fc{}'.format(i))
                    nout.append(fc_layers(x))
                    cf = torch.max(torch.softmax(nout[i], dim=1), dim=1)[0]
                    gate[:, i] = torch.ge(cf, thres).float()
            else:
                nout = []
                for i in range(self.num_nodes):
                    fc_layers = getattr(self, 'fc{}'.format(i))
                    nout.append(fc_layers(x))

            outs = []
            # root node (no dependency to other nodes)
            cw = torch.from_numpy(self.nodes[0].codeword).float().to(nout[0].device)
            outs.append(torch.matmul(nout[0], cw) * gate[:, 0].view(-1, 1))
            # other internal nodes
            for i in range(1, self.num_nodes):
                cw = torch.from_numpy(self.nodes[i].codeword).float().to(nout[i].device)
                cond = self.nodes[i].cond
                cond_gate = torch.ones([x.size(0), 1]).to(nout[i].device)
                while cond:
                    parent, _ = cond
                    cond_gate = torch.mul(cond_gate, gate[:, parent].view(-1, 1))
                    cond = self.nodes[parent].cond
                outs.append(torch.matmul(nout[i], cw) * cond_gate * gate[:, i].view(-1, 1))

            self.output = torch.sum(torch.stack(outs), 0)
            return self.output, nout

    def filter_used_features(self, t_num):
        if self.feature_mode == 'full':
            use_tasks = list(range(t_num+1))
            # raise('Error')
        elif self.feature_mode == 'add_zero_only_prev':
            use_tasks = list(range(self.task_info.iloc[t_num]['task_order']+1))
        elif self.feature_mode == 'add_zero_only_ance':
            ancestors = self.task_info.iloc[t_num]["ancestor_tasks"]
            try:
                anc_tasks = self.task_info.loc[self.task_info['parent_node'].isin(ancestors)]
                use_tasks = list(anc_tasks['task_order'])
                use_tasks.append(t_num)
            except:
                use_tasks = [t_num]  
        elif self.feature_mode == 'add_zero_only_self':
            use_tasks = [t_num]
        return use_tasks

    def reset_parameters(self):
        for fc_name in self.cls_train:
            fc_layers = getattr(self, fc_name)
            fc_layers.reset_parameters()
            # for p in fc_layers.parameters():
            #     p.requires_grad = True 

            # #important
            # self._modules[fc_name].bias = torch.nn.Parameter(torch.zeros_like(self._modules[fc_name].bias))
            # b = np.load('classifier_para_2.npy')
            # # self._modules[fc_name].weight = torch.nn.Parameter(torch.ones_like(self._modules[fc_name].weight))
            # self._modules[fc_name].weight = torch.nn.Parameter(torch.from_numpy(b))


        return

def hiernet(**kwargs):
    model = HierNetExp(**kwargs)
    return model

