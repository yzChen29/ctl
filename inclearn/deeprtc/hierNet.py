import numpy as np
import torch
from torch import nn


class HierNet(nn.Module):
    """Module of hierarchical classifier"""
    def __init__(self, input_size, nodes, reuse=False):
        super(HierNet, self).__init__()
        self.input_size = input_size
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.cur_task = 1 + int((input_size-512) / 128)
        self.reuse_old = reuse
        for i in range(self.num_nodes):
            for j in range(self.cur_task):
                fc_name = self.nodes[i].name + f'_TF{j}'
                fs = 512 if j==0 else 128
                self.add_module(fc_name, nn.Linear(fs, len(self.nodes[i].children)))


    def forward(self, x, gate=None, pred=False, thres=0):
        if pred is False:
            # for training
            nout = []
            for i in range(self.num_nodes):
                prod = 0.0
                for j in range(self.cur_task):
                    fc_name = self.nodes[i].name + f'_TF{j}'
                    fc_layers = getattr(self, fc_name)
                    if j==0:
                        s_ind = 0
                        e_ind = 512
                    else:
                        s_ind = 512 + 128 * (j-1)
                        e_ind = 512 + 128 * j
                    prod += fc_layers(x[:, s_ind: e_ind])
                nout.append(prod / 5)

            outs = []
            out_masks = []
            # root node (no dependency to other nodes)
            # cw = torch.from_numpy(self.nodes[0].codeword).float().to(nout[0].device)
            # outs.append(torch.matmul(nout[0], cw) * gate[:, 0].view(-1, 1))
            # other internal nodes
            for i in range(self.num_nodes):
                cw = torch.from_numpy(self.nodes[i].codeword).float().to(nout[i].device)
                cond = self.nodes[i].cond
                cond_gate = torch.ones([x.size(0), 1]).to(nout[i].device)
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

    def reset_parameters(self, node2TFind_dict, feature_mode='full', ancestor_self_nodes_list=None):

        if self.reuse_old:

            if feature_mode=='full':
                j = self.cur_task - 1
                for i in range(self.num_nodes):
                    fc_name = self.nodes[i].name + f'_TF{j}'
                    self._modules[fc_name].reset_parameters()


            node2TFind_dict_inv = {node2TFind_dict[i]:i for i in node2TFind_dict}
            curr_node_name = node2TFind_dict_inv[len(node2TFind_dict_inv)-1]

            i = None

            for node_ind in range(len(self.nodes)):
                if self.nodes[node_ind].name == curr_node_name:
                    i = node_ind
                    break
            assert i != None


            if ancestor_self_nodes_list:

                for ancestor_j in ancestor_self_nodes_list:
                    fc_name = self.nodes[i].name + f'_TF{node2TFind_dict[ancestor_j]}'
                    self._modules[fc_name].reset_parameters()


            else:
                for j in range(self.num_nodes):
                    fs = 512 if j==0 else 128
                    fc_name = self.nodes[i].name + f'_TF{j}'
                    self._modules[fc_name].reset_parameters()


        else:

            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    # fs = 512 if j==0 else 128
                    fc_name = self.nodes[i].name + f'_TF{j}'
                    # self.add_module(fc_name, nn.Linear(fs, len(self.nodes[i].children)))
                    self._modules[fc_name].reset_parameters()


        return

def hiernet(**kwargs):
    model = HierNet(**kwargs)
    return model
