import copy
import torch
from torch import nn
import torch.nn.functional as F

from inclearn.tools import factory
from inclearn.convnet.imbalance import BiC, WA
from inclearn.convnet.classifier import CosineClassifier, RealTaxonomicClassifier
from inclearn.convnet.resnet_con import resconnect18
from inclearn.deeprtc import get_model
from inclearn.deeprtc.hiernet_exp import HierNetExp
from inclearn.deeprtc.pivot import Pivot


class TaxConnectionDer(nn.Module):  # used in incmodel.py
    def __init__(self, convnet_type, cfg, nf=64, use_bias=False, init="kaiming", device=None, dataset="cifar100",
                 at_info={}, ct_info={}, feature_mode='full', connect=True):
        super(TaxConnectionDer, self).__init__()
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        # self.exp_module = resconnect18()
        # self.exp_classifier = HierNetExp(reuse=cfg['reuse_oldfc'])
        self.dataset = dataset
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.der = cfg['der']
        self.use_aux_cls = cfg['use_aux_cls']
        self.aux_nplus1 = cfg['aux_n+1']
        self.reuse_oldfc = cfg['reuse_oldfc']
        self.module_cls = cfg['model_cls']
        self.module_pivot = cfg['model_pivot']
        self.at_info = at_info
        self.ct_info = ct_info
        self.connect = connect

        if self.der:
            print("Enable dynamical representation expansion!")
            self.out_dim = 0
            
            self.exp_module = resconnect18(False, dataset=dataset, connect=cfg['use_connection'], zero_init_residual=cfg['zero_init_residual'], bn_reset_running=cfg['bn_reset_running'], bn_no_tracking=cfg['bn_no_tracking'], 
                                           full_connect=cfg['full_connect'])
            self.exp_classifier = HierNetExp(reuse=cfg['reuse_oldfc'], 
                                                 feature_mode=feature_mode)
        else:
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.out_dim = self.convnet.out_dim
        self.taxonomy = cfg['taxonomy']
        self.classifier = None
        self.aux_classifier = None

        self.n_classes = 0
        self.device = device
        self.feature_mode = feature_mode
        self.node2TFind_dict = {}
        self.ancestor_self_nodes_list = None
        # self.device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" , index = 0)

        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == "bic":
                self.postprocessor = BiC(cfg['postprocessor']["lr"], cfg['postprocessor']["scheduling"],
                                         cfg['postprocessor']["lr_decay_factor"], cfg['postprocessor']["weight_decay"],
                                         cfg['postprocessor']["batch_size"], cfg['postprocessor']["epochs"])
            elif cfg['postprocessor']['type'].lower() == "wa":
                self.postprocessor = WA()
        else:
            self.postprocessor = None

        self.to(self.device)

    def forward(self, x):
        if self.classifier is None and self.exp_classifier is None:
            raise Exception("Add some classes before training.")

        # get feature
        if self.der:
            # features = [convnet(x) for convnet in self.convnets]
            features = self.exp_module(x)
            # fe = features[0]
            # for k in range(1, len(features)):
            #     fe = torch.cat((fe, features[k]), dim=1)
            # features = torch.squeeze(fe)
        else:
            features = self.convnet(x)

        # classification
        if self.taxonomy is not None:
            gate = self.model_pivot(torch.ones([x.size(0), len(self.used_nodes)]))
            # gate[:, 0] = 1
            # print(features)
            # output, nout, sfmx_base = self.classifier(x=features, gate=gate)
            output, nout, sfmx_base = self.exp_classifier(x_list=features, gate=gate)
            # logits = self.classifier(features)
        else:
            output = self.classifier(features)
            nout, sfmx_base = None, None

        if self.use_aux_cls and self.current_task > 0:


            #switch aux
            # use_features = []
            # ancestor_name = self.ct_info['ancestor_tasks']
            # for i in range(self.at_info.shape[0]):
            #     if self.at_info.iloc[i]['parent_node'] in ancestor_name:
            #         use_features.append(features[i])
            # use_features.append(features[self.ct_info['task_order']])   
            # aux_logits = self.aux_classifier(torch.cat(use_features, dim=1))

            aux_logits = self.aux_classifier(features[self.ct_info['task_order']])
        else:
            aux_logits = None
        return {'feature': features, 'output': output, 'nout': nout, 'sfmx_base': sfmx_base, 'aux_logit': aux_logits}

    @property
    def features_dim(self):
        if self.der:
            # return self.out_dim * len(self.convnets)
            return self.out_dim
        else:
            return self.out_dim

    def set_train(self):
        self.exp_module.set_train()
        self.exp_classifier.set_train()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    
    def new_task(self, at_info, feature_mode):
        # update info
        self.at_info = at_info
        self.current_task = len(at_info) - 1
        self.ct_info = at_info.iloc[self.current_task]

        expand_info = at_info.loc[:, at_info.columns != 'part_tree']
        self._update_tree_info()
        self._gen_pivot()

        self.exp_module.update_task_info(expand_info)
        self.exp_classifier.update_task_info(self.used_nodes, expand_info)
        self.out_dim = sum(at_info["feature_size"])

        # add classes
        if self.der:
            n_classes = self.ct_info["task_size"]

            # expand network part
            base_nf = int(self.ct_info['feature_size'] / 8)
            self.exp_module.expand(base_nf)

            # expand classifier part
            self.exp_classifier.expand()
            # self._add_classes_multi_fc(feature_mode)

            if self.aux_nplus1:
            # aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
                ct_fs = self.ct_info['feature_size']  # feature size of current task
                ct_nclass = self.ct_info["task_size"]  # number of classes for current task

                #switch aux
                # ancestor_name = self.ct_info['ancestor_tasks']
                # for i in range(self.at_info.shape[0]):
                #     if self.at_info.iloc[i]['parent_node'] in ancestor_name:
                #         ct_fs += self.at_info.iloc[i]['feature_size']

                aux_fc = nn.Linear(ct_fs, ct_nclass + 1, bias=self.use_bias).to(self.device)
                if self.init == "kaiming":
                    nn.init.kaiming_normal_(aux_fc.weight, nonlinearity="linear")
                if self.use_bias:
                    nn.init.constant_(aux_fc.bias, 0.0)
            else:
                aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
            del self.aux_classifier
            self.aux_classifier = aux_fc
        else:
            self._add_classes_single_fc()


    def _add_classes_multi_fc(self, feature_mode='full'):
        n_classes = self.ct_info["task_size"]
        if self.taxonomy:
            all_classes = len(self.ct_info['part_tree'].leaf_nodes)
        else:
            all_classes = self.n_classes + n_classes

        # expand network part
        base_nf = self.ct_info['base_nf']
        self.exp_module.expand(base_nf)

        # expand classifier part
        # new_clf = self._gen_classifier(self.out_dim * len(self.convnets), all_classes)
        self.exp_classifier.expand(base_nf)
    
        new_clf = self._gen_classifier(self.out_dim, all_classes)
        if self.taxonomy:
            if self.classifier is None:
                self.node2TFind_dict['root'] = self.current_task

            if self.classifier is not None and self.reuse_oldfc:
                old_clf = self.classifier

                if feature_mode == 'full':
                    j_task_range = old_clf.cur_task
                elif 'add_zero' in feature_mode:
                    j_task_range = new_clf.cur_task
                else:
                    raise('feature_mode not implement')

                for k in range(old_clf.num_nodes):
                    for j in range(j_task_range):
                        fc_name = old_clf.nodes[k].name + f'_TF{j}'
                        fc_old = getattr(old_clf, fc_name, None)
                        fc_new = getattr(new_clf, fc_name, None)
                        if feature_mode == 'full':
                            assert fc_old is not None
                        assert fc_new is not None
                        # weight = copy.deepcopy(fc_old.weight.data)
                        if feature_mode == 'full':
                            fc_new.weight.data = copy.deepcopy(fc_old.weight.data)
                            fc_new.bias.data = copy.deepcopy(fc_old.bias.data)
                        elif 'add_zero' in feature_mode:
                            if fc_old is not None:
                                fc_new.weight.data = copy.deepcopy(fc_old.weight.data)
                                fc_new.bias.data = copy.deepcopy(fc_old.bias.data)
                            else:
                                fc_new.weight.data = torch.zeros_like(fc_new.weight.data)
                                fc_new.bias.data = torch.zeros_like(fc_new.bias.data)

                        for param in fc_new.parameters():
                            param.requires_grad = False
                        fc_new.eval()

                curr_nodes_name_list = []
                for i in [j.name for j in new_clf.nodes.values()]:
                    if i not in [j.name for j in old_clf.nodes.values()]:
                        curr_nodes_name_list.append(i)
                assert len(curr_nodes_name_list) == 1

                curr_node_name = curr_nodes_name_list[0]

                self.node2TFind_dict[curr_node_name] = self.current_task

            if self.classifier is not None and self.feature_mode=='add_zero_only_ancestor_fea':

                for j in range(self.current_task):
                    cur_tree = self.ct_info["part_tree"]
                    ancestor_nodes_list = cur_tree.get_ancestor_list(new_clf.nodes[len(new_clf.nodes)-1].name)
                    ancestor_self_nodes_list = ancestor_nodes_list + [curr_node_name]
                    self.ancestor_self_nodes_list = ancestor_self_nodes_list

                    useless_TF_list = [self.node2TFind_dict[i] for i in ancestor_self_nodes_list]
                    if j not in useless_TF_list:
                        fc_name = curr_node_name + f'_TF{j}'
                        fc_useless = getattr(new_clf, fc_name, None)
                        fc_useless.weight.data = torch.zeros_like(fc_useless.weight.data)
                        fc_useless.bias.data = torch.zeros_like(fc_useless.bias.data)
                        for param in fc_useless.parameters():
                            param.requires_grad = False

            for k in range(new_clf.num_nodes):
                for j in range(new_clf.cur_task):
                    fc_name = new_clf.nodes[k].name + f'_TF{j}'
                    fc_new = getattr(new_clf, fc_name, None)

        else:
            if self.classifier is not None and self.reuse_oldfc:
                weight = copy.deepcopy(self.classifier.weight.data)
                new_clf.weight.data[:self.n_classes, :self.out_dim * (len(self.convnets) - 1)] = weight

        del self.classifier
        self.classifier = new_clf

        if self.aux_nplus1:
            # aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
            ct_fs = self.ct_info['feature_size']  # feature size of current task
            aux_fc = nn.Linear(ct_fs, n_classes + 1, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(aux_fc.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(aux_fc.bias, 0.0)
        else:
            aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
        del self.aux_classifier
        self.aux_classifier = aux_fc

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.taxonomy is not None:
            # self._update_tree_info()
            if self.taxonomy == 'rtc':
                # classifier
                # used_nodes = setup_tree(self.current_task, self.current_tax_tree)
                model_dict = {'arch': self.module_cls, 'feat_size': in_features}
                if self.device.type == 'cuda':
                    model_cls = get_model(model_dict, self.used_nodes, self.reuse_oldfc).cuda()
                    # model_cls = nn.DataParallel(model_cls, device_ids=range(torch.cuda.device_count()))
                else:
                    model_cls = get_model(model_dict, self.used_nodes, self.reuse_oldfc)
                    # model_cls = nn.DataParallel(model_cls, device_ids=range(0))
                classifier = model_cls

                # pivot
                self._gen_pivot()
            else:
                raise NotImplementedError('')
        else:
            if self.weight_normalization:
                classifier = CosineClassifier(in_features, n_classes).to(self.device)
            else:
                classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
                if self.init == "kaiming":
                    nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
                if self.use_bias:
                    nn.init.constant_(classifier.bias, 0.0)

        return classifier

    def _gen_pivot(self):
        if self.device.type == 'cuda':
            model_pivot = get_model(self.module_pivot).cuda()
            model_pivot = nn.DataParallel(model_pivot, device_ids=range(torch.cuda.device_count()))
        else:
            model_pivot = get_model(self.module_pivot)
        self.model_pivot = model_pivot

    def _update_tree_info(self):
        used_nodes, leaf_id, node_labels = self.ct_info['part_tree'].prepro()
        self.used_nodes = used_nodes
        self.node_labels = node_labels
        self.leaf_id = leaf_id


