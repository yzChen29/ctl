"""
@Author : Yan Shipeng, Xie Jiangwei
@Contact: yanshp@shanghaitech.edu.cn, xiejw@shanghaitech.edu.cn
"""

import sys
import os
import os.path as osp
import copy
import time
import torch.multiprocessing as mp
import torch.distributed as dist
from pathlib import Path
import numpy as np
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter
from copy import deepcopy
import shutil

repo_name = 'ctl'
base_dir = osp.realpath(".")[:osp.realpath(".").index(repo_name) + len(repo_name)]
sys.path.append(base_dir)

from sacred import Experiment

ex = Experiment(base_dir=base_dir, save_git_info=False)

import torch

from inclearn.tools import factory, results_utils, utils
from inclearn.learn.pretrain import pretrain
from inclearn.tools.metrics import IncConfusionMeter


def initialization(config, seed, mode, exp_id):
    # Add it if your input size is fixed
    # ref: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    torch.backends.cudnn.benchmark = True  # This will result in non-deterministic results.
    # ex.captured_out_filter = lambda text: 'Output capturing turned off.'
    cfg = edict(config)
    utils.set_seed(cfg['seed'])
    utils.set_save_paths(cfg, mode)

    if exp_id is None:
        exp_id = -1
    #     cfg.exp.savedir = "./logs"
    logger = utils.make_logger(f"{mode}", savedir=cfg['sp']['log'])

    # Tensorboard
    # exp_name = f'{exp_id}_{cfg["exp"]["name"]}' if exp_id is not None else f'../inbox/{cfg["exp"]["name"]}'
    # tensorboard_dir = cfg["exp"]["tensorboard_dir"] + f"/{exp_name}"

    tensorboard = SummaryWriter(cfg['sp']['tensorboard'])
    # shutil.copyfile('./configs/ctl2_gpu.yaml', f"{cfg['sp']['log']}/ctl2_gpu.yaml")
    return cfg, logger, tensorboard


def _train(rank, cfg, world_size, logger=None):
    if cfg["is_distributed"]:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        print('start process', rank)
        torch.cuda.set_device(rank)
        cfg["rank"] = rank
        cfg["world_size"] = world_size
        logger = factory.MyCustomLoader(rank=rank)
    # else:
    #     logger = factory.MyCustomLoader(rank=0)
    inc_dataset = factory.get_data(cfg)
    model = factory.get_model(cfg, logger, inc_dataset)

    logger.info("curriculum")
    logger.info(inc_dataset.curriculum)

    for task_i in range(inc_dataset.n_tasks):
    # for task_i in range(1):
        model.before_task()
        enforce_decouple = False


        if task_i >= cfg['retrain_from_task']:
            model.train_task()
        elif task_i == cfg['retrain_from_task']-1:

            if task_i == 0:
#                 state_dict = torch.load(f"result/{cfg['exp']['load_model_name']}/train/ckpts/step0.ckpt")
                state_dict = torch.load(f"{cfg['exp']['load_model_name']}/train/ckpts/step0.ckpt")

            else:
#                 load_path = f"result/{cfg['exp']['load_model_name']}/train/ckpts"
                load_path = f"{cfg['exp']['load_model_name']}/train/ckpts"



                if os.path.exists(f'{load_path}/decouple_step{task_i}.ckpt'):
                    state_dict = torch.load(f'{load_path}/decouple_step{task_i}.ckpt')
                else:
                    state_dict = torch.load(f'{load_path}/step{task_i}.ckpt')
                    enforce_decouple = True
            model._parallel_network.load_state_dict(state_dict)
        else:
            print(f'passing task {task_i}')

        if task_i >= cfg['retrain_from_task'] - 1:

            if cfg['device'].type == 'cuda':
                model.eval_task(model._cur_val_loader, save_path=model.sp['exp'], name='eval_before_decouple', save_option={
                    "acc_details": True,
                    "acc_aux_details": True,
                    "preds_details": True,
                    "preds_aux_details": True
                })

        # for debug
        # if cfg['device'].type == 'cpu':
        #     model.eval_task(model._cur_val_loader, save_path=model.sp['exp'], name='eval_before_decouple', save_option={
        #         "acc_details": True,
        #         "acc_aux_details": True,
        #         "preds_details": True,
        #         "preds_aux_details": True
        #     })

        model.after_task(inc_dataset, enforce_decouple=enforce_decouple)

        if task_i >= cfg['retrain_from_task'] - 1:
            if cfg['device'].type == 'cuda':
                model.eval_task(model._cur_val_loader, save_path=model.sp['exp'], name='eval_after_decouple', save_option={
                    "acc_details": True,
                    "acc_aux_details": True,
                    "preds_details": True,
                    "preds_aux_details": True
                })


    #         if cfg['device'].type == 'cuda' and cfg['dataset'] == 'cifar100':
    #             model.eval_task(model._cur_test_loader, save_path=model.sp['exp'], name='test', save_option={
    #                 "acc_details": True,
    #                 "acc_aux_details": True,
    #                 "preds_details": True,
    #                 "preds_aux_details": True
    #             })


@ex.command
def train(_run, _rnd, _seed):

    try:
        print('before multiprocess')
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        cfg, ex.logger, tensorboard = initialization(_run.config, _seed, "train", _run._id)
        ex.logger.info(cfg)

        # adjust config
        if cfg["device_auto_detect"]:
            cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            factory.set_device(cfg)
        if cfg["device"].type == 'cuda' and 'imagenet' in cfg["dataset"]:
            cfg.data_folder = '/datasets'
        else:
            cfg.data_folder = osp.join(base_dir, "data")

        start_time = time.time()
        if cfg["is_distributed"]:
            gpu_num = torch.cuda.device_count()
            mp.spawn(_train, args=(cfg, gpu_num), nprocs=gpu_num, join=True)
        else:
            _train(0, cfg, 1, ex.logger)

        ex.logger.info("Training finished in {}s.".format(int(time.time() - start_time)))
        # with open(cfg["exp"]["name"] + '/delete_warning.txt', 'a') as dw:
        #     dw.write('This is a fully conducted experiment without errors and interruptions. Please be careful as deleting'
        #              ' it may lose important data and results. See log file for configuration details.')

    except Exception as e:
        import traceback
        traceback.print_exc(file=open(
            '/datasets/imagenet100_results/terminal_log.txt','a'))
        print('Error Message', e)
        print('\n\n\n\n')
        raise('Error')


def do_pretrain(cfg, ex, model, device, train_loader, test_loader):
    if not os.path.exists(osp.join(ex.base_dir, 'pretrain/')):
        os.makedirs(osp.join(ex.base_dir, 'pretrain/'))
    model_path = osp.join(
        ex.base_dir,
        "pretrain/{}_{}_cosine_{}_multi_{}_aux{}_nplus1_{}_{}_trial_{}_{}_seed_{}_start_{}_epoch_{}.pth".format(
            cfg["model"],
            cfg["convnet"],
            cfg["weight_normalization"],
            cfg["der"],
            cfg["use_aux_cls"],
            cfg["aux_n+1"],
            cfg["dataset"],
            cfg["trial"],
            cfg["train_head"],
            cfg['seed'],
            cfg["start_class"],
            cfg["pretrain"]["epochs"],
        ),
    )
    if osp.exists(model_path):
        print("Load pretrain model")
        if hasattr(model._network, "module"):
            model._network.module.load_state_dict(torch.load(model_path))
        else:
            model._network.load_state_dict(torch.load(model_path))
    else:
        pretrain(cfg, ex, model, device, train_loader, test_loader, model_path)


@ex.command
def test(_run, _rnd, _seed):
    cfg, ex.logger, tensorboard = initialization(_run.config, _seed, "test", _run._id)
    cfg.data_folder = osp.join(base_dir, "data")
    if cfg["device_auto_detect"]:
        cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    else:
        factory.set_device(cfg)
    cfg["exp"]["mode_train"] = False
    ex.logger.info(cfg)

    inc_dataset = factory.get_data(cfg)
    model = factory.get_model(cfg, _run, ex, tensorboard, inc_dataset)
    # model._network.task_size = cfg.increment

    test_results = results_utils.get_template_results(cfg)
    for task_i in range(inc_dataset.n_tasks):
        model.before_task()
        if task_i == 20:
            model_path = 'results/' + cfg['exp']['load_model_name'] + f'/train/ckpts/decouple_step{task_i}.ckpt'
            state_dict = torch.load(model_path)
            # state_dict = torch.load(f'../../../cyz_codes/ctl/codes/base/ckpts/step{task_i}.ckpt')
            model._parallel_network.load_state_dict(state_dict)
            model.eval()
            model.eval_task(model._cur_test_loader, save_path=model.sp['exp'], name=cfg['exp']['name'], save_option={
                "acc_details": True,
                "acc_aux_details": True,
                "preds_details": True,
                "preds_aux_details": True
            })
            # model.save_acc_detail_info('test')


if __name__ == "__main__":
    # ex.add_config('./codes/base/configs/default.yaml')
    ex.add_config("./codes/base/configs/ctl2_gpu_cifar100.yaml")
    ex.run_commandline()


