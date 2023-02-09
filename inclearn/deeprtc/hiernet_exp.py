import numpy as np
import torch
from torch import nn


class HierNetExp(nn.Module):
    """Module of hierarchical classifier"""
    def __init__(self, task_info=None, reuse=False):
        super(HierNetExp, self).__init__()
        self.task_info = task_info
        self.reuse = reuse

    def update_task_info(self, task_info):
        self.task_info = task_info

    def expand(self):
        prev_fs = self.task_info["feature_size"]
        c = 9
