from numbers import Number

import torch
import numpy as np

import sys
sys.path.insert(0, '../..') # add parent-parent dir

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _check_inputs(size, mu, v):
    """helper function to ensure inputs are compatible"""
    if size is None and mu is None and v is None:
        raise ValueError("inputs can't all be None")
    elif size is not None:
        if mu is None:
            mu = torch.Tensor([0])
        if v is None:
            v = torch.Tensor([1])
        if isinstance(v, Number):
            v = torch.Tensor([v]).type_as(mu)
        v = v.expand(size)
        mu = mu.expand(size)
        return mu, v
    elif mu is not None and v is not None:
        if isinstance(v, Number):
            v = torch.Tensor([v]).type_as(mu)
        if v.size() != mu.size():
            v = v.expand(mu.size())
        return mu, v
    elif mu is not None:
        v = torch.Tensor([1]).type_as(mu).expand(mu.size())
        return mu, v
    elif v is not None:
        mu = torch.Tensor([0]).type_as(v).expand(v.size())
        return mu, v
    else:
        raise ValueError('Given invalid inputs: size={}, mu_logsigma={})'.format(size, (mu, v)))

def log_normal(x, mu=None, v=None, broadcast_size=False):
    """compute the log-pdf of a normal distribution with diagonal covariance"""
    if not broadcast_size:
        mu, v = _check_inputs(None, mu, v)
    else:
        mu, v = _check_inputs(x.size(), mu, v)
    assert mu.shape == v.shape
    return -0.5 * (np.log(2 * np.pi) + v.log() + (x - mu).pow(2).div(v))

class BetaScheduler:
    def __init__(self, start_beta=0.01, end_beta=1.0, warmup_steps=100):
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def get_beta(self):
        # if self.current_step >= self.warmup_steps:
        #     return self.end_beta
        
        # # Linear interpolation
        # fraction = self.current_step / self.warmup_steps
        # beta = self.start_beta + fraction * (self.end_beta - self.start_beta)
        # self.current_step += 1
        # return beta

        self.current_step += 1
        
        if self.current_step >= self.warmup_steps:
            return self.end_beta

        if self.current_step <= 0:
            return self.start_beta

        t = self.current_step / (self.warmup_steps - 1)
        return np.exp((self.current_step - self.warmup_steps) * 0.1) * self.end_beta