import torch
import numpy as np

import sys
sys.path.insert(0, '../..') # add parent-parent dir

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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