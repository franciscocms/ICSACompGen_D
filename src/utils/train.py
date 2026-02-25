import torch
import numpy as np

import sys
sys.path.insert(0, '../..') # add parent-parent dir

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_cc_coefficient(epoch):
    """
    Epoch < 100: 0.0
    Epoch 100: Jumps to 0.01
    Epoch 100-300: Linear increase from 0.01 to 0.5
    Epoch > 300: 0.5
    """
    
    FINAL_VALUE = 0.1
    INIT_VALUE = 0.01
    
    if epoch < 100:
        return INIT_VALUE
    
    if 100 <= epoch <= 300:
        # Linear interpolation: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        start_val = INIT_VALUE
        end_val = FINAL_VALUE
        start_epoch = 100
        end_epoch = 300
        
        coeff = start_val + (epoch - start_epoch) * (end_val - start_val) / (end_epoch - start_epoch)
        return coeff
    
    return FINAL_VALUE

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