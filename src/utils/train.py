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
    
def get_scheduled_coefficient(epoch, 
                              start_epoch=100, 
                              end_epoch=200, 
                              initial_val=1.0, 
                              final_val=3.0):
    """
    Returns a coefficient that:
    - Is initial_val until start_epoch.
    - Linearly interpolates to final_val between start_epoch and end_epoch.
    - Stays at final_val after end_epoch.
    """
    if epoch < start_epoch:
        return initial_val
    
    if epoch >= end_epoch:
        return final_val
    
    # Linear interpolation formula: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    progress = (epoch - start_epoch) / (end_epoch - start_epoch)
    coeff = initial_val + progress * (final_val - initial_val)
    
    return coeff