import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def save_loss(LOSS_PATH, loss_dict):
  with open(LOSS_PATH + "/loss_dict.pkl", 'wb') as f:
    pkl.dump(loss_dict, f)

def save_loss_plot(loss_list, LOSS_PATH, resume_step, nsteps):
  step = int((resume_step + nsteps)/10)
  plt.figure()
  for i, loss in enumerate(loss_list):
      plt.plot(range(len(loss_list[loss])), loss_list[loss], label = loss)
  plt.xlabel('steps')
  plt.xticks(np.arange(0, len(loss_list[loss]), step))
  plt.ylabel('loss')
  plt.legend()
  plt.savefig(LOSS_PATH + '/loss_plot.png', bbox_inches = 'tight')
  plt.close()

def hungarian_loss(pred, target, loss_fn=F.smooth_l1_loss):
    
    """
    adapted from 'https://github.com/davzha/MESH/blob/main/losses.py'
    """
    
    pdist = loss_fn(
        pred.unsqueeze(1).expand(-1, target.size(1), -1, -1), 
        target.unsqueeze(2).expand(-1, -1, pred.size(1), -1),
        reduction='none').mean(3)

    pdist_ = pdist.detach().cpu().numpy()

    indices = np.array([linear_sum_assignment(p) for p in pdist_])

    indices_ = indices.shape[2] * indices[:, 0] + indices[:, 1]
    losses = torch.gather(pdist.flatten(1,2), 1, torch.from_numpy(indices_).to(device=pdist.device))
    total_loss = torch.mean(losses.sum(1))

    return total_loss, dict(indices=indices)