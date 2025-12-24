import torch

import sys
import logging
sys.path.insert(0, '../..') # add parent-parent dir

from src.main import params

logger = logging.getLogger(params["running_type"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_kl_loss(mu, logvars):
  assert len(mu.shape) == len(logvars.shape) == 2
  kl_div = torch.mean(-0.5 * torch.sum(1 + logvars - mu**2 - torch.exp(logvars), dim=-1))
  return kl_div

def compute_log_det(L_slopes):
  log_det = 0.
  for i in range(len(params["latent_vars"])):
    s = L_slopes[:, :, i, :]
    if params["normalized_logdet"]: log_det += torch.sum(s, dim=-1)/s.size(-1)
    else: log_det += torch.sum(s, dim=-1)
  log_det = torch.mean(log_det)
  return log_det

def compute_prior_z_loss(z, cp):
  assert z.shape == cp.shape

  B, n_s, _, _ = z.shape
  z = z.reshape(B*n_s, *z.shape[2:]) # [B*N, L, d]
  cp = cp.reshape(B*n_s, *cp.shape[2:])
  
  var = torch.ones(cp.size(-1)).to(device)
  causal_loss = 0.
  for i in range(len(params["latent_vars"])):
    temp = torch.sum(0.5 * (torch.log(var) - torch.log(var) + var/var + (z[:, i, :] - cp[:, i, :]).pow(2)/var - 1), dim=-1)    
    causal_loss += torch.mean(temp)
  return causal_loss
