import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
from scipy.optimize import linear_sum_assignment
import sklearn

import sys
import logging
sys.path.insert(0, '../..') # add parent-parent dir

from src.main import params

logger = logging.getLogger(params["running_type"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_matching_cost(preds, labels):
  B = preds.size(0)
  assert preds.shape == labels.shape
  
  preds = preds.unsqueeze(2)
  labels_u = labels.unsqueeze(1)
  
  # Cost is MSE across all L causal factors
  # Result: [B, N, N]
  
  if len(preds.shape) == 4: # [B, N, 1, L]
    cost_matrix = torch.sum((preds - labels_u)**2, dim=-1)
  elif len(preds.shape) == 3: # [B, N, 1]
    cost_matrix = torch.square(preds - labels_u)
  cost_matrix_ = cost_matrix.detach().cpu().numpy()
  
  indices = np.array([linear_sum_assignment(p) for p in cost_matrix_])
  _, col_ind = indices[:, 0], indices[:, 1]  
  indices_ = indices.shape[2] * indices[:, 0] + indices[:, 1]

  losses = torch.gather(cost_matrix.flatten(1,2), 1, torch.from_numpy(indices_).to(device=cost_matrix.device))
  total_loss = losses.mean()

  # Reorder labels to match slot order
  matched_labels = labels[torch.arange(B, device=labels.device).view(B, 1),
                      col_ind]
  
  
  logger.info(labels.shape)
  logger.info(matched_labels.shape)

  return total_loss, matched_labels


def compute_kl_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    var_ratio = torch.exp(logvar_q - logvar_p)
    t2 = torch.square(mu_p - mu_q) / torch.exp(logvar_p)
    logvar_diff = logvar_p - logvar_q
    kl = 0.5 * torch.sum(var_ratio + t2 - 1.0 + logvar_diff, dim=-1)
    return torch.mean(kl)

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

def compute_prior_noise_loss(z, cp):
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

def hungarian_r2_score(pred, target):
    """
    pred:   [B, N] int predictions
    target: [B, N] int targets

    Returns: scalar R² averaged over batch.
    """

    B = pred.shape[0]

    # ---- 1. Batched cost matrix: cost[b, i, j] = (pred[b,i] - target[b,j])^2 ----
    cost = (pred.unsqueeze(2) - target.unsqueeze(1))**2      # [B, N, N]
    cost = cost.cpu().numpy()

    r2_scores = []
    all_target = []
    all_pred = []

    # ---- 2. Hungarian algorithm per batch (cannot be parallelized) ----
    for b in range(B):
        row_ind, col_ind = linear_sum_assignment(cost[b])

        # ---- 3. Apply matching ----
        matched_pred   = pred[b, row_ind]
        matched_target = target[b, col_ind]

        # ---- 4. Compute R² using sklearn ----
        # sklearn handles edge cases (constant targets, etc.)

        all_target.append(matched_target)
        all_pred.append(matched_pred)
        
        #r2_scores.append(torch.tensor(sklearn.metrics.r2_score(matched_target, matched_pred)))
    
    all_target = torch.flatten(torch.stack(all_target))
    all_pred = torch.flatten(torch.stack(all_pred))

    r2_scores = sklearn.metrics.r2_score(all_target.cpu().numpy(), all_pred.cpu().numpy())
    
    # ---- 5. Average over batch ----
    return r2_scores, all_target.cpu().numpy(), all_pred.cpu().numpy()

def hungarian_matching_loss_scalar(predictions, targets, loss_type='mse'):
    """
    Computes set-based loss for a single variable across N objects.
    
    Args:
        predictions: Tensor [B, N] - Sigmoid activated model outputs.
        targets: Tensor [B, N] - Ground truth continuous scalars.
        loss_type: 'mse' or 'huber'.
        
    Returns:
        loss: Mean scalar loss across the batch.
    """
    B, N = predictions.shape
    device = predictions.device
    
    # 1. Compute Pairwise Cost Matrix [B, N, N]
    # We use the absolute difference as the cost for matching
    # (B, N, 1) - (B, 1, N) -> (B, N, N)
    cost_matrix = torch.square(predictions.unsqueeze(2) - targets.unsqueeze(1))
    
    # 2. Hungarian Matching (Batch-wise)
    # We move to CPU because Scipy is a CPU-based library
    cost_matrix_cpu = cost_matrix.detach().cpu().numpy()
    
    # To keep this fast, we reorder the targets to match the prediction order
    matched_targets = torch.zeros_like(targets)
    
    for b in range(B):
        # row_ind: indices of predictions [0...N-1]
        # col_ind: the indices of targets that best match the predictions
        row_ind, col_ind = linear_sum_assignment(cost_matrix_cpu[b])
        matched_targets[b] = targets[b, col_ind]

    # 3. Final Loss Calculation
    # Now predictions[b, i] is aligned with matched_targets[b, i]
    if loss_type == 'huber':
        loss = F.smooth_l1_loss(predictions, matched_targets)
    else:
        loss = F.mse_loss(predictions, matched_targets)
        
    return loss
