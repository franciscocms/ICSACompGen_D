import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import shutil
from scipy.optimize import linear_sum_assignment
import sklearn
from sklearn.decomposition import PCA

# add project path to sys to import relative modules
import sys
sys.path.insert(0, '..') # add parent dir

from models.guide import InvSlotAttentionGuide
from main import params
from datasets.data_utils import MyDataset, ProbeDataset
from utils.guide import LinearProbe

import logging
logfile_name = f"eval-{params['jobID']}.log"
logger = logging.getLogger(params["running_type"])
logger.setLevel(logging.INFO)
fh = logging.FileHandler(logfile_name, mode='w')
logger.addHandler(fh)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_device(device)
main_dir = os.path.abspath(__file__+'/../')

def hungarian_r2_score(pred, target):
    """
    pred:   [B, N] int predictions
    target: [B, N] int targets

    Returns: scalar R² averaged over batch.
    """

    B, N = pred.shape

    # ---- 1. Batched cost matrix: cost[b, i, j] = (pred[b,i] - target[b,j])^2 ----
    diff = pred.unsqueeze(2) - target.unsqueeze(1)      # [B, N, N]
    cost = (diff * diff).cpu()                          # convert once for scipy

    r2_scores = []

    # ---- 2. Hungarian algorithm per batch (cannot be parallelized) ----
    for b in range(B):
        row_ind, col_ind = linear_sum_assignment(cost[b].numpy())

        # ---- 3. Apply matching ----
        matched_pred   = pred[b, row_ind].cpu().numpy()
        matched_target = target[b, col_ind].cpu().numpy()

        # ---- 4. Compute R² using sklearn ----
        # sklearn handles edge cases (constant targets, etc.)
        r2_scores.append(sklearn.metrics.r2_score(matched_target, matched_pred))

    # ---- 5. Average over batch ----
    return torch.tensor(r2_scores) #/ B

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
    cost_matrix = torch.abs(predictions.unsqueeze(2) - targets.unsqueeze(1))
    
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

def main():    

    assert params["running_type"] == "eval"

    logger.info(device)

    threshold = [-1.]
    
    A = torch.tensor([
                    [0., 0., 0., 0.],
                    [1., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                ])
    guide = InvSlotAttentionGuide(resolution = params['resolution'],
                              num_iterations = 3,
                              slot_dim = params["slot_dim"],
                              latent_dim = params["slot_dim"],
                              nlatents = len(params["latent_vars"]),
                              A = A,
                              stage="train"
                              ).to(device)

    GUIDE_PATH = f"{main_dir}/checkpoint-{params['jobID']}/guide_{params['epoch_to_load']}.pth"
    guide.load_state_dict(torch.load(GUIDE_PATH))
    guide.eval()
    
    #logger.info(f'\n\nseed {seed}\n\n')
    logger.info(GUIDE_PATH)

    plots_dir = os.path.abspath("set_prediction_plots")
    if not os.path.isdir(plots_dir): os.mkdir(plots_dir)
    else: 
        shutil.rmtree(plots_dir)
        os.mkdir(plots_dir)

    for OOD_EVAL in [False]:

        logger.info(f"\nOOD: {OOD_EVAL}\n")

        dataset_dir = "/nas-ctm01/datasets/public/CG/eval"

        dataset = MyDataset(dataset_dir, OOD_EVAL)
        dataloader = DataLoader(dataset, 64, shuffle=False, generator=torch.Generator(device='cuda'))

        z_noise = torch.tensor([])  
        targets = torch.tensor([])      

        with torch.no_grad():
            for img, target in dataloader:
                img = img.to(device)
                target = target.to(device)
                
                batch_noise = guide(img, target) # [B, N, L, d]
                z_noise = torch.cat((z_noise, batch_noise), dim=0)

                B, N, _, d = batch_noise.shape
                targets = torch.cat((targets, target), dim=0)
        
        logger.info(z_noise.shape)
        logger.info(targets.shape)
        
        train_val_test_split = [0.7, 0.2, 0.1]
        for i in range(len(params["latent_vars"])):

            #if i == 0: j = 1
            #elif i == 1: j = 0
            j = i
            
            logger.info(f"\nprobe for variable {params['latent_vars'][i]}\n")
            
            model = LinearProbe(z_noise.size(-1), 1).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            n_train_samples = int(train_val_test_split[0]*z_noise.size(0))
            n_val_samples = n_train_samples + int(train_val_test_split[1]*z_noise.size(0))
            
            probe_ds_train = ProbeDataset(z_noise[:n_train_samples, :, i], targets[:n_train_samples], j, split="train")
            probe_ds_val = ProbeDataset(z_noise[n_train_samples:n_val_samples, :, i], targets[n_train_samples:n_val_samples], j, split="validation")
            probe_ds_test = ProbeDataset(z_noise[n_val_samples:, :, i], targets[n_val_samples:], j, split="test")

            probe_dl_train = DataLoader(probe_ds_train, 256, shuffle=True, generator=torch.Generator(device='cuda'))
            probe_dl_val = DataLoader(probe_ds_val, len(probe_ds_val), shuffle=False, generator=torch.Generator(device='cuda'))
            probe_dl_test = DataLoader(probe_ds_test, len(probe_ds_test), shuffle=False, generator=torch.Generator(device='cuda'))

            training_epochs = 21
            best_loss = 1.
            for epoch in range(training_epochs):
                epoch_train_loss = 0.
                epoch_val_loss = 0.
                
                model.train()
                for z, t in probe_dl_train:
                    z = z.to(device)
                    t = t.to(device)

                    optimizer.zero_grad()
                    
                    preds = model(z) # [B, N]
                    
                    # loss must be computed under a matching algorithm
                    loss = hungarian_matching_loss_scalar(preds, t)

                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss/z.size(0)
                
                model.eval()
                with torch.no_grad():
                    for z, t in probe_dl_val:
                        z = z.to(device)
                        t = t.to(device)                        
                        
                        preds = model(z)
                        loss = hungarian_matching_loss_scalar(preds, t)
                        epoch_val_loss += loss/z.size(0)
                
                if epoch_val_loss < best_loss:
                    #logger.info(f"saving new model with val loss: {epoch_val_loss}")
                    best_loss = epoch_val_loss
                    torch.save(model.state_dict(), 'pred_model.pth')
                
                if epoch % 10 == 0:
                    logger.info(f"epoch {epoch}/{training_epochs-1}")
                    logger.info(f"... train loss: {epoch_train_loss}")
                    logger.info(f"... val loss: {epoch_val_loss}")

            # testing
            model = LinearProbe(z_noise.size(-1), 1).to(device) 
            model.load_state_dict(torch.load('pred_model.pth')) 
            model.eval()
            with torch.no_grad():
                for z, t in probe_dl_test:
                    B, N, d = z.shape

                    z = z.to(device)
                    t = t.to(device)
                    preds = model(z)

                    logger.info(f"preds: {preds[:5, :]}")
                    logger.info(f"target: {t[:5, :]}")
                    r2_score = hungarian_r2_score(preds, t)
                    logger.info(f"r2: {r2_score[:5]}")


            logger.info(f"test r2 score: {torch.mean(r2_score)}")

if __name__ == '__main__':
    main()