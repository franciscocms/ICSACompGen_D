import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import os
import shutil
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '..') # add parent dir

from models.guide import InvSlotAttentionGuide
from main import params
from utils.guide import permute_dims, linear_scheduler, exp_scheduler
from utils.loss import compute_log_det, compute_kl_gaussians
from datasets.data_utils import MyDataset

import wandb # type: ignore

import logging
logfile_name = f"log-{params['jobID']}.log"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(logfile_name, mode='w')
logger.addHandler(fh)

main_dir = os.path.abspath(__file__+'/../')

torch.autograd.detect_anomaly(check_nan=True)

assert params["running_type"] == "train"

# start a new wandb run to track this script
run = wandb.init(project="ICSA",
                  name=f"{params['jobID']}"
                  )

DEVICE = params["device"]
TRAINING_FROM_SCRATCH = params["training_from_scratch"]
GUIDE_PATH = f"{main_dir}/checkpoint-{params['jobID']}"
for p in [GUIDE_PATH]: 
  if not os.path.isdir(p): os.mkdir(p)
logger.info(f"\n... saving model checkpoints in {GUIDE_PATH}")
CHECK_ATTN = params["check_attn"]
root_folder = params["check_attn_folder"]
TRAIN_BATCH_SIZE = params["batch_size"]
LR = params["lr"]
training_epochs = params["training_epochs"]
epoch_step = params["epoch_step"]
L = len(params["latent_vars"])

logger.info(params)

l_to_label = {i: l for i, l in enumerate(params["latent_vars"])}
# A = torch.tensor([
#     [0., 0., 0., 0.],
#     [1., 0., 0., 0.],
#     [0., 0., 0., 0.],
#     [0., 0., 0., 0.],
#   ])

A = torch.tensor([
  [0., 0.],
  [1., 0.],
])

logger.info(f"\nA: {A}\n")
torch.manual_seed(params["random_seed"])

guide = InvSlotAttentionGuide(resolution = params['resolution'],
                              num_iterations = 3,
                              latent_dim = params["latent_dim"],
                              nlatents = len(params["latent_vars"]),
                              A = A
                              ).to(DEVICE)
run.watch(guide)

logger.info(f"Training from scratch using {DEVICE}...\n")

if not os.path.isdir(root_folder): os.mkdir(root_folder)

if CHECK_ATTN and TRAINING_FROM_SCRATCH:
  if len(os.listdir(root_folder)) > 0:
    try:
      folders = os.listdir(root_folder)
      for folder in folders: 
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path) and folder.split("-")[0] == "attn":
          shutil.rmtree(folder_path)
    except OSError:
      logger.info("Error occurred while deleting files.")

optimizer = torch.optim.Adam(guide.parameters(), lr=params["lr"])
if params["TC_minimization"]:
  dis_optimizer = torch.optim.Adam(guide.tc_discriminator.parameters(), lr=params["lr"] * 0.1)


# instatiate dataset and dataloader
dataset_dir = "/nas-ctm01/datasets/public/CG/train"
img_dir = os.path.join(dataset_dir, "images")
target_dir = os.path.join(dataset_dir, "metadata")

train_ds = MyDataset(dataset_dir)
logger.info(len(train_ds))

train_dl = DataLoader(train_ds, params["batch_size"], shuffle=True, num_workers=8)

recon_loss_fn = nn.MSELoss()
slot_rec_loss_crit = nn.MSELoss()

guide.train()
for epoch in range(params["training_epochs"]):

  guide.epoch = epoch

  # update lr every 50 epochs
  
  for group in optimizer.param_groups:
    if epoch % 50 == 0 and epoch != 0:
      group['lr'] /= 2
    epoch_lr = group['lr']
  

  # if epoch < 50:
  #   for name, params in guide.named_parameters():
  #     params.requires_grad = False
  #   for name, params in guide.model.named_parameters():
  #     params.requires_grad = True
  # elif epoch >= 50:
  #   for name, params in guide.named_parameters():
  #     params.requires_grad = True
  #   for name, params in guide.model.named_parameters():
  #     params.requires_grad = False
  
  guide.epoch = epoch
  epoch_train_loss = 0.
  epoch_val_loss = 0.

  # update coefficients
  TC_coeff = params["TC_coeff"]
  #alpha = linear_scheduler(epoch, 20, 0., params["alpha"])
  alpha = params["alpha"]
  #alpha = (1 - torch.sigmoid(torch.tensor(20 - epoch)*100))*params["alpha"]
  log_det_coeff = 0. # 0.001
  #guide.learn_transform = 0. if epoch < 20 else 1.
  guide.learn_transform = 1.

  # update gradient computation flags
  if params["TC_minimization"]:
    for name, param in guide.tc_discriminator.named_parameters():
      param.requires_grad = False
  
  printed = False
  num_iters = 0
  for img, target in train_dl:
    img = img.to(DEVICE)
    target = target.to(DEVICE)
    B = img.size(0)

    guide.iter = num_iters

    batch_loss = 0.
    optimizer.zero_grad()
  
    slots, img_hat, z_noise, mu_noise, logvar_noise, prior_means, prior_logvars, z_scm, slot_hat, figs = guide(img, target)
    
    # logger.info(img.shape)
    # logger.info(img_hat.shape)

    recon_loss = recon_loss_fn(img_hat, img)
    slot_recon_loss = slot_rec_loss_crit(slot_hat, slots)

    target_logvar = torch.tensor(-4.0).to(DEVICE)
    prior_logvar_loss = F.mse_loss(prior_logvars, target_logvar.expand_as(prior_logvars))
    batch_loss += 100*recon_loss + slot_recon_loss + guide.pred_loss + torch.mean(guide.noise_pred_loss) + prior_logvar_loss
    
    if epoch % epoch_step == 0 and not printed:
      if not os.path.isdir(f"{params['check_attn_folder']}/epoch-{epoch}"):
        os.mkdir(f"{params['check_attn_folder']}/epoch-{epoch}")     
    
    if params["TC_minimization"]:
      disc_input = torch.flatten(z_noise, -2)
      D_joint = guide.tc_discriminator(disc_input)
      tc_loss = torch.square(torch.mean(D_joint[:, :1] - D_joint[:, 1:]))
      fake_z_noise, perms = permute_dims(disc_input, guide.slot_dim)
      batch_loss += TC_coeff*tc_loss
      assert not torch.isnan(tc_loss).any()
    else:
      tc_loss = torch.tensor(0.)
      
    kl_noise = 0.
    prior_means_d = prior_means.detach()
    prior_logvars_d = prior_logvars.detach()
    for i in range(L):
      kl_noise += compute_kl_gaussians(mu_noise[:, :, i, :],
                                       logvar_noise[:, :, i, :],
                                       prior_means_d[:, :, i, :],
                                       prior_logvars_d[:, :, i, :])
      # logger.info(f"{i} - {kl_noise}")
    
    # if params["causal_dropout_prob"] < 1.:
    #   log_det = compute_log_det(guide.L_slopes)
    #   assert not torch.isnan(log_det).any()
    # else:
    #   log_det = torch.tensor(0.)
    # batch_loss += alpha*(kl_noise - log_det_coeff*log_det)
    log_det = compute_log_det(guide.L_slopes)
    batch_loss += alpha * 0.1 * kl_noise - log_det_coeff*log_det
    
    batch_loss.backward()
    optimizer.step()
    epoch_train_loss += batch_loss 
    num_iters += 1
    
    if params["TC_minimization"]:
      for name, param in guide.tc_discriminator.named_parameters():
        param.requires_grad = True
      
      fake_z_noise = fake_z_noise.detach()
      disc_input = disc_input.detach()
      D_perm = guide.tc_discriminator(fake_z_noise)
      D_joint = guide.tc_discriminator(disc_input)
      ones = torch.ones(D_perm.size(0), dtype=torch.long, device=DEVICE)
      zeros = torch.zeros(D_perm.size(0), dtype=torch.long, device=DEVICE)
      disc_tc_loss = torch.mean(0.5*(F.cross_entropy(D_joint, zeros) + F.cross_entropy(D_perm, ones)))
      assert not torch.isnan(disc_tc_loss).any()
      
      dis_optimizer.zero_grad()
      disc_tc_loss.backward()
      dis_optimizer.step()

      if epoch % epoch_step == 0 and not printed:
        logger.info(f"real batches pred: {torch.mean(D_joint)}")
        logger.info(f"fake batches pred: {torch.mean(D_perm)}")
    else:
      disc_tc_loss = torch.tensor(0.)
    
    if (epoch % epoch_step == 0 or epoch == 0) and not printed:
      fig, ax = plt.subplots(1, 4)
      ax[0].imshow(np.transpose(img[0].detach().cpu().numpy(), (1, 2, 0)))
      ax[1].imshow(np.transpose(img_hat[0].detach().cpu().numpy(), (1, 2, 0)))
      ax[2].imshow(np.transpose(figs[0, 0].detach().cpu().numpy(), (1, 2, 0)))
      ax[3].imshow(np.transpose(figs[1, 0].detach().cpu().numpy(), (1, 2, 0)))

      for j in range(4):
        ax[j].grid(False)
        ax[j].axis('off')   
      plt.savefig(f"{params['check_attn_folder']}/epoch-{epoch}/img_recon.png")
      plt.close()

    with torch.no_grad():
      
      prior_means_distance_e = torch.sum(torch.square(prior_means_d[:, :, 0] - prior_means_d[:, :, 1]), dim=-1) # [B, N]
      
      u_min = torch.zeros(1, 1, L).to(DEVICE) # Label = 0
      u_max = torch.ones(1, 1, L).to(DEVICE) # Label = 1
      
      mu_min, _ = guide.prior(u_min) 
      mu_max, _ = guide.prior(u_max)
        
      # Calculate the Euclidean distance in the 16D space
      dist = []
      for i in range(L):
        dist.append(torch.norm(mu_max[:, :, i] - mu_min[:, :, i]))   

    printed = True
    
    
  epoch_train_loss /= num_iters
  if epoch % epoch_step == 0:
    logger.info(f"epoch {epoch} - train_loss: {epoch_train_loss}")  

  dict_to_log = {'train_loss': epoch_train_loss}
  dict_to_log["rec_loss"] = recon_loss
  dict_to_log["pred_loss"] = guide.pred_loss
  #dict_to_log["noise_pred_loss"] = guide.noise_pred_loss
  dict_to_log["slot_rec_loss"] = slot_recon_loss
  dict_to_log["log_det"] = log_det
  dict_to_log["log_det_coeff"] = log_det_coeff
  dict_to_log["kl_noise"] = kl_noise
  dict_to_log["alpha"] = alpha

  dict_to_log["logvar_loss"] = prior_logvar_loss
  
  dict_to_log["prior_means_distance_e"] = torch.mean(prior_means_distance_e)
  dict_to_log["LR"] = epoch_lr

  for i in range(L):
    dict_to_log[f"mu_{params['latent_vars'][i]}"] = torch.mean(torch.mean(mu_noise[:, :, i, :], dim=-1))
    dict_to_log[f"prior_mu_{params['latent_vars'][i]}"] = torch.mean(torch.mean(prior_means[:, :, i, :], dim=-1))
    
    dict_to_log[f"logvar_{params['latent_vars'][i]}"] = torch.mean(torch.mean(logvar_noise[:, :, i, :], dim=-1))
    dict_to_log[f"prior_logvar_{params['latent_vars'][i]}"] = torch.mean(torch.mean(prior_logvars[:, :, i, :], dim=-1))

    dict_to_log[f"s_{params['latent_vars'][i]}"] = torch.mean(torch.sum(guide.ss[:, :, i, :], dim=-1))
    dict_to_log[f"t_{params['latent_vars'][i]}"] = torch.mean(torch.sum(guide.ts[:, :, i, :], dim=-1))

    dict_to_log[f"dist_{params['latent_vars'][i]}"] = dist[i]

    dict_to_log[f"noise_pred_loss_{params['latent_vars'][i]}"] = guide.noise_pred_loss[i]

  dict_to_log["causal_dropout_prob"] = params["causal_dropout_prob"]
  if params["TC_minimization"]:
    dict_to_log["latent_TC"] = tc_loss
    dict_to_log["latent_TC_disc"] = disc_tc_loss
  
  
  torch.save(guide.state_dict(), GUIDE_PATH+'/guide_'+str(epoch)+'.pth')
  run.log(dict_to_log)

logger.info("\ntraining ended...")
wandb.finish()
