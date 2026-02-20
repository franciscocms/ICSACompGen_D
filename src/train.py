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
from utils.loss import compute_log_det, compute_kl_gaussians, log_normal
from utils.train import BetaScheduler
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
ldim = params["latent_dim"]

logger.info(params)

l_to_label = {i: l for i, l in enumerate(params["latent_vars"])}
# A = torch.tensor([
#   [0., 0.],
#   [1., 0.],
# ])
A = torch.tensor([
    [0., 0., 0., 0.],
    [1., 0., 0., 0.],
    [0., 0., 0., 0.],
    [0., 0., 0., 0.],
  ])

logger.info(f"\nA: {A}\n")
torch.manual_seed(params["random_seed"])

guide = InvSlotAttentionGuide(model_name=params["model_name"], #["slot_attention", "additiveAE"]
                              resolution = params['resolution'],
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
  resume_epoch = 0

else:
  resume_epoch = params['epoch_to_load']
  logger.info(f"resume training from epoch {resume_epoch}...")
  checkpoint = torch.load(f"{main_dir}/checkpoint-{params['jobID']}/guide_{resume_epoch}.pth", weights_only=False)
  #checkpoint = torch.load(f"{main_dir}/checkpoint-157/guide_{resume_epoch}.pth", weights_only=False)
  guide.load_state_dict(checkpoint["model"])
  
optimizer = torch.optim.AdamW(guide.parameters(), lr=1e-7, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=2)
beta_scheduler = BetaScheduler(0., 1., 100)
if resume_epoch != 0: 
  optimizer.load_state_dict(checkpoint["optimizer"])
  scheduler = checkpoint["scheduler"]
  beta_scheduler.current_step = resume_epoch  

if params["TC_minimization"]:
  dis_optimizer = torch.optim.Adam(guide.tc_discriminator.parameters(), lr=params["lr"] * 0.1)

# instatiate dataset and dataloader
dataset_dir = os.path.join(params["dataset_path"], "train")
img_dir = os.path.join(dataset_dir, "images")
target_dir = os.path.join(dataset_dir, "metadata")

train_ds = MyDataset(dataset_dir)
logger.info(len(train_ds))

train_dl = DataLoader(train_ds, params["batch_size"], shuffle=True, num_workers=8)

recon_loss_fn = nn.MSELoss()
slot_rec_loss_crit = nn.MSELoss()

guide.train()
for epoch in range(resume_epoch + 1, resume_epoch + params["training_epochs"] + 1):

  logger.info(epoch)

  guide.epoch = epoch

  if scheduler.get_last_lr()[0] >= 1e-7:
    scheduler.step()
  
  if scheduler.get_last_lr()[0] > params["lr"]:
    optimizer.param_groups[0]["lr"] = params["lr"]
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.5
    )

  for group in optimizer.param_groups:    
    epoch_lr = group['lr']
  
  guide.epoch = epoch
  epoch_train_loss = 0.
  epoch_val_loss = 0.

  # update coefficients
  TC_coeff = params["TC_coeff"]
  alpha = 1.
  log_det_coeff = 0.01 # 0.1
  kl_coeff = 0.1

  kl_beta = beta_scheduler.get_beta()

  # update gradient computation flags
  if params["TC_minimization"]:
    for name, param in guide.tc_discriminator.named_parameters():
      param.requires_grad = False
  
  printed = False
  num_iters = 0
  epoch_recon_loss = 0.
  epoch_pred_loss = 0.
  epoch_noise_pred_loss = 0.
  epoch_flatt_noise_pred_loss = 0.
  epoch_log_e = 0.
  epoch_tc_loss = 0.
  epoch_log_z = 0.
  epoch_log_det = 0.
  epoch_logvar_noise = 0.
  epoch_consistency_loss = 0. 
  
  for img, target in train_dl:
    img = img.to(DEVICE)
    target = target.to(DEVICE)
    B = img.size(0)
    guide.iter = num_iters

    batch_loss = 0.
    optimizer.zero_grad()
  
    slots, img_hat, z_noise, mu_noise, logvar_noise, z_scm, recon_figures = guide(img, target, step=epoch)
    recon_loss = recon_loss_fn(img_hat, img)
    batch_loss += 100*recon_loss + kl_beta * (guide.pred_loss + torch.mean(guide.noise_pred_loss) + torch.mean(guide.flatt_noise_pred_loss))
    
    if epoch % epoch_step == 0 and not printed:
      if not os.path.isdir(f"{params['check_attn_folder']}/epoch-{epoch}"):
        os.mkdir(f"{params['check_attn_folder']}/epoch-{epoch}")     
    
    log_e = 0.
    for i in range(L):
      log_e += torch.mean(log_normal(z_noise[:, :, i], guide.mu_noise_prior[:, :, i], torch.exp(guide.logvar_noise_prior[:, :, i])).sum(-1))
    
    
    if params["TC_minimization"]:
      disc_input = torch.flatten(z_noise, -2)
      disc_input = torch.flatten(disc_input, 0, 1)
      D_joint = guide.tc_discriminator(disc_input)
      tc_loss = torch.square(torch.mean(D_joint[:, :1] - D_joint[:, 1:]))
      fake_z_noise, perms = permute_dims(disc_input, guide.slot_dim)
      batch_loss += kl_beta * TC_coeff * tc_loss
      
      assert not torch.isnan(tc_loss).any()
    else:
      tc_loss = torch.tensor(0.)

    log_z = 0.
    cp_v = torch.ones_like(guide.prior_z).to(DEVICE)
    for i in range(L):
      log_z += log_normal(z_scm[:, :, i],
                          guide.prior_z[:, :, i],
                          cp_v[:, :, i]).sum(-1)
    log_z = torch.mean(log_z)

    log_det = guide.log_det
    batch_loss += kl_beta * (-0.1*log_e - log_det_coeff*log_det + 0.01*torch.square(torch.mean(logvar_noise)) - 0.1*log_z)

    # causal consistency
    if epoch >= 100 and guide.causal_flow.z_int != None:
      z_int = guide.causal_flow.z_int.detach()
      with torch.no_grad():
        slots_int = guide.causal_decoder(z_int)
        x_int, _ = guide.scene_decoder(slots_int)
      mu_noise_int, logvar_noise_int, z_noise_int = guide.noise_encoder(x_int)

      mu_noise_int = mu_noise_int.reshape(mu_noise_int.size(0), mu_noise_int.size(1), L, ldim)
      logvar_noise_int = logvar_noise_int.reshape(logvar_noise_int.size(0), logvar_noise_int.size(1), L, ldim)
      z_noise_int = z_noise_int.reshape(z_noise_int.size(0), z_noise_int.size(1), L, ldim)

      consistency_loss = 0.
      if guide.int_flag == 0: invariant_vars = [1, 2, 3]
      elif guide.int_flag == 1: invariant_vars = [0, 2, 3]
      elif guide.int_flag == 2: invariant_vars = [0, 1, 3]
      elif guide.int_flag == 3: invariant_vars = [0, 1, 2]
      for i in invariant_vars:
        mu, logvar= mu_noise[:, :, i], logvar_noise[:, :, i]
        consistency_loss += recon_loss_fn(mu_noise_int[:, :, i], mu.detach()) + recon_loss_fn(logvar_noise_int[:, :, i], logvar.detach())
      
      if guide.int_flag == 0:
        consistency_loss += recon_loss_fn(guide.causal_flow.backward(z_int).detach(), z_noise_int[:, :, 0])
      elif guide.int_flag == 1:
        consistency_loss += recon_loss_fn(guide.causal_flow.backward(z_int).detach(), z_noise_int[:, :, 1])
      
      batch_loss += kl_beta * 0.001 * consistency_loss # 178 has 0.01
      #batch_loss += kl_beta * consistency_loss
    else:
      consistency_loss = 0.
      x_int = torch.zeros_like(img_hat)

    batch_loss.backward()
    optimizer.step()

    epoch_train_loss += batch_loss 
    epoch_recon_loss += recon_loss
    epoch_pred_loss += guide.pred_loss
    epoch_noise_pred_loss += torch.mean(guide.noise_pred_loss)
    epoch_flatt_noise_pred_loss += torch.mean(guide.flatt_noise_pred_loss)
    epoch_log_e += log_e
    epoch_tc_loss += tc_loss
    epoch_log_z += log_z
    epoch_log_det += log_det
    epoch_logvar_noise += torch.square(torch.mean(logvar_noise))
    epoch_consistency_loss += consistency_loss
    
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

      # if epoch % epoch_step == 0 and not printed:
      #   logger.info(f"real batches pred: {torch.mean(D_joint)}")
      #   logger.info(f"fake batches pred: {torch.mean(D_perm)}")
    else:
      disc_tc_loss = torch.tensor(0.)
    
    if guide.model_name == "additiveAE":
      if num_iters == 0:
        recon_figures = recon_figures.permute(1, 0, 2, 3, 4) # shape is (N, B, C, W, H)
        
        if (epoch % epoch_step == 0 or epoch == 0) and not printed:
          #if guide.int_flag == 1: logger.info(f"check color counterfactual at epoch {epoch}...")
          
          fig, ax = plt.subplots(5, 3 + params["num_slots"])
          for i in range(5):
              ax[i, 0].imshow(np.transpose(img[i].detach().cpu().numpy(), (1, 2, 0)))
              if i == 0: ax[i, 0].set_title("img")
              ax[i, 1].imshow(np.transpose(img_hat[i].detach().cpu().numpy(), (1, 2, 0)))
              if i == 0: ax[i, 1].set_title("img_hat")
              ax[i, 2].imshow(np.transpose(x_int[i].detach().cpu().numpy(), (1, 2, 0)))
              if i == 0: ax[i, 2].set_title(f"int {guide.int_flag}")

              for j in range(params["num_slots"]):
                  ax[i, j+3].imshow(np.transpose(recon_figures[j, i].detach().cpu().numpy(), (1, 2, 0)))
                  if i == 0: ax[i, j+3].set_title(f"figs {j}")

              for j in range(3 + params["num_slots"]):
                  ax[i, j].grid(False)
                  ax[i, j].axis('off')

          plt.savefig(f"{params['check_attn_folder']}/epoch-{epoch}/img_recon.png")
          plt.close()   

    num_iters += 1
    printed = True
    
  epoch_train_loss /= num_iters
  epoch_recon_loss /= num_iters
  epoch_pred_loss /= num_iters
  epoch_noise_pred_loss /= num_iters
  epoch_flatt_noise_pred_loss /= num_iters
  epoch_log_e /= num_iters
  epoch_tc_loss /= num_iters
  epoch_log_z /= num_iters
  epoch_log_det /= num_iters
  epoch_logvar_noise /= num_iters
  epoch_consistency_loss /= num_iters


  if epoch % epoch_step == 0:
    logger.info(f"epoch {epoch} - train_loss: {epoch_train_loss}")  

  dict_to_log = {'train_loss': epoch_train_loss}
  dict_to_log["rec_loss"] = epoch_recon_loss
  #dict_to_log["causal_rec_loss"] = causal_recon_loss
  dict_to_log["pred_loss"] = epoch_pred_loss
  #dict_to_log["slot_rec_loss"] = slot_recon_loss
  dict_to_log["log_det"] = epoch_log_det
  dict_to_log["log_det_coeff"] = log_det_coeff
  #dict_to_log["kl_noise"] = kl_noise
  dict_to_log["log_e"] = epoch_log_e
  dict_to_log["log_z"] = epoch_log_z
  dict_to_log["consistency_loss"] = epoch_consistency_loss
  dict_to_log["alpha"] = alpha
  dict_to_log["beta"] = kl_beta
  dict_to_log["kl_coeff"] = kl_coeff

  #dict_to_log["logvar_loss"] = prior_logvar_loss
  
  # dict_to_log["prior_means_distance_e"] = torch.mean(prior_means_distance_e)
  dict_to_log["LR"] = epoch_lr

  logvar_noise = logvar_noise.reshape(B, params["num_slots"], L, ldim)
  for i in range(L):
    # dict_to_log[f"mu_{params['latent_vars'][i]}"] = torch.mean(torch.mean(mu_noise[:, :, i, :], dim=-1))
    # dict_to_log[f"prior_mu_{params['latent_vars'][i]}"] = torch.mean(torch.mean(prior_means[:, :, i, :], dim=-1))
    
    dict_to_log[f"logvar_{params['latent_vars'][i]}"] = torch.mean(torch.mean(logvar_noise[:, :, i, :], dim=-1))
    dict_to_log[f"prior_logvar_{params['latent_vars'][i]}"] = torch.mean(torch.mean(guide.logvar_noise_prior[:, :, i, :], dim=-1))

    dict_to_log[f"s_{params['latent_vars'][i]}"] = torch.mean(torch.sum(guide.ss[:, :, i, :], dim=-1))
    dict_to_log[f"t_{params['latent_vars'][i]}"] = torch.mean(torch.sum(guide.ts[:, :, i, :], dim=-1))

    #dict_to_log[f"dist_{params['latent_vars'][i]}"] = dist[i]

    dict_to_log[f"noise_pred_loss_{params['latent_vars'][i]}"] = guide.noise_pred_loss[i]
    dict_to_log[f"flatt_noise_pred_loss_{params['latent_vars'][i]}"] = guide.flatt_noise_pred_loss[i]

  if params["TC_minimization"]:
    dict_to_log["latent_TC"] = tc_loss
    dict_to_log["latent_TC_disc"] = disc_tc_loss
  
  if epoch % epoch_step == 0 or epoch == params["training_epochs"]:
    checkpoint = {
      'epoch': epoch,
      'model': guide.state_dict(),
      'optimizer': optimizer.state_dict(),
      'scheduler': scheduler
    }
    torch.save(checkpoint, GUIDE_PATH + '/guide_' + str(epoch) + '.pth')

  run.log(dict_to_log)

logger.info("\ntraining ended...")
wandb.finish()
