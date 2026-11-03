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

from models.architectures import FullNoiseEncoder, VAEAdditiveDecoder, VAE
from main import params
from utils.loss import compute_kl_loss
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
epoch_step = params["epoch_step"]
L = len(params["latent_vars"])
ldim = params["latent_dim"]

logger.info(params)


torch.manual_seed(params["random_seed"])

model = VAE(num_slots=params["num_slots"], L=L, ldim=ldim, stage=params["running_type"]).to(DEVICE)
run.watch(model)
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
  
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# instatiate dataset and dataloader
dataset_dir = os.path.join(params["dataset_path"], "train")
img_dir = os.path.join(dataset_dir, "images")
target_dir = os.path.join(dataset_dir, "metadata")
train_ds = MyDataset(dataset_dir)
logger.info(len(train_ds))
train_dl = DataLoader(train_ds, params["batch_size"], shuffle=True, num_workers=8)
recon_loss_fn = nn.MSELoss()
slot_rec_loss_crit = nn.MSELoss()

beta_scheduler = BetaScheduler(0., 1., 100) # 613

model.train()
for epoch in range(resume_epoch + 1, resume_epoch + 120 + 1):
    epoch_train_loss = 0.
    
    printed = False
    num_iters = 0
    epoch_recon_loss = 0.
    epoch_kl = 0.

    kl_beta = beta_scheduler.get_beta()
    
    for img, target in train_dl:
        img = img.to(DEVICE)
        target = target.to(DEVICE)
        B = img.size(0)

        optimizer.zero_grad()
    
        img_hat, recon_figures, mean, logvar, z = model(img)
        recon_loss = model.compute_recon_loss(img_hat, img)
        kl_loss = model.compute_kl_div(mean, logvar)
        batch_loss = recon_loss + kl_beta * kl_loss
        
        if epoch % 5 == 0 and not printed:
            if not os.path.isdir(f"{params['check_attn_folder']}/epoch-{epoch}"):
                os.mkdir(f"{params['check_attn_folder']}/epoch-{epoch}")     
        
        batch_loss.backward()
        optimizer.step()

        epoch_train_loss += batch_loss.item() 
        epoch_recon_loss += recon_loss.item()
        epoch_kl += kl_loss.item()
        
        if num_iters == 0:
            recon_figures = recon_figures.permute(1, 0, 2, 3, 4) # shape is (N, B, C, W, H)
            if (epoch % 5 == 0 or epoch == 0) and not printed:
                fig, ax = plt.subplots(5, 2 + params["num_slots"])
                for i in range(5):
                    ax[i, 0].imshow(np.clip(np.transpose(img[i].detach().cpu().numpy(), (1, 2, 0)), 0., 1.))
                    if i == 0: ax[i, 0].set_title("img")
                    ax[i, 1].imshow(np.clip(np.transpose(img_hat[i].detach().cpu().numpy(), (1, 2, 0)), 0., 1.))
                    if i == 0: ax[i, 1].set_title("img_hat")
                    for j in range(params["num_slots"]):
                        ax[i, j+2].imshow(np.clip(np.transpose(recon_figures[j, i].detach().cpu().numpy(), (1, 2, 0)), 0., 1.))
                        if i == 0: ax[i, j+2].set_title(f"figs {j}")
                    for j in range(2 + params["num_slots"]):
                        ax[i, j].grid(False)
                        ax[i, j].axis('off')
                plt.savefig(f"{params['check_attn_folder']}/epoch-{epoch}/img_recon.png")
                plt.close()   

        num_iters += 1
        printed = True
        
    epoch_train_loss /= num_iters
    epoch_recon_loss /= num_iters
    epoch_kl /= num_iters

    if epoch % epoch_step == 0:
        logger.info(f"epoch {epoch} - train_loss: {epoch_train_loss}")  

    dict_to_log = {'vae_train_loss': epoch_train_loss}
    dict_to_log["vae_rec_loss"] = epoch_recon_loss
    dict_to_log["vae_kl"] = epoch_kl
    dict_to_log["vae_mean"] = torch.mean(mean)
    dict_to_log["vae_logvar"] = torch.mean(logvar)
    dict_to_log["vae_kl_beta"] = kl_beta
    
    if epoch % 5 == 0:
        checkpoint = {
        #'epoch': epoch,
        'model': model.state_dict(),
        #'optimizer': optimizer.state_dict(),
        #'scheduler': scheduler.state_dict()
        }
        torch.save(checkpoint, GUIDE_PATH + '/guide_' + str(epoch) + '.pth')
    run.log(dict_to_log)
logger.info("\ntraining ended...")
wandb.finish()
