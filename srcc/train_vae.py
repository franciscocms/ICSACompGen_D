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

from models.architectures import FullNoiseEncoder, VAEAdditiveDecoder
from main import params
from utils.loss import compute_kl_loss
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


torch.manual_seed(params["random_seed"])

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.L = len(params["latent_vars"])
        self.ldim = params["latent_dim"]
        self.encoder = FullNoiseEncoder(in_channels=3,
                                        n_slots=params["num_slots"],
                                        n_latents=len(params["latent_vars"]),
                                        latent_dim=params["latent_dim"],
                                        stage=params["running_type"],
                                        clamp_var=False)

        self.decoder = VAEAdditiveDecoder(in_channels=3,
                                          n_slots=params["num_slots"],
                                          n_latents=len(params["latent_vars"]),
                                          n_slot_latents=params["latent_dim"],
                                          add_sigmoid=True
                                          )

    def forward(self, x):
        mean, logvar, z = self.encoder(x)
        
        mean = mean.reshape(mean.size(0), mean.size(1), self.L, self.ldim)
        logvar = logvar.reshape(logvar.size(0), logvar.size(1), self.L, self.ldim)
        z = z.reshape(z.size(0), z.size(1), self.L, self.ldim) 

        img_hat, recon_figures = self.decoder(z)
        #img_hat, recon_figures = self.decoder(mean) # job 611
        return img_hat, recon_figures, mean, logvar

    def compute_recon_loss(self, x_hat, x):
        return F.mse_loss(x_hat, x)

    def compute_kl_div(self, mu, logvar):
        assert len(mu.shape) == 4
        mu = torch.flatten(torch.flatten(mu, 0, 1), -2)
        logvar = torch.flatten(torch.flatten(logvar, 0, 1), -2)
        return compute_kl_loss(mu, logvar)


model = VAE().to(DEVICE)
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
  
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# instatiate dataset and dataloader
dataset_dir = os.path.join(params["dataset_path"], "train")
img_dir = os.path.join(dataset_dir, "images")
target_dir = os.path.join(dataset_dir, "metadata")
train_ds = MyDataset(dataset_dir)
logger.info(len(train_ds))
train_dl = DataLoader(train_ds, params["batch_size"], shuffle=True, num_workers=8)
recon_loss_fn = nn.MSELoss()
slot_rec_loss_crit = nn.MSELoss()

model.train()
for epoch in range(resume_epoch + 1, resume_epoch + params["training_epochs"] + 1):
    epoch_train_loss = 0.
    
    printed = False
    num_iters = 0
    epoch_recon_loss = 0.
    epoch_kl = 0.
    
    for img, target in train_dl:
        img = img.to(DEVICE)
        target = target.to(DEVICE)
        B = img.size(0)

        batch_loss = 0.
        optimizer.zero_grad()
    
        img_hat, recon_figures, mean, logvar = model(img)
        recon_loss = model.compute_recon_loss(img_hat, img)
        kl_loss = model.compute_kl_div(mean, logvar)
        batch_loss += recon_loss + 0.01 * kl_loss
        
        if epoch % epoch_step == 0 and not printed:
            if not os.path.isdir(f"{params['check_attn_folder']}/epoch-{epoch}"):
                os.mkdir(f"{params['check_attn_folder']}/epoch-{epoch}")     
        

        batch_loss.backward()
        optimizer.step()

        epoch_train_loss += batch_loss.item() 
        epoch_recon_loss += recon_loss.item()
        epoch_kl += kl_loss.item()
        
        if num_iters == 0:
            recon_figures = recon_figures.permute(1, 0, 2, 3, 4) # shape is (N, B, C, W, H)
            if (epoch % epoch_step == 0 or epoch == 0) and not printed:
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
    
    if epoch % 50 == 0 or epoch == params["training_epochs"]:
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
