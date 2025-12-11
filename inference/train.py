import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# add project path to sys to import relative modules
import json
from PIL import Image
import sys
import glob
sys.path.append(os.path.abspath(__file__+'/../../'))

from guide import InvSlotAttentionGuide
from main.setup import params
from utils.guide import permute_dims, linear_scheduler, exp_scheduler

import wandb # type: ignore

import logging
logfile_name = f"log-{params['jobID']}.log"
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(logfile_name, mode='w')
logger.addHandler(fh)

main_dir = os.path.abspath(__file__+'/../../')

torch.autograd.set_detect_anomaly(True)

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

shape_vals = {v: params["shape_vals"].index(v) for v in params["shape_vals"]}
size_vals = {v: params["size_vals"].index(v) for v in params["size_vals"]}
color_vals = {v: params["color_vals"].index(v) for v in params["color_vals"]}

logger.info(params)

l_to_label = {i: l for i, l in enumerate(params["latent_vars"])}
A = params["adj_matrix"]
logger.info(f"\nA: {A}\n")
torch.manual_seed(params["random_seed"])

guide = InvSlotAttentionGuide(resolution = params['resolution'],
                              num_iterations = 3,
                              slot_dim = params["slot_dim"],
                              latent_dim = params["slot_dim"],
                              nlatents = len(params["latent_vars"]),
                              A = A,
                              stage="train"
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

def process_targets(target_dict):   
    n = int(target_dict['scene_attr']['N'])
    if "size" in params["latent_vars"]:
        target = torch.zeros(params['num_slots'], len(shape_vals) + len(size_vals) + len(color_vals) + 2) 
    else:
        target = torch.zeros(params['num_slots'], len(shape_vals) + len(color_vals) + 2) 

    for i in range(n):
        s = len(color_vals)
        target[i, :s] = F.one_hot(torch.tensor(color_vals[target_dict['scene_attr'][f'object_{i}']['color']]), len(color_vals))
        
        if "size" in params["latent_vars"]:
            target[i, s:s+len(size_vals)] = F.one_hot(torch.tensor(size_vals[target_dict['scene_attr'][f'object_{i}']['size']]), len(size_vals))
            s = s+len(size_vals)
        
        target[i, s:s+len(shape_vals)] = F.one_hot(torch.tensor(shape_vals[target_dict['scene_attr'][f'object_{i}']['shape']]), len(shape_vals))
        s = s+len(shape_vals)
        target[i, s] = target_dict['scene_attr'][f'object_{i}']['initLocX']
        target[i, s+1] = target_dict['scene_attr'][f'object_{i}']['initLocY']
    return target

class MyDataset(Dataset):
  def __init__(self, dataset_path):
      self.dataset_path = dataset_path
      self.img_path = os.path.join(self.dataset_path, "images")
      self.target_path = os.path.join(self.dataset_path, "metadata")
      self.nscenes = len(glob.glob(os.path.join(self.img_path, "*.png")))
      self.transform = transforms.Compose([transforms.ToTensor()])

  def __len__(self):
      return self.nscenes
  
  def __getitem__(self, idx):
    img = self.transform(Image.open(os.path.join(self.img_path, f"{str(idx).zfill(len(str(self.nscenes)))}.png")))
    target_dict = json.load(open(os.path.join(self.target_path, f"{str(idx).zfill(len(str(self.nscenes)))}.json")))
    target = process_targets(target_dict)
    return img, torch.cat((torch.argmax(target[:, :3], dim=-1).unsqueeze(-1),
                           torch.argmax(target[:, 3:6], dim=-1).unsqueeze(-1)), dim=-1)


# instatiate dataset and dataloader
dataset_dir = "/nas-ctm01/datasets/public/CG/train"
img_dir = os.path.join(dataset_dir, "images")
target_dir = os.path.join(dataset_dir, "metadata")

train_ds = MyDataset(dataset_dir)
logger.info(len(train_ds))

train_dl = DataLoader(train_ds, params["batch_size"], shuffle=True, generator=torch.Generator(device='cuda'))

def compute_e_loss(mu, logvar, e, print_mixt_params, yc):
  kl_noise = 0.
  for i in range(L):
    mixt_kl, kl_reg, logvar_reg, yc = guide.z_nets[l_to_label[i]].compute_elbo(mu[:, i, :].unsqueeze(-2),
                                                                                logvar[:, i, :].unsqueeze(-2),
                                                                                e[:, i, :],
                                                                                print_mixt_params=print_mixt_params,
                                                                                yc_temperature=yc)
    kl_noise += params["mixt_kl_coeff"] * mixt_kl + kl_reg - params["logvar_reg_coeff"]*logvar_reg

    if print_mixt_params:
      logger.info(f"\nGMMVAE losses for variable {l_to_label[i]}:")
      logger.info(f"yc: {yc[:5]}")
      logger.info(f"yc mean: {torch.mean(yc, dim=0)}")
      logger.info(f"logvars: {torch.mean(logvar[:, i, :])}")
  
  return kl_noise

def compute_log_det(L_slopes):
  log_det = 0.
  for i in range(L):
    s = L_slopes[:, :, i, :]
    if params["normalized_logdet"]: log_det += torch.sum(s, dim=-1)/s.size(-1)
    else: log_det += torch.sum(s, dim=-1)
  log_det = torch.mean(log_det)
  return log_det

def compute_prior_z_loss(z, cp):
  B, n_s, _, _ = z.shape
  causal_loss = 0.
  for i in range(L):
    causal_loss += guide.mmd_loss(z[:, :, i, :].reshape(B*n_s, -1), cp[:, :, i, :].reshape(B*n_s, -1))
  return

recon_loss_fn = nn.L1Loss()
slot_rec_loss_crit = nn.L1Loss()
prior_labels_breakin = 1

guide.train()
for epoch in range(params["training_epochs"]):
  
  guide.epoch = epoch
  epoch_train_loss = 0.
  epoch_val_loss = 0.

  # update coefficients
  yc_t = linear_scheduler(epoch, 20, params["yc_temperature"], 1.)
  prior_coeff = exp_scheduler(epoch, 20, 0., 1.)
  TC_coeff = params["TC_coeff"] if epoch >= prior_labels_breakin else 0.
  log_det_coeff = params["log_det_coeff"] if epoch >= prior_labels_breakin else 0.
  alpha = params["alpha"] if epoch >= prior_labels_breakin else 0.

  # update gradient computation flags
  if params["TC_minimization"]:
    for name, param in guide.tc_discriminator.named_parameters():
      param.requires_grad = False
  
  for img, target in train_dl:
    img = img.to(DEVICE)
    target = target.to(DEVICE)

    logger.info(img.shape)

    printed = False
    batch_loss = 0.
    optimizer.zero_grad()
    
    slots, recon, e, mu, logvar, z, slot_hat, cp = guide(img, target)
    recon_loss = recon_loss_fn(recon, img)
    slot_recon_loss = slot_rec_loss_crit(slot_hat, slots)
    batch_loss += alpha*(recon_loss + slot_recon_loss)
    
    if epoch % epoch_step == 0 and not printed:
      print_mixt_params = True
      logger.info(f"img recon loss: {recon_loss}")
      logger.info(f"slot recon loss: {slot_recon_loss}")

    kl_noise = compute_e_loss(mu, logvar, e, print_mixt_params, yc_t)
    batch_loss += alpha*kl_noise
    
    if params["TC_minimization"]:
      disc_input = torch.flatten(e, -2)
      D_joint = guide.tc_discriminator(disc_input)
      tc_loss = torch.square(torch.mean(D_joint[:, :1] - D_joint[:, 1:]))
      fake_z_noise, perms = permute_dims(disc_input, params["slot_dim"])
      batch_loss += TC_coeff*tc_loss
    else:
      tc_loss = torch.tensor(0.)
      
    if params["causal_dropout_prob"] <= 1.:
      log_det = compute_log_det(guide.L_slopes)
      batch_loss -= log_det_coeff*log_det
    else:
      log_det = torch.tensor(0.)
    
    if params["beta"] > 0:
      log_det_prior = compute_log_det(guide.prior_L_slopes)
      causal_loss = compute_prior_z_loss(z, cp)
      batch_loss += prior_coeff*causal_loss - prior_coeff*log_det_coeff*log_det_prior
    else:
      causal_loss = torch.tensor(0.)
      log_det_prior = torch.tensor(0.)

    batch_loss.backward()
    optimizer.step()
    epoch_train_loss += batch_loss/img.size(0) 

    if params["TC_minimization"]:
      for name, param in guide.tc_discriminator.named_parameters():
        param.requires_grad = True
      
      D_perm = guide.tc_discriminator(fake_z_noise.detach())
      D_joint = guide.tc_discriminator(disc_input.detach())
      ones = torch.ones(D_perm.size(0), dtype=torch.long, device=DEVICE)
      zeros = torch.zeros(D_perm.size(0), dtype=torch.long, device=DEVICE)
      disc_tc_loss = alpha * torch.mean(0.5*(F.cross_entropy(D_joint, zeros) + F.cross_entropy(D_perm, ones)))
      
      dis_optimizer.zero_grad()
      disc_tc_loss.backward()
      dis_optimizer.step()

      if epoch % epoch_step == 0 and not printed:
        logger.info(f"real batches pred: {torch.mean(D_joint)}")
        logger.info(f"fake batches pred: {torch.mean(D_perm)}")
    else:
      disc_tc_loss = torch.tensor(0.)
    
    printed = True
    
  if epoch % epoch_step == 0:
    logger.info(f"epoch {epoch} - train_loss: {epoch_train_loss}")  

  dict_to_log = {'train_loss': epoch_train_loss}
  dict_to_log["causal_loss"] = causal_loss
  dict_to_log["log_det"] = log_det
  dict_to_log["kl_noise"] = kl_noise
  if params["beta"] > 0:
    dict_to_log["prior_coeff"] = prior_coeff
    dict_to_log["prior_s"] = guide.prior_ss
    dict_to_log["prior_t"] = guide.prior_ts
  dict_to_log["causal_dropout_prob"] = guide.causal_dropout_prob
  dict_to_log["yc_temperature"] = yc_t
  if params["TC_minimization"]:
    dict_to_log["latent_TC"] = tc_loss
    dict_to_log["latent_TC_disc"] = disc_tc_loss
  dict_to_log["log_det_prior"] = guide.log_det_prior
  
  torch.save(guide.state_dict(), GUIDE_PATH+'/guide_'+str(epoch)+'.pth')
  run.log(dict_to_log)

logger.info("\ntraining ended...")
wandb.finish()
