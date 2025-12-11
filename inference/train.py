import os
import shutil
import torch
import torch.nn.functional as F
import pyro
import pyro.infer
import pyro.optim
import numpy as np
import pickle as pkl
from itertools import permutations
import random

# add project path to sys to import relative modules
import sys
sys.path.append(os.path.abspath(__file__+'/../../'))

from guide import InvSlotAttentionGuide
from main.models import model
from main.modifiedCSIS import CSIS
from utils.var import Variable
from main.setup import params
from utils.loss import save_loss, save_loss_plot
from utils.guide import get_pretrained_wts, load_trained_guide, compute_leakage_loss, linear_scheduler, exp_scheduler, exp_scheduler_neg

import wandb # type: ignore

import logging
logfile_name = f"log-{params['jobID']}.log"
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(logfile_name, mode='w')
logger.addHandler(fh)

main_dir = os.path.abspath(__file__+'/../../')

pyro.clear_param_store()

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

torch.manual_seed(params["random_seed"])

# fetch some variables from setup file
CHECK_ATTN = params["check_attn"]
root_folder = params["check_attn_folder"]
TRAIN_BATCH_SIZE = params["batch_size"]
VAL_BATCH_SIZE = params["batch_size"]
LR = params["lr"]
nsteps = params["training_iters"]
step_size = params['step_size']
L = len(params["latent_vars"])

logger.info(params)

l_to_label = {i: l for i, l in enumerate(params["latent_vars"])}
A = params["adj_matrix"]

# A = torch.zeros(9)
# dep_ind = 3
# A[dep_ind] = 1
# A = A.reshape((3, 3))

logger.info(f"\nA: {A}\n")



guide = InvSlotAttentionGuide(resolution = params['resolution'],
                              num_iterations = 3,
                              slot_dim = params["slot_dim"],
                              latent_dim = params["slot_dim"],
                              nlatents = len(params["latent_vars"]),
                              A = A,
                              stage="train"
                              ).to(DEVICE)
run.watch(guide)

# get model pre-trained weights
if not TRAINING_FROM_SCRATCH:
  pretrained_guide_path = GUIDE_PATH + "/guide_" + str(params["guide_step"]) + ".pth"
  logger.info(f"pretrained guide path: {pretrained_guide_path}")
  resume_training = False
  guide = load_trained_guide(guide, pretrained_guide_path)
  logger.info(f"Guide from step {params['guide_step']} successfully loaded...\n")
  resume_training = True
else: 
  logger.info(f"Training from scratch using {DEVICE}...\n")
  resume_training = False

optimiser = pyro.optim.Adam({'lr': LR})
csis = CSIS(model = model,
            guide = guide,
            optim = optimiser,
            num_inference_samples=params["num_inference_samples"],
            training_batch_size=TRAIN_BATCH_SIZE,
            validation_batch_size=VAL_BATCH_SIZE)

if resume_training: resume_step = int(pretrained_guide_path.split("/")[-1].split(".")[0].split("_")[1]) + 1
else: resume_step = 0

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

  # if len(os.listdir(root_folder)) == 0:
  #   logger.info("Folders for all steps deleted...")


# 'TRAINING_FROM_SCRATCH' is True either if the slot module is to be trained or not
# it is only False if it is supposed to load weights from some checkpoint saved while training the whole model
if TRAINING_FROM_SCRATCH:

  for name, param in csis.guide.named_parameters():
    if not param.requires_grad:
      logger.info(f"{name} - {param.requires_grad}")


if params["TC_minimization"]:
  dis_optimizer = torch.optim.Adam(csis.guide.tc_discriminator.parameters(), lr=params["lr"] * 0.1)

if params["reduce_noise_leak_adv"]:
  leak_optimizer = torch.optim.Adam(csis.guide.disentangle_mlps.parameters(), lr=1e-4)


all_val_loss = []
min_val_loss = 0.
for s in range(resume_step, resume_step + nsteps):    
  
  if CHECK_ATTN and (s % params["step_size_img_sv"] == 0 or s == csis.guide.gmm_fit_step): 
    if not os.path.isdir(f"{root_folder}/attn-step-{s}"): os.mkdir(f"{root_folder}/attn-step-{s}")   
  
  # if s >= csis.guide.gmm_fit_step:
  #   for name, param in csis.guide.label_embeddings.named_parameters():
  #     param.requires_grad = False
    
  breakin = torch.tensor(s - csis.guide.gmm_fit_step)
  
  if params["alpha_beta_scheduled"]:
    #csis.guide.alpha = torch.sigmoid(breakin) * linear_scheduler(s, 2000, 0.0, 1.0)
    csis.guide.alpha = torch.sigmoid(breakin) * exp_scheduler(s, 2000, 0.0, params["alpha"])
    #csis.guide.beta = torch.sigmoid(breakin) * linear_scheduler(s, 2000, 0.0, params["beta"])
    if params["beta"] != 0.:
      csis.guide.beta = torch.sigmoid(breakin) * exp_scheduler(s, 2000, 0.0, params["beta"])
    else:
      csis.guide.beta = 0.
  else:
    csis.guide.alpha = torch.sigmoid(breakin) * params["alpha"]
    csis.guide.beta = torch.sigmoid(breakin) * params["beta"]

  if params["causal_dropout_scheduled"]:
    #csis.guide.causal_dropout_prob = torch.sigmoid(breakin) * linear_scheduler(s, 2000, 1.0, params["causal_dropout_prob"])
    csis.guide.causal_dropout_prob = torch.sigmoid(breakin) * exp_scheduler_neg(s, 2000, 1.0, params["causal_dropout_prob"])
  
  csis.guide.pred_coeff = torch.sigmoid(breakin) * 1.
  csis.guide.TC_coeff = torch.sigmoid(breakin) * params["TC_coeff"]
  csis.guide.log_det_coeff = torch.sigmoid(breakin) * params["log_det_coeff"]

  csis.guide.gamma = 1.0 - torch.sigmoid(breakin)

  csis.guide.yc_t = linear_scheduler(s, 2200, params["yc_temperature"], 1.)
  #csis.guide.yc_t = params["yc_temperature"]
  if params["noise_logvar_constraint"]:
    csis.guide.logvar_coeff = linear_scheduler(s, 2200, 0., 1.)
  else:
    csis.guide.logvar_coeff = 0.

  if params["TC_minimization"]:
    for name, param in csis.guide.tc_discriminator.named_parameters():
      param.requires_grad = False
  
  if params["reduce_noise_leak_adv"]:
    latent_permutations = permutations(params["latent_vars"], 2)
    for perm in latent_permutations:
      for name, param in csis.guide.disentangle_mlps[f"{perm[0]}_{perm[1]}"].named_parameters():
        param.requires_grad = False
  
  csis.nstep = s
  loss = csis.step()

  if params["TC_minimization"]:
    if s >= csis.guide.gmm_fit_step:
      for name, param in csis.guide.tc_discriminator.named_parameters():
        param.requires_grad = True
      
      D_perm = csis.guide.tc_discriminator(csis.guide.fake_z_noise.detach())
      D_joint = csis.guide.tc_discriminator(csis.guide.disc_input.detach())
      ones = torch.ones(D_perm.size(0), dtype=torch.long, device=DEVICE)
      zeros = torch.zeros(D_perm.size(0), dtype=torch.long, device=DEVICE)
      disc_tc_loss = torch.mean(0.5*(F.cross_entropy(D_joint, zeros) + F.cross_entropy(D_perm, ones)))

      if s % step_size == 0:
        logger.info(f"real batches pred: {torch.mean(D_joint)}")
        logger.info(f"fake batches pred: {torch.mean(D_perm)}")

      dis_optimizer.zero_grad()
      disc_tc_loss.backward()
      dis_optimizer.step()

      disc_tc_loss = disc_tc_loss.item()
    else:
      disc_tc_loss = 0.
  

  if params["reduce_noise_leak_adv"]:
    if s >= csis.guide.gmm_fit_step:

      for perm in permutations(params["latent_vars"], 2):
        for name, param in csis.guide.disentangle_mlps[f"{perm[0]}_{perm[1]}"].named_parameters():
          param.requires_grad = True
      
      noise_leak_discriminator_loss = 0.
      var_idx = {l: i for i, l in enumerate(params["latent_vars"])}
      
      for lp in permutations(params["latent_vars"], 2):
        # lp_input = csis.guide.mask_pred.unsqueeze(-1) * csis.guide.z_noise[:, var_idx[lp[0]], :]
        # lp_target = csis.guide.mask_pred.unsqueeze(-1) * csis.guide.z_noise[:, var_idx[lp[1]], :]
        # lp_input = lp_input.detach()
        # lp_target = lp_target.detach()
        # rec = csis.guide.disentangle_mlps[lp[0]](lp_input)
        # rec_loss = F.mse_loss(rec, lp_target)
        # noise_leak_discriminator_loss += rec_loss
        
        _, L, ldim = csis.guide.z_noise.shape
        
        lp_input = csis.guide.z_noise.reshape(params["batch_size"], params["num_slots"], L, ldim)[:, :, var_idx[lp[0]], :]  # [B, N, ldim]
        lp_input = lp_input.detach()

        lp_target = csis.guide.labels[:, :, var_idx[lp[1]]]                                                                 # [B, N]  
        leak_logits = csis.guide.disentangle_mlps[f"{lp[0]}_{lp[1]}"](lp_input)                                                                                    
        leak_loss = compute_leakage_loss(leak_logits, lp_target)
        noise_leak_discriminator_loss += leak_loss
      
      leak_optimizer.zero_grad()
      noise_leak_discriminator_loss.backward()
      leak_optimizer.step()

      noise_leak_discriminator_loss = noise_leak_discriminator_loss.item()
    else:
      noise_leak_discriminator_loss = 0.

  dict_to_log = {'train_loss': loss}
    
  #if params["ood_eval"]:
  dict_to_log["alpha"] = guide.alpha
  dict_to_log["beta"] = guide.beta
  dict_to_log["gamma"] = guide.gamma
  dict_to_log["causal_loss"] = guide.causal_loss
  dict_to_log["log_det"] = guide.log_det
  dict_to_log["kl_noise"] = guide.kl_noise
  dict_to_log["assign_loss"] = guide.assignments_loss

  if params["beta"] > 0:
    dict_to_log["prior_coeff"] = csis.prior_coeff
    dict_to_log["prior_s"] = guide.prior_ss
    dict_to_log["prior_t"] = guide.prior_ts



  #if params["causal_dropout_scheduled"]:
  dict_to_log["causal_dropout_prob"] = guide.causal_dropout_prob

  for l in range(L):
    dict_to_log[f"logvar_noise_{l_to_label[l]}"] = torch.mean(guide.logvar_noise[:, l, :])
    dict_to_log[f"s_{l_to_label[l]}"] = torch.mean(torch.sum(csis.guide.ss[:, :, l, :], dim=-1))
    dict_to_log[f"t_{l_to_label[l]}"] = torch.mean(torch.sum(csis.guide.ts[:, :, l, :], dim=-1))

  
  if params["noise_logvar_constraint"]:
    dict_to_log["logvar_div"] = guide.logvar_div
  
  dict_to_log["mu_dist"] = guide.mu_dist
  dict_to_log["yc_temperature"] = guide.yc_t
  
  if params["TC_minimization"]:
    dict_to_log["latent_TC"] = guide.tc_loss
    dict_to_log["latent_TC_disc"] = disc_tc_loss
  
  if params["reduce_noise_leak_adv"]:
    dict_to_log["noise_leak_encoder_loss"] = csis.guide.noise_leak_encoder_loss
    dict_to_log["noise_leak_discriminator_loss"] = noise_leak_discriminator_loss

  dict_to_log["pred_loss"] = guide.pred_loss
  dict_to_log["log_det_prior"] = guide.log_det_prior
  #dict_to_log["label_embedd_loss"] = guide.label_embeddings_loss

  if s >= csis.guide.gmm_fit_step: 
    validation_ood_loss = csis.validation_loss()
  else:
    validation_ood_loss = 0.
  dict_to_log["validation_OOD_loss"] = validation_ood_loss
  all_val_loss.append(validation_ood_loss)
  
  if validation_ood_loss < min_val_loss:
    torch.save(csis.guide.state_dict(), GUIDE_PATH+'/guide_'+str(s)+'.pth')
    min_val_loss = validation_ood_loss 
  
  if s % step_size == 0 or s == nsteps-1: 
    # logger.info(f"step {s}/{resume_step + nsteps-1} - train_loss: {loss} - val_loss: {val_loss}")
    # dict_to_log = {'train_loss': loss, 'val_loss': val_loss}
    logger.info(f"step {s}/{resume_step + nsteps-1} - train_loss: {loss}")
    torch.save(csis.guide.state_dict(), GUIDE_PATH+'/guide_'+str(s)+'.pth')
    run.log(dict_to_log)
  
  # logger.info(f"\ntrain loss: {loss}\n\n")
     
  
logger.info(f"\nmin validation loss of {min(all_val_loss)} at iteration {all_val_loss.index(min(all_val_loss))}\n")

logger.info("\ntraining ended...")
wandb.finish()
