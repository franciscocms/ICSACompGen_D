import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import shutil
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# add project path to sys to import relative modules
import sys
sys.path.insert(0, '..') # add parent dir

from models.guide import InvSlotAttentionGuide
from models.architectures import Predictor
from main import params
from datasets.data_utils import MyDataset, ProbeDataset
from utils.loss import hungarian_r2_score, compute_matching_cost

import wandb # type: ignore

import logging
logfile_name = f"eval-{params['jobID']}.log"
logger = logging.getLogger(params["running_type"])
logger.setLevel(logging.INFO)
fh = logging.FileHandler(logfile_name, mode='w')
logger.addHandler(fh)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_device(device)
main_dir = os.path.abspath(__file__+'/../')

def main():    

    assert params["running_type"] == "eval"



    logger.info(device)
    
    # A = torch.tensor([
    #                 [0., 0., 0., 0.],
    #                 [1., 0., 0., 0.],
    #                 [0., 0., 0., 0.],
    #                 [0., 0., 0., 0.],
    #             ])

    A = torch.tensor([
        [0., 0.],
        [1., 0.],
    ])

    guide = InvSlotAttentionGuide(model_name=params["model_name"],
                                  resolution = params['resolution'],
                                  num_iterations = 3,
                                  latent_dim = params["latent_dim"],
                                  nlatents = len(params["latent_vars"]),
                                  A = A
                                  ).to(device)

    GUIDE_PATH = f"{main_dir}/checkpoint-{params['jobID']}/guide_{params['epoch_to_load']}.pth"
    #guide.load_state_dict(torch.load(GUIDE_PATH, weights_only=False)["model"])
    guide.eval()
    
    #logger.info(f'\n\nseed {seed}\n\n')
    logger.info(GUIDE_PATH)

    plots_dir = os.path.abspath("set_prediction_plots")
    if not os.path.isdir(plots_dir): os.mkdir(plots_dir)
    else: 
        shutil.rmtree(plots_dir)
        os.mkdir(plots_dir)
    
    with torch.no_grad():
        u = torch.arange(0., 1., 0.001).unsqueeze(-1).unsqueeze(-1).repeat(1, params["num_slots"], len(params["latent_vars"])).to(device)
        mu_noise_prior, logvar_noise_prior = guide.noise_prior(u)

        u = u.view(-1, u.size(-1))
        mu_noise_prior = mu_noise_prior.view(-1, mu_noise_prior.size(-2), mu_noise_prior.size(-1))
        logvar_noise_prior = logvar_noise_prior.view(-1, logvar_noise_prior.size(-2), logvar_noise_prior.size(-1))
        
        # for i, var in enumerate(params["latent_vars"]):
        #     logger.info(f"var {i} mu - {mu_noise_prior[[0, 500, 1000, 1500, 1999], i, :]}")
        #     #plt.scatter(u[:, i].cpu(), torch.norm(mu_noise_prior[:, i], dim=-1).cpu(), s=2, label=var)
        #     for j in range(mu_noise_prior.size(-1)):
        #         plt.scatter(u[:, i].cpu(), mu_noise_prior[:, i, j].cpu(), s=2, label=f"mean {j}")
        #     plt.legend()
        #     plt.savefig(f"{params['check_attn_folder']}/{var}_mu_prior.png")
        #     plt.close()


        #     logger.info(f"var {i} logvar - {logvar_noise_prior[[0, 500, 1000, 1500, 1999], i, :]}")
        #     #plt.scatter(u[:, i].cpu(), torch.norm(logvar_noise_prior[:, i], dim=-1).cpu(), s=2, label=var)
        #     for j in range(logvar_noise_prior.size(-1)):
        #         plt.scatter(u[:, i].cpu(), logvar_noise_prior[:, i, j].cpu(), s=2, label=f"logvar {j}")
        #     plt.legend()
        #     plt.savefig(f"{params['check_attn_folder']}/{var}_logvar_prior.png")
        #     plt.close()
        
        for i, var in enumerate(params["latent_vars"]):
            if mu_noise_prior.size(-1) > 1: plt.scatter(u[:, i].cpu(), torch.norm(mu_noise_prior[:, i], dim=-1).cpu(), s=2, label=var)
            else: plt.scatter(u[:, i].cpu(), mu_noise_prior[:, i, 0].cpu(), s=2, label=var)
        plt.legend()
        plt.savefig(f"{params['check_attn_folder']}/mu_prior.png")
        plt.close()


        for i, var in enumerate(params["latent_vars"]):
            if logvar_noise_prior.size(-1) > 1: plt.scatter(u[:, i].cpu(), torch.norm(logvar_noise_prior[:, i], dim=-1).cpu(), s=2, label=var)
            else: plt.scatter(u[:, i].cpu(), logvar_noise_prior[:, i, 0].cpu(), s=2, label=var)
        plt.legend()
        plt.savefig(f"{params['check_attn_folder']}/logvar_prior.png")
        plt.close()

        logger.info(mu_noise_prior)
        logger.info(logvar_noise_prior)


    

if __name__ == '__main__':
    main()