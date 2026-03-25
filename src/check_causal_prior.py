import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from torch.utils.data import DataLoader

import os
import shutil
import matplotlib.pyplot as plt


# add project path to sys to import relative modules
import sys
sys.path.insert(0, '..') # add parent dir

from models.guide import InvSlotAttentionGuide
from models.architectures import Predictor
from main import params
from datasets.data_utils import MyDataset


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
    
    A = torch.tensor([
                    [0., 0., 0., 0.],
                    [1., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                ])
    L = len(params["latent_vars"])
    ldim = params["latent_dim"]

    guide = InvSlotAttentionGuide(model_name=params["model_name"],
                                  resolution = params['resolution'],
                                  num_iterations = 3,
                                  latent_dim = params["latent_dim"],
                                  nlatents = len(params["latent_vars"]),
                                  A = A
                                  ).to(device)

    GUIDE_PATH = f"{main_dir}/checkpoint-{params['jobID']}/guide_{params['epoch_to_load']}.pth"
    guide.load_state_dict(torch.load(GUIDE_PATH, weights_only=False)["model"])
    guide.eval()
    
    #logger.info(f'\n\nseed {seed}\n\n')
    logger.info(GUIDE_PATH)

    plots_dir = os.path.abspath("set_prediction_plots")
    if not os.path.isdir(plots_dir): os.mkdir(plots_dir)
    else: 
        shutil.rmtree(plots_dir)
        os.mkdir(plots_dir)
    
    for p in guide.parameters():
        p.requires_grad = False

    
    for OOD_EVAL in [False]:

        logger.info(f"\nOOD: {OOD_EVAL}\n")

        dataset_dir = os.path.join(params["dataset_path"], "eval")

        dataset = MyDataset(dataset_dir, OOD_EVAL)
        dataloader = DataLoader(dataset, 64, shuffle=True, generator=torch.Generator(device='cuda'))

        with torch.no_grad():
            for b, (img, target) in enumerate(dataloader):
                img = img.to(device)
                target = target.to(device)

                logger.info(img.shape)

                B = img.size(0)
                
                z_noise, z_scm, img_hat, recon_figures = guide(img, target)
                cp_m, cp_logv = guide.prior_z

                cp_m = cp_m.view(-1, cp_m.size(-2), cp_m.size(-1))
                cp_logv = cp_logv.view(-1, cp_logv.size(-2), cp_logv.size(-1))
                target = target.view(-1, target.size(-1))

                for i, var in enumerate(params["latent_vars"]):
                    plt.scatter(np.arange(cp_m.size(0)), torch.norm(cp_m[:, i], dim=-1).cpu(), s=3, label=var, c=target[:, i].cpu())
                    #plt.scatter(u[:, i].cpu(), logvar_noise_prior[:, i].mean(-1).cpu(), s=2, c="red")
                plt.legend()
                plt.savefig(f"{params['check_attn_folder']}/mu_causal_prior.png")
                plt.close()

                for i, var in enumerate(params["latent_vars"]):
                    plt.scatter(np.arange(cp_m.size(0)), torch.norm(cp_logv[:, i], dim=-1).cpu(), s=3, label=var, c=target[:, i].cpu())
                plt.legend()
                plt.savefig(f"{params['check_attn_folder']}/logvar_causal_prior.png")
                plt.close()
                if b == 0: break
     

if __name__ == '__main__':
    main()