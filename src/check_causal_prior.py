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
        dataloader = DataLoader(dataset, 2, shuffle=True, generator=torch.Generator(device='cuda'))

        with torch.no_grad():
            for v, var in enumerate(params["latent_vars"]):

                for b, (img, target) in enumerate(dataloader):
                    img = img.to(device)
                    target = target.to(device)

                    got_it = False
                    if 0 <= target[0, 0, v] <= 0.1 and 0.9 <= target[0, 1, v] <= 1.:
                        got_it = True

                        logger.info(f"\nrunning for {var}...")

                        logger.info(f"small object {var}: {target[0, 0, v]}")
                        logger.info(f"large object {var}: {target[0, 1, v]}")
                        
                        z_noise, z_scm, img_hat, recon_figures = guide(img, target)
                        
                        if v == 1:
                            z0_small = z_scm[0, 0, 0, :]
                            z0_large = z_scm[0, 1, 0, :]
                        
                        alphas = torch.arange(0., 1., 0.01)
                        means = []
                        logvars = []
                        u_i = torch.rand((1, 1, 1, 1)).expand((alphas.size(0), -1, -1, -1))
                        for i, alpha in enumerate(alphas):
                            if v == 1:
                                z0 = (1-alpha)*z0_small + alpha*z0_large
                                pa_zi = z0
                            else:
                                pa_zi = torch.zeros_like((z_scm[0, 0, 0]))

                            # s1 = guide.causal_flow.s_cond(z0.reshape((1, 1, z0.size(-1)))) * guide.causal_flow.scale_gain
                            # t1 = guide.causal_flow.t_cond(z0.reshape((1, 1, z0.size(-1)))) * guide.causal_flow.shift_gain
                            # z1 = torch.exp(s1)*z_noise[0, 0, 1, :] + t1

                            # for fixed color label, vary z_scale
                            freq_coeff = (v - 6) * np.pi
                            if v == 1:
                                zu_i = torch.cat((pa_zi.reshape((1, 1, pa_zi.size(-1))),
                                                u_i[i]), dim=-1)
                            else:
                                zu_i = torch.cat((pa_zi.reshape((1, 1, pa_zi.size(-1))),
                                                torch.tensor(alpha[i]).unsqueeze(0).unsqueeze(0).unsqueeze(0)), dim=-1)

                            # zu_i = torch.cat((z0.reshape((1, 1, z0.size(-1))),
                            #                   u_i[i]
                                            #torch.sin(freq_coeff * u_i[i])), dim=-1)
                                            #torch.sin(freq_coeff * u_i[i] * torch.norm(z0.reshape((1, 1, z0.size(-1))), 2))), dim=-1)
                            mean, logvar = guide.prior.net[str(v)](zu_i)
                            #logvar = logvar - 3

                            means.append(mean)
                            logvars.append(logvar)
                        means = torch.stack(means)
                        logvars = torch.stack(logvars)

                        logger.info(means.shape)
                        logger.info(logvars.shape) 

                        
                        #plt.scatter(alphas.cpu(), torch.norm(means, dim=-1).cpu(), s=12, label="mean")
                        for d in range(means.size(-1)):
                            plt.scatter(alphas.cpu(), means.squeeze()[:, d].cpu(), s=12, label=f"mean {d}")
                        #plt.plot(alphas.cpu(), u_i.squeeze().cpu(), label=f"u_color = {torch.round(u_i, decimals=2).squeeze()[0]}")
                        plt.legend()
                        if v == 1: plt.xlabel("z_scale interpolation (small to large)")
                        else: plt.xlabel(f"u_{var}")
                        plt.savefig(f"{params['check_attn_folder']}/{var}_mu_causal_prior.png")
                        plt.close()

                        #plt.scatter(alphas.cpu(), torch.norm(logvars, dim=-1).cpu(), s=12, label="logvar")
                        for d in range(logvars.size(-1)):
                            plt.scatter(alphas.cpu(), logvars.squeeze()[:, d].cpu(), s=12, label=f"logvar {d}")
                        #plt.plot(alphas.cpu(), u_i.squeeze().cpu(), label=f"u_color = {torch.round(u_i, decimals=2).squeeze()[0]}")
                        if v == 1: plt.xlabel("z_scale interpolation (small to large)")
                        else: plt.xlabel(f"u_{var}")
                        plt.legend()
                        plt.savefig(f"{params['check_attn_folder']}/{var}_logvar_causal_prior.png")
                        plt.close()

                    
                    
                    if got_it: break
     

if __name__ == '__main__':
    main()