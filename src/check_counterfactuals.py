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
        dataloader = DataLoader(dataset, 2, shuffle=False, generator=torch.Generator(device='cuda'))

        def causal_link(z0, e1):
            s1 = guide.causal_flow.s_cond(z0)
            t1 = guide.causal_flow.t_cond(z0)
            z1 = torch.exp(s1)*e1 + t1
            return z1

        with torch.no_grad():
            for i, (img, target) in enumerate(dataloader):
                img = img.to(device)
                target = target.to(device)

                logger.info(img.shape)

                B = img.size(0)
                
                z_noise, z_scm, img_hat, recon_figures = guide(img, target)
                z0, z1, z2, z3 = torch.split(torch.flatten(z_scm, -2), params["latent_dim"], -1)

                logger.info(f"z0: {z0}")
                z0_int = torch.flip(z0, dims=(1,))
                logger.info(f"z0_int: {z0_int}")
                
                s1_int = guide.causal_flow.s_cond(z0_int)
                t1_int = guide.causal_flow.t_cond(z0_int)
                z1_int = torch.exp(s1_int)*z_noise[:, :, 1, :] + t1_int

                logger.info(f"\nexp(s1_int): {torch.exp(s1_int).mean()} -- {torch.exp(s1_int).std()}")
                logger.info(f"\ne1: {z_noise[:, :, 1, :].mean()} -- {z_noise[:, :, 1, :].std()}")

                z_int = torch.cat((z0_int.unsqueeze(-2),
                                   z1_int.unsqueeze(-2),
                                   z2.unsqueeze(-2),
                                   z3.unsqueeze(-2)), dim=-2) 
                   
                x_int, _ = guide.scene_decoder(torch.flatten(z_int, -2))
                x_int = torch.clamp(x_int, 0., 1.)

                mu_noise_int, logvar_noise_int, z_noise_int = guide.noise_encoder(x_int)
                mu_noise_int = mu_noise_int.reshape(mu_noise_int.size(0), mu_noise_int.size(1), L, ldim)
                logvar_noise_int = logvar_noise_int.reshape(logvar_noise_int.size(0), logvar_noise_int.size(1), L, ldim)
                z_noise_int = z_noise_int.reshape(z_noise_int.size(0), z_noise_int.size(1), L, ldim)
                z_causal_int, _, _, _ = guide.causal_flow(z_noise_int, 0, False)

                with torch.no_grad():  
                    x_int_cons, _ = guide.scene_decoder(torch.flatten(z_causal_int, -2))

                fig, ax = plt.subplots(B, 4)
                for i in range(B):
                    ax[i, 0].imshow(np.clip(np.transpose(img[i].detach().cpu().numpy(), (1, 2, 0)), 0., 1.))
                    if i == 0: ax[i, 0].set_title("img")
                    ax[i, 1].imshow(np.clip(np.transpose(img_hat[i].detach().cpu().numpy(), (1, 2, 0)), 0., 1.))
                    if i == 0: ax[i, 1].set_title("img_hat")
                    ax[i, 2].imshow(np.clip(np.transpose(x_int[i].detach().cpu().numpy(), (1, 2, 0)), 0., 1.))
                    if i == 0: ax[i, 2].set_title(f"int 0")
                    ax[i, 3].imshow(np.clip(np.transpose(x_int_cons[i].detach().cpu().numpy(), (1, 2, 0)), 0., 1.))
                    if i == 0: ax[i, 3].set_title(f"img_hat_cons")

                    for j in range(4):
                        ax[i, j].grid(False)
                        ax[i, j].axis('off')

                plt.savefig(f"{params['check_attn_folder']}/counterfactuals.png")
                plt.close()

                if i == 0: break
     

if __name__ == '__main__':
    main()