import torch
import numpy as np
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA

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
                z0, z1, z2, z3 = torch.split(torch.flatten(z_scm, -2), params["latent_dim"], -1)
                
                C = 1

                if C == 0:

                    # logger.info(f"e0: {z_noise[:, :, 0]}")
                    # logger.info(f"z0: {z0}")
                    z0_int = torch.flip(z0, dims=(1,))
                    # logger.info(f"z0_int: {z0_int}")

                    z_noise, _, _, _ = guide.causal_flow.backward(z_scm)

                    s1_int = guide.causal_flow.s_cond(z0_int) * guide.causal_flow.scale_gain
                    t1_int = guide.causal_flow.t_cond(z0_int) * guide.causal_flow.shift_gain
                    z1_int = torch.exp(s1_int)*z_noise[:, :, 1, :] + t1_int

                    # logger.info(f"\nz1: {z1}")
                    # logger.info(f"\ne1: {z_noise[:, :, 1, :]}")
                    # logger.info(f"\nexp(s1_int): {torch.exp(s1_int)}")
                    # logger.info(f"\nexp(s1_int)e1: {torch.exp(s1_int)*z_noise[:, :, 1, :]}")
                    # logger.info(f"\nt1_int: {t1_int}")
                    # logger.info(f"\nz1_int: {z1_int}")
                    

                    z_int = torch.cat((z0_int.unsqueeze(-2),
                                    z1_int.unsqueeze(-2),
                                    z2.unsqueeze(-2),
                                    z3.unsqueeze(-2)), dim=-2) 
                
                elif C == 1:
                    z1_perm = torch.flip(z1, dims=(1,))
                    z0_perm = torch.flatten(z0, 0, 1)
                    z0_perm = torch.flip(z0, dims=(1,))
                    s1_int = guide.causal_flow.s_cond(z0_perm) * guide.causal_flow.scale_gain
                    t1_int = guide.causal_flow.t_cond(z0_perm) * guide.causal_flow.shift_gain
                    e1_int = (z1_perm - t1_int) * torch.exp(-s1_int)

                    logger.info(f"\ne1: {e1_int}")
                    logger.info(f"\nexp(s0): {torch.exp(guide.causal_flow.s_cond(z0))}")
                    logger.info(f"\nexp(s0)*e1: {torch.exp(guide.causal_flow.s_cond(z0))*e1_int}")
                    logger.info(f"\nt0: {guide.causal_flow.t_cond(z0)}")
                    
                    
                    #e1_int = e1_int.detach()
                    z1_int = torch.exp(guide.causal_flow.s_cond(z0) * guide.causal_flow.scale_gain) * e1_int + guide.causal_flow.t_cond(z0) * guide.causal_flow.shift_gain

                    logger.info(f"\nz1: {z1}")
                    logger.info(f"\nz1_int: {z1_int}")
                    
                    
                    z_int = torch.cat((z0.unsqueeze(-2),
                                            z1_int.unsqueeze(-2),
                                            z2.unsqueeze(-2),
                                            z3.unsqueeze(-2)), dim=-2)

                x_int, _ = guide.scene_decoder(torch.flatten(z_int, -2))
                x_int = torch.clamp(x_int, 0., 1.)
                _, _, z_causal_int = guide.encoder(x_int)
                x_int_cons, _ = guide.scene_decoder(z_causal_int)
                

                fig, ax = plt.subplots(B, 4)
                for i in range(B):
                    ax[i, 0].imshow(np.clip(np.transpose(img[i].detach().cpu().numpy(), (1, 2, 0)), 0., 1.))
                    if i == 0: ax[i, 0].set_title("img")
                    ax[i, 1].imshow(np.clip(np.transpose(img_hat[i].detach().cpu().numpy(), (1, 2, 0)), 0., 1.))
                    if i == 0: ax[i, 1].set_title("img_hat")
                    ax[i, 2].imshow(np.clip(np.transpose(x_int[i].detach().cpu().numpy(), (1, 2, 0)), 0., 1.))
                    if i == 0: ax[i, 2].set_title(f"int {C}")
                    ax[i, 3].imshow(np.clip(np.transpose(x_int_cons[i].detach().cpu().numpy(), (1, 2, 0)), 0., 1.))
                    if i == 0: ax[i, 3].set_title(f"img_hat_cons")

                    for j in range(4):
                        ax[i, j].grid(False)
                        ax[i, j].axis('off')

                plt.savefig(f"{params['check_attn_folder']}/{params['latent_vars'][C]}_counterfactuals.png")
                plt.close()

                z = torch.flatten(torch.flatten(z_scm[:, :, :2], 0, 1), -2)
                z_count = torch.flatten(torch.flatten(z_int[:, :, :2], 0, 1), -2)

                pca = PCA(n_components=2)
                z_pca = pca.fit_transform(z)
                z_count_pca = pca.fit_transform(z_count)

                plt.scatter(z_pca[:, 0], z_pca[:, 1], s=2, alpha=0.5, c='blue')
                plt.scatter(z_count_pca[:, 0], z_count_pca[:, 1], s=2, alpha=0.5, c='red')
                plt.savefig(f"{params['check_attn_folder']}/{params['latent_vars'][C]}_counterfactuals_manifold.png")
                plt.close()

                #####################################################################################################

                if C == 0:
                    z_int = torch.cat((z0_int.unsqueeze(-2),
                                    z1.unsqueeze(-2),
                                    z2.unsqueeze(-2),
                                    z3.unsqueeze(-2)), dim=-2) 
                elif C == 1:
                    z_int = torch.cat((z0.unsqueeze(-2),
                                    z1_perm.unsqueeze(-2),
                                    z2.unsqueeze(-2),
                                    z3.unsqueeze(-2)), dim=-2) 
                    
                z_perm = torch.flatten(torch.flatten(z_int[:, :, :2], 0, 1), -2)

                pca = PCA(n_components=2)
                z_perm_pca = pca.fit_transform(z_perm)

                plt.scatter(z_pca[:, 0], z_pca[:, 1], s=2, alpha=0.5, c='blue')
                plt.scatter(z_perm_pca[:, 0], z_perm_pca[:, 1], s=2, alpha=0.5, c='red')
                plt.savefig(f"{params['check_attn_folder']}/{params['latent_vars'][C]}_permutations_manifold.png")
                plt.close()

                   
                x_int, _ = guide.scene_decoder(torch.flatten(z_int, -2))
                x_int = torch.clamp(x_int, 0., 1.)
                _, _, z_causal_int = guide.encoder(x_int)
                x_int_cons, _ = guide.scene_decoder(z_causal_int)

                fig, ax = plt.subplots(B, 4)
                for i in range(B):
                    ax[i, 0].imshow(np.clip(np.transpose(img[i].detach().cpu().numpy(), (1, 2, 0)), 0., 1.))
                    if i == 0: ax[i, 0].set_title("img")
                    ax[i, 1].imshow(np.clip(np.transpose(img_hat[i].detach().cpu().numpy(), (1, 2, 0)), 0., 1.))
                    if i == 0: ax[i, 1].set_title("img_hat")
                    ax[i, 2].imshow(np.clip(np.transpose(x_int[i].detach().cpu().numpy(), (1, 2, 0)), 0., 1.))
                    if i == 0: ax[i, 2].set_title(f"perm {C}")
                    ax[i, 3].imshow(np.clip(np.transpose(x_int_cons[i].detach().cpu().numpy(), (1, 2, 0)), 0., 1.))
                    if i == 0: ax[i, 3].set_title(f"img_hat_cons")

                    for j in range(4):
                        ax[i, j].grid(False)
                        ax[i, j].axis('off')

                plt.savefig(f"{params['check_attn_folder']}/{params['latent_vars'][C]}_permutation.png")
                plt.close()

                if b == 0: break
     

if __name__ == '__main__':
    main()