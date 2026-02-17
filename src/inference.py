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

    figure, ax_m = plt.subplots()
    
    for OOD_EVAL in [False, True]:

        logger.info(f"\nOOD: {OOD_EVAL}\n")

        dataset_dir = os.path.join(params["dataset_path"], "eval")

        dataset = MyDataset(dataset_dir, OOD_EVAL)
        dataloader = DataLoader(dataset, 64, shuffle=False, generator=torch.Generator(device='cuda'))

        z_noise = torch.tensor([])  
        targets = torch.tensor([])  
        mse = torch.tensor([])
        all_img = torch.tensor([])
        all_img_hat = torch.tensor([])
        recon_loss = 0.  
        with torch.no_grad():
            for i, (img, target) in enumerate(dataloader):
                img = img.to(device)
                target = target.to(device)
                
                batch_z_noise, img_hat, recon_figures = guide(img, target) # [B, N, L]
                recon_figures = recon_figures.permute(1, 0, 2, 3, 4) # shape is (N, B, C, W, H)
                recon_loss += F.mse_loss(img_hat, img)
                mse = torch.cat((mse, F.mse_loss(img_hat, img, reduction='none').mean((-1, -2, -3))), dim=0)
                z_noise = torch.cat((z_noise, batch_z_noise), dim=0)
                targets = torch.cat((targets, target), dim=0)  
                all_img = torch.cat((all_img, img), dim=0)
                all_img_hat = torch.cat((all_img_hat, img_hat), dim=0)

        logger.info(z_noise.shape) # [B, N, L, d] -> B = 5k
        logger.info(targets.shape)
        logger.info(f"MSE: {recon_loss / (i+1)}")
        logger.info(f"recon r2: {r2_score(all_img.cpu().numpy().flatten(), all_img_hat.cpu().numpy().flatten())}")
        
        u_scale = torch.flatten(targets[:, :, 0], 0, 1).cpu().numpy()
        u_color = torch.flatten(targets[:, :, 1], 0, 1).cpu().numpy()
        mse = mse.repeat_interleave(params["num_slots"])
        
        
        scatter = ax_m.scatter(u_scale, u_color, s=2, c=mse.cpu().numpy(), cmap="Wistia", vmin=0., vmax=0.05)
        if OOD_EVAL: cb = figure.colorbar(scatter, ax=ax_m)
        

        if False:
            z, _, _, _, _ = guide.causal_flow(z_noise)

            # split in z_scale, z_color, ....
            z_scale, z_color, z_x, z_y = torch.flatten(z, -2, -1).split(params["latent_dim"], -1)
            z_scale = z_scale.detach()
            z_color = z_color.detach()
            logger.info(z_color[0, 0])
            z_color = torch.randn_like(z_scale).to(device)
            z_color.requires_grad = True
            optimizer = torch.optim.Adam([z_color], lr=1.)

            for i in range(301):
                
                optimizer.zero_grad()

                e_color = (z_color - guide.causal_flow.t_cond(z_scale)) * torch.exp(-guide.causal_flow.s_cond(z_scale))
                loss = F.mse_loss(e_color, z_noise[:, :, 1])

                if i % 50 == 0: logger.info(f"iter {i} - loss: {loss}")
                #logger.info(z_color[0, 0])

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                slots = guide.causal_decoder(torch.cat((z_scale.unsqueeze(-2), z_color.unsqueeze(-2), z_x.unsqueeze(-2), z_y.unsqueeze(-2)), dim=-2))
                img_hat, recon_figures = guide.scene_decoder(slots)
                recon_figures = recon_figures.permute(1, 0, 2, 3, 4) # shape is (N, B, C, W, H)
        
        save_recons = False
        if save_recons:
            fig, ax = plt.subplots(5, 2 + params["num_slots"])
            for i in range(5):
                ax[i, 0].imshow(np.transpose(img[i].detach().cpu().numpy(), (1, 2, 0)))
                if i == 0: ax[i, 0].set_title("img")
                ax[i, 1].imshow(np.transpose(img_hat[i].detach().cpu().numpy(), (1, 2, 0)))
                if i == 0: ax[i, 1].set_title("img_hat")

                for j in range(params["num_slots"]):
                    ax[i, j+2].imshow(np.transpose(recon_figures[j, i].detach().cpu().numpy(), (1, 2, 0)))
                    if i == 0: ax[i, j+2].set_title(f"figs {j}")
                
                for j in range(2 + params["num_slots"]):
                    ax[i, j].grid(False)
                    ax[i, j].axis('off')  

            if OOD_EVAL: plt.savefig(f"{params['check_attn_folder']}/img_recon_ood.png")
            else: plt.savefig(f"{params['check_attn_folder']}/img_recon_id.png")
            plt.close()  
        
        make_regression = True
        if make_regression:
            for i, var in enumerate(params["latent_vars"]):            
                logger.info(f"\nprobe for variable {params['latent_vars'][i]}\n")
                
                probe = Predictor(params["latent_dim"],
                                params["latent_dim"],
                                1)
                
                train_set = ProbeDataset(z_noise[:4000], targets[:4000], var_idx=i, split="train")
                test_set = ProbeDataset(z_noise[4000:], targets[4000:], var_idx=i, split="test")
                train_loader = DataLoader(train_set, 64, shuffle=True, generator=torch.Generator(device='cuda'))
                test_loader = DataLoader(test_set, len(test_set), shuffle=False, generator=torch.Generator(device='cuda'))

                optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.001)
                
                for epoch in range(10):
                    train_loss = 0.
                    train_iters = 0
                    for e, target in train_loader:
                        e = e.to(device)
                        target = target.to(device)

                        pred = probe(e[:, :, i])
                        if pred.size(-1) == 1: pred = pred.squeeze(-1)
                        batch_loss, _ = compute_matching_cost(pred, target) 

                        optimizer.zero_grad()
                        batch_loss.backward()
                        optimizer.step()

                        train_loss += batch_loss.item()
                        train_iters += 1
                    
                    with torch.no_grad():
                        for e, target in test_loader:
                            e = e.to(device)
                            target = target.to(device)

                            pred = probe(e[:, :, i])
                            if pred.size(-1) == 1: pred = pred.squeeze(-1)
                            r2_score, a_target, a_preds = hungarian_r2_score(pred, target)

                    train_loss /= train_iters
                    if epoch % 50 == 0:
                        logger.info(f"epoch {epoch}:")
                        logger.info(f"train loss {train_loss} - test r2_score: {r2_score}")
                    
                save_preds = False
                if save_preds:
                    plt.scatter(a_target, a_preds, s=2)
                    if OOD_EVAL: plt.savefig(f"{params['check_attn_folder']}/preds_{params['latent_vars'][i]}_ood.png")
                    else: plt.savefig(f"{params['check_attn_folder']}/preds_{params['latent_vars'][i]}_id.png")
                    plt.close()

                logger.info(f"test r2 score: {r2_score}")

                if i == 1: break
    
    
    plt.savefig(f"{params['check_attn_folder']}/mse.png")
    plt.close()    

if __name__ == '__main__':
    main()