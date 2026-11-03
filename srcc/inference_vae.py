import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import shutil
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# add project path to sys to import relative modules
import sys
sys.path.insert(0, '..') # add parent dir

from models.architectures import VAE, Probe
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
    
    model = VAE(num_slots=params["num_slots"],
                L=len(params["latent_vars"]), 
                ldim=params["latent_dim"],
                stage=params["running_type"]).to(device)

    GUIDE_PATH = f"{main_dir}/checkpoint-{params['jobID']}/guide_{params['epoch_to_load']}.pth"
    model.load_state_dict(torch.load(GUIDE_PATH, weights_only=False)["model"])
    model.eval()
    
    #logger.info(f'\n\nseed {seed}\n\n')
    logger.info(GUIDE_PATH)

    plots_dir = os.path.abspath("set_prediction_plots")
    if not os.path.isdir(plots_dir): os.mkdir(plots_dir)
    else: 
        shutil.rmtree(plots_dir)
        os.mkdir(plots_dir)
    
    for p in model.parameters():
        p.requires_grad = False

    figure, ax_m = plt.subplots()
    
    for OOD_EVAL in [False, True]:

        logger.info(f"\nOOD: {OOD_EVAL}\n")

        dataset_dir = os.path.join(params["dataset_path"], "eval")

        dataset = MyDataset(dataset_dir, OOD_EVAL)
        dataloader = DataLoader(dataset, 64, shuffle=False, generator=torch.Generator(device='cuda'))

        z_noise = torch.tensor([])  
        z_scm = torch.tensor([])  
        targets = torch.tensor([])  
        mse = torch.tensor([])
        all_img = torch.tensor([])
        all_img_hat = torch.tensor([])
        recon_loss = 0.  
        r2 = []
        z_preds = torch.tensor([]) 
        with torch.no_grad():
            for i, (img, target) in enumerate(dataloader):
                img = img.to(device)
                target = target.to(device)
                
                img_hat, recon_figures, mean, logvar, z = model(img) # [B, N, L]
                recon_loss += F.mse_loss(img_hat, img)
                mse = torch.cat((mse, F.mse_loss(img_hat, img, reduction='none').mean((-1, -2, -3))), dim=0)
                targets = torch.cat((targets, target), dim=0)  
                all_img = torch.cat((all_img, img), dim=0)
                all_img_hat = torch.cat((all_img_hat, img_hat), dim=0)

        #logger.info(z_noise.shape) # [B, N, L, d] -> B = 5k
        logger.info(targets.shape)
        logger.info(f"MSE: {recon_loss / (i+1)}")
        
        for b in range(all_img.size(0)):
            r2.append(r2_score(all_img[b].cpu().numpy().flatten(), all_img_hat[b].cpu().numpy().flatten()))

        
        #logger.info(f"recon r2: {r2_score(all_img.cpu().numpy().flatten(), all_img_hat.cpu().numpy().flatten())}")
        r2 = np.array(r2)
        r2 = np.clip(r2, 0., 1.)
        logger.info(f"recon r2: {r2.mean()} - {r2.std()} - {np.median(r2)}")
        
        u_scale = torch.flatten(targets[:, :, 0], 0, 1).cpu().numpy()
        u_color = torch.flatten(targets[:, :, 1], 0, 1).cpu().numpy()
        mse = mse.repeat_interleave(params["num_slots"])

        mse = torch.clamp(mse, 0., 0.03)
        logger.info(torch.max(mse))
        
        mse = mse.cpu().numpy()

        scatter = ax_m.scatter(u_scale, u_color, s=10, c=mse, alpha=0.5, cmap="YlGn", vmin=0., vmax=mse.max()) # vmax=
        if OOD_EVAL: cb = figure.colorbar(scatter, ax=ax_m)
        
        save_recons = False
        if save_recons:
            n_figures = 5
            assert img.size(0) >= n_figures
            fig, ax = plt.subplots(n_figures, 2 + params["num_slots"])
            for i in range(n_figures):
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
        
        compute_prediction_performance = False
        if compute_prediction_performance:

            for i, var in enumerate(params["latent_vars"]):            
                logger.info(f"\npredictions for variable {params['latent_vars'][i]}\n")
                r2_pred, _, _ = hungarian_r2_score(z_preds[:, :, i], targets[:, :, i])
                logger.info(f"\nprediction r2 score: {r2_pred}\n")
                # r2_pred, _, _ = hungarian_r2_score(z_scm[:, :, i, 0], targets[:, :, i])
                # logger.info(f"\nprediction r2 score (without regressor): {r2_pred}\n")

        make_regression = True
        if make_regression:
            avg_pred_r2 = []
            for i, var in enumerate(params["latent_vars"]):            
                logger.info(f"\nprobe for variable {params['latent_vars'][i]}\n")
                
                probe = Probe(nin = params["latent_dim"], nhidden = 64, nout = 1)
                
                train_set = ProbeDataset(z[:4000], targets[:4000], var_idx=i, split="train")
                test_set = ProbeDataset(z[4000:], targets[4000:], var_idx=i, split="test")
                train_loader = DataLoader(train_set, 64, shuffle=True, generator=torch.Generator(device='cuda'))
                test_loader = DataLoader(test_set, len(test_set), shuffle=False, generator=torch.Generator(device='cuda'))

                optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.001)
                
                for epoch in range(101):
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
                            r2_pred, a_target, a_preds = hungarian_r2_score(pred, target)
                    train_loss /= train_iters
                    # dict_to_log = {#f"{OOD_EVAL}_probe_{params['latent_vars'][i]}_train_loss": train_loss,
                    #                f"{OOD_EVAL}_probe_{params['latent_vars'][i]}_test_r2": r2_pred}
                    
                    # run.log(dict_to_log)
                    
                save_preds = False
                if save_preds:
                    plt.scatter(a_target, a_preds, s=2)
                    if OOD_EVAL: plt.savefig(f"{params['check_attn_folder']}/preds_{params['latent_vars'][i]}_ood.png")
                    else: plt.savefig(f"{params['check_attn_folder']}/preds_{params['latent_vars'][i]}_id.png")
                    plt.close()

                logger.info(f"test r2 score: {r2_pred}")
                avg_pred_r2.append(r2_pred)
            

                #if i == 1: break
    
            logger.info(f"prediction r2: {avg_pred_r2} with mean {round(np.asarray(avg_pred_r2).mean(), 3)} and std {round(np.asarray(avg_pred_r2).std(), 3)}")
    
    # wandb.finish()
    plt.savefig(f"{params['check_attn_folder']}/mse.png")
    plt.close()    

if __name__ == '__main__':
    main()