import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import shutil
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# add project path to sys to import relative modules
import sys
sys.path.insert(0, '..') # add parent dir

from models.guide import InvSlotAttentionGuide
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
    guide.load_state_dict(torch.load(GUIDE_PATH))
    guide.eval()
    
    #logger.info(f'\n\nseed {seed}\n\n')
    logger.info(GUIDE_PATH)

    plots_dir = os.path.abspath("set_prediction_plots")
    if not os.path.isdir(plots_dir): os.mkdir(plots_dir)
    else: 
        shutil.rmtree(plots_dir)
        os.mkdir(plots_dir)

    for OOD_EVAL in [False]:

        logger.info(f"\nOOD: {OOD_EVAL}\n")

        dataset_dir = "/nas-ctm01/datasets/public/CG/eval"

        dataset = MyDataset(dataset_dir, OOD_EVAL)
        dataloader = DataLoader(dataset, 64, shuffle=False, generator=torch.Generator(device='cuda'))

        noise_preds = torch.tensor([])  
        flat_noise_preds = torch.tensor([])  
        targets = torch.tensor([])      
        with torch.no_grad():
            for img, target in dataloader:
                img = img.to(device)
                target = target.to(device)
                
                batch_noise_preds, batch_flat_noise_preds = guide(img, target) # [B, N, L]
                
                noise_preds = torch.cat((noise_preds, batch_noise_preds), dim=0)
                flat_noise_preds = torch.cat((flat_noise_preds, batch_flat_noise_preds), dim=0)

                targets = torch.cat((targets, target), dim=0)             
        
        logger.info(noise_preds.shape)
        logger.info(targets.shape)
        
        for i, var in enumerate(params["latent_vars"]):            
            logger.info(f"\nprobe for variable {params['latent_vars'][i]}\n")
            
            # r2_score, a_target, a_preds = hungarian_r2_score(noise_preds[:, :, i], targets[:, :, i])
            r2_score, a_target, a_preds = hungarian_r2_score(flat_noise_preds[:, :, i], targets[:, :, i])

            plt.scatter(a_target, a_preds, s=2)
            plt.savefig(f"{params['check_attn_folder']}/preds_{params['latent_vars'][i]}.png")
            plt.close()

            logger.info(f"test r2 score: {r2_score}")
        

if __name__ == '__main__':
    main()