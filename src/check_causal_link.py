import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from torch.utils.data import DataLoader

import os
import shutil


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

    # A = torch.tensor([
    #     [0., 0.],
    #     [1., 0.],
    # ])

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
        dataloader = DataLoader(dataset, 64, shuffle=False, generator=torch.Generator(device='cuda'))

        def causal_link(z0, e1):
            s1 = guide.causal_flow.s_cond(z0)
            t1 = guide.causal_flow.t_cond(z0)
            z1 = torch.exp(s1)*e1 + t1
            return z1

        with torch.no_grad():
            for i, (img, target) in enumerate(dataloader):
                img = img.to(device)
                target = target.to(device)

                B = img.size(0)
                
                _, _, z_noise = guide.noise_encoder(img)
                z_noise = z_noise.reshape(z_noise.size(0), z_noise.size(1), len(params["latent_vars"]), params["latent_dim"])       

                s0 = guide.causal_flow.s_base
                t0 = guide.causal_flow.t_base
                z0 = torch.exp(s0)*z_noise[:, :, 0, :] + t0
                
                inputs = (z0, z_noise[:, :, 1, :])

                Js = jacobian(causal_link, inputs, vectorize=True)
                for J in Js:

                    logger.info(J)
                    logger.info(J.shape)

                if i == 0: break
     

if __name__ == '__main__':
    main()