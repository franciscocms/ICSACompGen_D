import torch
import logging
import os
import random

main_dir = os.path.abspath(__file__+'/../../')

params = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "random_seed": 22, # 22
    "dataset_path": "/nas-ctm01/datasets/public/CG_linear_015", # CG_linear -> 0.5, CG_linear_015 -> 0.15
    "ID_window": 0.15,
    "latent_vars": ["scale", "color", "x_pos", "y_pos"],
    "factor_ranges": {
        "scale": [0.3, 1.],
        "color": [0.05, 0.95],
        "x_pos": [0.15, 0.85],
        "y_pos": [0.15, 0.85]
    },
    "factor_scaling": {
        "scale": 15,
        "color": 360,
        "x_pos": 64,
        "y_pos": 64
    },
    "n_dependencies": 1,
    "train_dataset_len": 50000,
    "test_dataset_len": 5000,
    "resolution": (64, 64),
    "model_name": "additiveAE",
    "num_slots": 2,
    "batch_size": 64, 
    "lr": 4e-4, 
    "use_counterfactual_consistency": False,
    "cc_epoch": 100,
    "training_from_scratch" : True,
    "training_epochs": 400, # 400 
    "epoch_step": 10,
    "running_type": "train", # train, debug, eval, inspect
    "epoch_to_load" : 400, 
    "ood_eval": True,
    "TC_minimization": False, # usually True
    "alpha": 1., # usually 1.
    "beta": 1., 
    "TC_coeff": 1000., # usually 10
    "regress_from_flatt": False,
    "normalized_logdet": False,
    "causal_dropout_prob": 0., # 0.6 for models with causal flow, 1.0 for models without
    "latent_dim": 4, # 4
    "check_attn": True,
    "jobID": 308,  
    "mesh_iters": 4,
    "mesh_lr": 5, 
    "mesh_noise": 1e-3,
    "logprob_coeff": 1.,
    "plots_dir": f'{main_dir}/plots',
}

if not os.path.isdir(params['plots_dir']): os.mkdir(params['plots_dir'])

if params["check_attn"]: params["check_attn_folder"] = f"{main_dir}/check_imgs_{params['jobID']}"
