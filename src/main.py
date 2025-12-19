import torch
import logging
import os
import random

main_dir = os.path.abspath(__file__+'/../../')

params = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "random_seed": 1,
    "dataset_path": "/nas-ctm01/datasets/public/CG",
    "latent_vars": ["scale", "color", "x_pos", "y_pos"],
    "factor_ranges": {
        "scale": [0.2 , 1.  ],
        "color": [0.  , 1.  ],
        "x_pos": [0.15, 0.85],
        "y_pos": [0.15, 0.85]
    },
    "factor_scaling": {
        "scale": 15,
        "color": 360,
        "x_pos": 128,
        "y_pos": 128
    },
    "n_dependencies": 1,
    "train_dataset_len": 50000,
    "test_dataset_len": 5000,
    "resolution": (128, 128),
    "num_slots": 2,
    "batch_size": 512, # 256 for 32GB
    "lr": 1e-3, # 256 with 1e-4
    "training_from_scratch" : True,
    "training_epochs": 200, 
    "epoch_step": 1,
    "running_type": "train", # train, debug, eval, inspect
    "epoch_to_load" : 5, 
    "ood_eval": True,
    "TC_minimization": True, # usually True
    "alpha": 1., # usually 1.
    "beta": 1., 
    "TC_coeff": 10., # usually 10
    "log_det_coeff": 0.01, # previously 0.01
    "normalized_logdet": False,
    "mixt_kl_coeff": 0.01, # previosuly 0.01
    "logvar_reg_coeff": 0.01, # prev 0.001
    "yc_temperature": 10, # previosuly 10
    "label_embeddings_std": 1.,  # previously 1.0
    "label_embeddings_mu_mult": 1., # previously 1.0 #######################################################
    "causal_dropout_prob": 0., # 0.6 for models with causal flow, 1.0 for models without
    "slot_dim" : 16,
    "check_attn": True,
    "jobID": 1,  
    "mesh_iters": 4,
    "mesh_lr": 5, 
    "mesh_noise": 1e-3,
    "logprob_coeff": 1.,
    "plots_dir": f'{main_dir}/plots',
}

if not os.path.isdir(params['plots_dir']): os.mkdir(params['plots_dir'])

if params["check_attn"]: params["check_attn_folder"] = f"{main_dir}/check_imgs_{params['jobID']}"
