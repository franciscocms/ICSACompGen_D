import torch
import logging
import os
import random

main_dir = os.path.abspath(__file__+'/../../')

# 87 - 5005
# 82 - 3809
# 81 - 2442

params = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "dataset": "2Dobjects",
    "resolution": (128, 128),
    "num_slots": 2,
    "training_from_scratch" : True,
    "training_epochs": 100, 
    "epoch_step": 1,
    "running_type": "train", # train, debug, eval, inspect
    "epoch_to_load" : 6000, 
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
    "slope_constraint_flag": False,
    "slope_constraint": 0.1,
    "label_embeddings_std": 1.,  # previously 1.0
    "label_embeddings_mu_mult": 1., # previously 1.0 #######################################################
    "causal_dropout_prob": 0., # 0.6 for models with causal flow, 1.0 for models without
    "alpha_beta_scheduled": False, 
    "causal_dropout_scheduled": False,
    "conditioned_classifier": True, # True for models with causal flow
    "infer_by_prediction": False,
    "slot_dim" : 64,
    "softmax_temperature": 1.0,
    "strided_convs": True,
    "check_attn": True,
    "jobID": 1000,  
    "mesh_iters": 4,
    "mesh_lr": 5, 
    "mesh_noise": 1e-3,
    "logprob_coeff": 1.,
    "inference_method": "importance_sampling_only",
    "proposals": "data_driven",
    "plots_dir": f'{main_dir}/plots',
}

params["random_seed"] = 7 # 3 worker well for n_edges = 1

# exp with almost full supp(color, shape)
# 272 - new cond. class and s = t = 0
# 273 - new cond. class and s != t != 0
# 274 - old cond. class and s = t = 0
# 275 - old cond. class and s != t != 0 (inclusive KL)

# (compare 277 with 275 - different loss)
# 277 - old cond. class and s != t != 0 (L1 loss)  --- can we see object-centric slots? yes
# 278 - no causal flow, alpha 1. and TC min (inclusive KL)
# 279 - no causal flow, alpha 1. and TC min (L1 loss)  --- can we see object-centric slots?
# 280 - no causal flow, alpha 0. and TC min (L1 loss)  --- can we see object-centric slots? yes


# exp with more extreme supp(color, shape)
# 276 = 275 causal dropout = 0.2
# 281 - old cond. class and s != t != 0 causal dropout = 0.5 (inclusive KL) 
# 282 - old cond. class and s != t != 0 causal dropout = 0.6 (inclusive KL) 
# 283 - no causal flow, alpha 1. and TC min (inclusive KL) 
# (still to run)
# 284 - no causal flow, alpha 0. and TC min (inclusive KL) 
# 285 - no causal flow, alpha 0. and no TC min (inclusive KL) 

# if params["causal_dropout_prob"] == 1.:
#     assert params["conditioned_classifier"] == False
#     assert params["causal_dropout_scheduled"] == False
#     assert params["alpha_beta_scheduled"] == False


# logger = logging.getLogger(params["running_type"])

# if params["jobID"] in [288]:
#     probs_0 = random.random()
#     probs_1 = random.random()
#     probs_2 = random.random()
#     color_to_shape_probs = {
#         0: torch.tensor([0.,      probs_0,   1-probs_0]),
#         1: torch.tensor([probs_1, 0.,        1-probs_1]),
#         2: torch.tensor([probs_2, 1-probs_2, 0.])
#         }
#     params["color_to_shape_probs"] = color_to_shape_probs
#     logger.info(f"\ncolor_to_shape_probs: {color_to_shape_probs}\n")
    


# params["latent_vars"] = ["color", "shape", "size"]
params["latent_vars"] = ["color", "shape"]
params["n_edges"] = 1
if params["n_edges"] == 1:
    if len(params["latent_vars"]) == 3:
        params["adj_matrix"] = torch.tensor([[0., 0., 0.], 
                                            [1., 0., 0.],
                                            [0., 0., 0.]])
    elif len(params["latent_vars"]) == 2:
        params["adj_matrix"] = torch.tensor([[0., 0.,], 
                                             [1., 0., ]])


if params["n_edges"] == 2:
    params["dag_type"] = "root" # "chain", "fork"
    params["adj_matrix"] = torch.tensor([[0., 0., 0.], 
                                         [1., 0., 0.],
                                         [1., 0., 0.]])
    

params["color_vals"] = ["red", "green", "blue"]
params["shape_vals"] = ["ball", "square", "triangle"]
if "size" in params["latent_vars"]:
    params["size_vals"] = ["small", "medium", "large"]
else: params["size_vals"] = ["large"]

# set job split settings for inference
JOB_SPLIT = {
            'id': 4,
            'total': 4
            }

if params["dataset"] == "clevr": 
    params["batch_size"] = 64 if params["running_type"] == "train" else 1
    params["lr"] = 4e-4
elif params["dataset"] == "2Dobjects": 
    params["batch_size"] = 256
    params["lr"] = 1e-4 #j221 1e-3, j222 3e-4, j223 1e-4
else: raise ValueError(f"Dataset error: dataset named {params['dataset']} not found!")

if not os.path.isdir(params['plots_dir']): os.mkdir(params['plots_dir'])


if params["check_attn"]: params["check_attn_folder"] = f"{main_dir}/check_imgs_{params['jobID']}"
