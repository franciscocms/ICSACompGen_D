import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import json
import shutil
import time

# add project path to sys to import relative modules
import sys
sys.path.append(os.path.abspath(__file__+'/../../'))

main_dir = os.path.abspath(__file__+'/../../')

from main import models
from main import modifiedCSIS as mcsis
from main import modifiedImportance as mImportance
from utils.guide import load_trained_guide
from utils.generate import img_to_tensor, render
from inference.guide import InvSlotAttentionGuide
from utils.distributions import Empirical, MyNormal
from utils.baseline import compute_AP
from main.setup import params

import logging
logfile_name = f"eval-{params['jobID']}.log"
logger = logging.getLogger(params["running_type"])
logger.setLevel(logging.INFO)
fh = logging.FileHandler(logfile_name, mode='w')
logger.addHandler(fh)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_device(device)
main_dir = os.path.abspath(__file__+'/../../')

shape_vals = {v: params["shape_vals"].index(v) for v in params["shape_vals"]}
size_vals = {v: params["size_vals"].index(v) for v in params["size_vals"]}
color_vals = {v: params["color_vals"].index(v) for v in params["color_vals"]}

shape_lib = list(shape_vals.keys())
size_lib = list(size_vals.keys())
color_lib = list(color_vals.keys())

img_transform = transforms.Compose([transforms.ToTensor()])

PRINT_INFERENCE_TIME = False

def process_preds(trace, id):
    
    """ returns a matrix of predictions from the proposed trace """
    if "size" in params["latent_vars"]:
        features_dim = len(shape_vals) + len(size_vals) + len(color_vals) + 3
    else:
        features_dim = len(shape_vals) + len(color_vals) + 3

    preds = torch.zeros(params['max_objects'], features_dim) # 11 - the dimension of all latent variables (countinuous and discrete after OHE)
    for name, site in trace.nodes.items():
        if site['type'] == 'sample':          
            s = len(shape_vals)
            if name == 'shape': preds[:, :s] = F.one_hot(site['value'][id], len(shape_vals))
            
            if "size" in params["latent_vars"]:
                if name == 'size':  preds[:, s:s+len(size_vals)] = F.one_hot(site['value'][id], len(size_vals))
                s = s+len(size_vals)

            if name == 'color': preds[:, s:s+len(color_vals)] = F.one_hot(site['value'][id], len(color_vals))
            s = s+len(color_vals)
            if name == 'locX': preds[:, s] = site['value'][id]
            if name == 'locY': preds[:, s+1] = site['value'][id]
            if name == 'mask': preds[:, s+2] = site['value'][id]
    return preds

def process_preds_argmax(trace, id):
    
    """ returns a matrix of predictions from the proposed trace """
    if "size" in params["latent_vars"]:
        features_dim = len(shape_vals) + len(size_vals) + len(color_vals) + 3
    else:
        features_dim = len(shape_vals) + len(color_vals) + 3

    preds = torch.zeros(params['max_objects'], features_dim) # 11 - the dimension of all latent variables (countinuous and discrete after OHE)
    for name, site in trace.nodes.items():
        if site['type'] == 'sample':              
            s = len(shape_vals)
            if name == 'shape': 
                preds[:, :s] = F.one_hot(torch.argmax(site['fn'].probs[id], dim=-1), len(shape_vals))            
            if "size" in params["latent_vars"]:
                if name == 'size':  
                    preds[:, s:s+len(size_vals)] = F.one_hot(torch.argmax(site['fn'].probs[id], dim=-1), len(size_vals))
                s = s+len(size_vals)
            if name == 'color': 
                preds[:, s:s+len(color_vals)] = F.one_hot(torch.argmax(site['fn'].probs[id], dim=-1), len(color_vals))            
            s = s+len(color_vals)
            if name == 'locX': preds[:, s] = site['value'][id]
            if name == 'locY': preds[:, s+1] = site['value'][id]
            if name == 'mask': preds[:, s+2] = site['value'][id]
    return preds

def process_targets(target_dict):   
    
    n = int(target_dict['scene_attr']['N'])

    if "size" in params["latent_vars"]:
        target = torch.zeros(params['max_objects'], len(shape_vals) + len(size_vals) + len(color_vals) + 3) 
    else:
        target = torch.zeros(params['max_objects'], len(shape_vals) + len(color_vals) + 3) 

    for i in range(n):
        s = len(shape_vals)
        target[i, :s] = F.one_hot(torch.tensor(shape_vals[target_dict['scene_attr'][f'object_{i}']['shape']]), len(shape_vals))
        
        if "size" in params["latent_vars"]:
            target[i, s:s+len(size_vals)] = F.one_hot(torch.tensor(size_vals[target_dict['scene_attr'][f'object_{i}']['size']]), len(size_vals))
            s = s+len(size_vals)
        
        target[i, s:s+len(color_vals)] = F.one_hot(torch.tensor(color_vals[target_dict['scene_attr'][f'object_{i}']['color']]), len(color_vals))
        s = s+len(color_vals)
        target[i, s] = target_dict['scene_attr'][f'object_{i}']['initLocX']
        target[i, s+1] = target_dict['scene_attr'][f'object_{i}']['initLocY']
        target[i, s+2] = torch.tensor(1.)
    return target
    
def main():    

    assert params["running_type"] == "eval"

    logger.info(device)

    seeds = [1]

    OOD_EVAL = params["ood_eval"]

    threshold = [-1.]
    all_mAP = {k: [] for k in threshold}

    if not isinstance(params["jobID"], list):
        params["jobID"] = [params['jobID']]
    
    for job_id in params["jobID"]:

        logger.info(f"\n\nJOB ID: {job_id}\n\n")
    
        for seed in seeds: 
            
            pyro.set_rng_seed(seed)
            model = models.model
            L = len(params["latent_vars"])
            A = torch.zeros(L, L)
            
            guide = InvSlotAttentionGuide(resolution = params['resolution'],
                                num_iterations = 3,
                                slot_dim = params["slot_dim"],
                                latent_dim = 64,
                                nlatents = L,
                                A = A,
                                stage="eval"
                                ).to(device)
            # GUIDE_PATH = f"{main_dir}/checkpoint-{params['jobID']}/guide_{params['guide_step']}.pth"
            GUIDE_PATH = f"{main_dir}/checkpoint-{job_id}/guide_{params['guide_step']}.pth"
            
            if os.path.isfile(GUIDE_PATH): guide = load_trained_guide(guide, GUIDE_PATH)
            else: raise ValueError(f'{GUIDE_PATH} is not a valid path!')
            
            #logger.info(f'\n\nseed {seed}\n\n')
            logger.info(GUIDE_PATH)

            optimiser = pyro.optim.Adam({'lr': 1e-4})
            csis = mcsis.CSIS(model, guide, optimiser, training_batch_size=256, num_inference_samples=params["num_inference_samples"])

            plots_dir = os.path.abspath("set_prediction_plots")
            if not os.path.isdir(plots_dir): os.mkdir(plots_dir)
            else: 
                shutil.rmtree(plots_dir)
                os.mkdir(plots_dir)
            
            
            for COUNT in range(2, 3):

                count_img_dir = os.path.join(plots_dir, str(COUNT))
                if not os.path.isdir(count_img_dir): os.mkdir(count_img_dir)

                #logger.info(f'\nEVALUATION STARTED FOR SCENES WITH {COUNT} OBJECTS\n')

                ap = {k: 0 for k in threshold}

                if not OOD_EVAL: n_test_samples = len(glob.glob(os.path.abspath(f'images/{COUNT}/*.png')))
                else: n_test_samples = len(glob.glob(os.path.abspath(f'images_ood/{COUNT}/*.png')))
                
                # run the inference module
                if not OOD_EVAL: count_img_path = glob.glob(os.path.abspath(f'images/{COUNT}/*.png'))
                else: count_img_path = glob.glob(os.path.abspath(f'images_ood/{COUNT}/*.png'))

                count_img_path.sort()
                
                resampled_logwts = {k: {} for k in range(len(count_img_path))}

                for img_idx, img_path in enumerate(count_img_path): 
                    sample = img_to_tensor(Image.open(img_path))      
                    sample = sample.to(device)
                    sample_id = img_path.split('/')[-1].split('.')[0]
                    
                    # plt.imshow(sample.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
                    # plt.savefig(f'{count_img_dir}/image_{sample_id}.png')
                    # plt.close()
                    
                    
                    if not OOD_EVAL: target_dict = json.load(open(os.path.abspath(f'metadata/{COUNT}/{sample_id}.json')))
                    else: target_dict = json.load(open(os.path.abspath(f'metadata_ood/{COUNT}/{sample_id}.json')))

                    #logger.info(target_dict)

                    if PRINT_INFERENCE_TIME: since = time.time()
                    
                    posterior = csis.run(observations={"image": sample})
                    prop_traces = posterior.prop_traces[0]
                    traces = posterior.exec_traces[0]
                    log_wts = posterior.log_weights[0]

                    target = process_targets(target_dict)

                    resampling = Empirical(torch.stack([torch.tensor(i) for i in range(len(log_wts))]), torch.stack(log_wts))
                    sample_size = 1

                    assert params["num_inference_samples"] > 0
                    assert sample_size <= params["num_inference_samples"]

                    if sample_size == 1:
                        
                        if params["num_inference_samples"] > 1:
                            resampling_id = resampling().item()
                            logger.info(f"log weights: {[l.item() for l in log_wts]} - resampled trace: {resampling_id}")
                            
                            # logger.info("\n")
                            # for name, site in traces.nodes.items():                    
                            #     # if site["type"] == "sample":
                            #     #     logger.info(f"{name} - {site['value'].shape}")# - {site['value'][resampling_id]}")
                                
                            #     if name == 'image':
                            #         for i in range(site["fn"].mean.shape[0]):
                            #             output_image = site["fn"].mean[i]
                            #             plt.imshow(visualize(output_image[:3].permute(1, 2, 0).cpu().numpy()))
                            #             plt.savefig(os.path.join(plots_dir, f"trace_{img_index}_{i}.png"))
                            #             plt.close()

                            preds = process_preds(prop_traces, resampling_id)
                        
                        else:
                            preds = process_preds_argmax(prop_traces, 0)

                        for t in threshold: 
                            ap[t] += compute_AP(preds, target, t)

                    else:
                        resampling_id = resampling(torch.Size((sample_size,)))
                        logger.info(f"log weights: {[l.item() for l in log_wts]} - resampled traces: {resampling_id}")

                        max_ap_idx = 0
                        best_overall_ap = 0.
                        for p in resampling_id:
                            p = int(p)
                            preds = process_preds(prop_traces, p)
                            aux_ap = {k: 0 for k in threshold}
                            for t in threshold: 
                                aux_ap[t] = compute_AP(preds, target, t)
                            overall_ap = np.mean(list(aux_ap.values()))
                            if overall_ap > best_overall_ap:
                                best_overall_ap = overall_ap
                                max_ap_idx = p
                        preds = process_preds(prop_traces, max_ap_idx)
                        for t in threshold: 
                            ap[t] += compute_AP(preds, target, t)
                

                mAP = {k: v/n_test_samples for k, v in ap.items()}
                # logger.info(f"COUNT {COUNT}: distance thresholds: \n {threshold[0]} - {threshold[1]} - {threshold[2]} - {threshold[3]} - {threshold[4]} - {threshold[5]}")
                # logger.info(f"COUNT {COUNT}: mAP values: {mAP[threshold[0]]} - {mAP[threshold[1]]} - {mAP[threshold[2]]} - {mAP[threshold[3]]} - {mAP[threshold[4]]} - {mAP[threshold[5]]}\n")

                logger.info(f"COUNT {COUNT}: distance thresholds: \n {threshold[0]}")
                logger.info(f"COUNT {COUNT}: mAP values: {mAP[threshold[0]]}\n")
                
                for k in threshold:
                    all_mAP[k].append(mAP[k])
                

        # get mAP values for each threshold, averaged over all scenes
        # logger.info(f"Average mAP: ")
        # for k in threshold:
        #     logger.info(f"{k}: {np.mean(all_mAP[k])} +- {np.std(all_mAP[k])}")
            




if __name__ == '__main__':
    main()