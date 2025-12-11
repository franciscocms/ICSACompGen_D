import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import json
import shutil
import time
import pickle
from scipy.optimize import linear_sum_assignment

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
    
class MyDataset(Dataset):
    def __init__(self, images_path, count, ood):
        self.img_path = images_path
        self.count = count
        self.ood = ood
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        sample_id = self.img_path[idx].split('/')[-1].split('.')[0]

        img = self.transform(Image.open(self.img_path[idx]))
        
        if params["n_edges"] == 1:
            if len(params["latent_vars"]) == 3:
                if self.ood: target_dict = json.load(open(os.path.abspath(f'metadata_ood/{self.count}/{sample_id}.json')))
                else: target_dict = json.load(open(os.path.abspath(f'metadata/{self.count}/{sample_id}.json')))
            elif len(params["latent_vars"]) == 2:
                if self.ood: target_dict = json.load(open(os.path.abspath(f'metadata_ood_2vars/{self.count}/{sample_id}.json')))
                else: target_dict = json.load(open(os.path.abspath(f'metadata_2vars/{self.count}/{sample_id}.json')))
        if params["n_edges"] == 2:
            if self.ood: target_dict = json.load(open(os.path.abspath(f'metadata_ood_2edges/{self.count}/{sample_id}.json')))
            else: target_dict = json.load(open(os.path.abspath(f'metadata_2edges/{self.count}/{sample_id}.json')))
        target = process_targets(target_dict)
        return img, target
        
def main():    

    assert params["running_type"] == "eval"

    logger.info(device)

    threshold = [-1.]
    all_mAP = {k: [] for k in threshold}

    # if not isinstance(params["jobID"], list):
    #     params["jobID"] = [params['jobID']]

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
    GUIDE_PATH = f"{main_dir}/checkpoint-{params['jobID']}/guide_{params['guide_step']}.pth"
    
    if os.path.isfile(GUIDE_PATH): guide = load_trained_guide(guide, GUIDE_PATH, stage="eval")
    else: raise ValueError(f'{GUIDE_PATH} is not a valid path!')
    
    #logger.info(f'\n\nseed {seed}\n\n')
    logger.info(GUIDE_PATH)

    plots_dir = os.path.abspath("set_prediction_plots")
    if not os.path.isdir(plots_dir): os.mkdir(plots_dir)
    else: 
        shutil.rmtree(plots_dir)
        os.mkdir(plots_dir)
    
    all_ap = []

    for COUNT in range(2, 3):

        for OOD_EVAL in [False, True]:

            logger.info(f"\nOOD: {OOD_EVAL}\n")
        
            count_img_dir = os.path.join(plots_dir, str(COUNT))
            if not os.path.isdir(count_img_dir): os.mkdir(count_img_dir)

            #logger.info(f'\nEVALUATION STARTED FOR SCENES WITH {COUNT} OBJECTS\n')

            ap = {k: 0 for k in threshold}
            ap_list = []

            if params["n_edges"] == 1:
                if len(params["latent_vars"]) == 3:
                    if not OOD_EVAL: img_dir = glob.glob(os.path.abspath(f'images/{COUNT}/*.png'))
                    else: img_dir = glob.glob(os.path.abspath(f'images_ood/{COUNT}/*.png'))
                elif len(params["latent_vars"]) == 2:
                    
                    if not OOD_EVAL: img_dir = glob.glob(os.path.abspath(f'images_2vars/{COUNT}/*.png'))
                    else: img_dir = glob.glob(os.path.abspath(f'images_ood_2vars/{COUNT}/*.png'))
            
            elif params["n_edges"] == 2:
                if not OOD_EVAL: img_dir = glob.glob(os.path.abspath(f'images_2edges/{COUNT}/*.png'))
                else: img_dir = glob.glob(os.path.abspath(f'images_ood_2edges/{COUNT}/*.png'))

            n_test_samples = len(img_dir)
            
            logger.info(img_dir[0])

            dataset = MyDataset(img_dir, COUNT, OOD_EVAL)
            dataloader = DataLoader(dataset, 64, shuffle=False, generator=torch.Generator(device='cuda'))

            z = torch.tensor([])
            color_props = torch.tensor([])
            shape_props = torch.tensor([])
            size_props = torch.tensor([])
            x_props = torch.tensor([])
            y_props = torch.tensor([])

            color_target = torch.tensor([])
            shape_target = torch.tensor([])
            size_target = torch.tensor([])
            x_target = torch.tensor([])
            y_target = torch.tensor([])

            ap_den = 0
            
            with torch.no_grad():
                for img, target in dataloader:

                    preds = guide(observations={"image": img})

                    # shape_target = torch.cat((shape_target, torch.argmax(target[:, :3], dim=-1).unsqueeze(0)), dim=0)
                    # size_target = torch.cat((size_target, torch.argmax(target[:, 3:6], dim=-1).unsqueeze(0)), dim=0)
                    # color_target = torch.cat((color_target, torch.argmax(target[:, 6:9], dim=-1).unsqueeze(0)), dim=0)
                    # x_target = torch.cat((x_target, target[:, -3].unsqueeze(0)), dim=0)
                    # y_target = torch.cat((y_target, target[:, -2].unsqueeze(0)), dim=0)
                    

                    # z = torch.cat((z, guide.z), dim=0)
                    # color_props = torch.cat((color_props, guide.color_prop), dim=0)
                    # shape_props = torch.cat((shape_props, guide.shape_prop), dim=0)
                    # size_props = torch.cat((size_props, guide.size_prop), dim=0)
                    # x_props = torch.cat((x_props, guide.slot_pos[..., 0]), dim=0)
                    # y_props = torch.cat((y_props, guide.slot_pos[..., 1]), dim=0)
                    
                    # preds = process_preds_argmax(prop_traces, 0)

                    for b in range(preds.size(0)):
                        for t in threshold: 
                            ap_value = compute_AP(preds[b], target[b], t)  
                            ap[t] += ap_value
                            ap_list.append(ap_value)
                            ap_den += 1               

            # z = torch.flatten(z, 0, 1).cpu().numpy()

            # aux_attn = guide.attn               
            # B, N, _ = color_props.shape
            
            # props=[color_props.reshape(B*N, -1), shape_props.reshape(B*N, -1), size_props.reshape(B*N, -1)]
            # gts = [color_target, shape_target, size_target]

            # logger.info(color_target.shape)
            # logger.info(color_target)
            # logger.info(gts[0])

            # logger.info(x_props.shape)
            # logger.info(y_props.shape)
            # logger.info(x_target.shape)
            # logger.info(y_target.shape)

            # pos_dist = torch.cdist(torch.cat((x_props.unsqueeze(-1), y_props.unsqueeze(-1)), dim=-1), 
            #                        torch.cat((x_target.unsqueeze(-1), y_target.unsqueeze(-1)), dim=-1))
            # indices = np.array([linear_sum_assignment(p.cpu().numpy()) for p in pos_dist])[:, -1, :]

            # titles = ["color", "shape", "size"]
            # classes = {0: ["red", "green", "blue"],
            #            1: ["ball", "square", "triangle"],
            #            2: ["small", "medium", "large"]}
            
            # from sklearn.manifold import TSNE
            # for i in range(L):
            #     tsne = TSNE(n_components=2, random_state=0)
            #     z_t = tsne.fit_transform(z[:, i, :])
            #     # Plot with class indices as color

            #     preds = torch.argmax(props[i], dim=-1).cpu().numpy()
            #     gt = torch.gather(gts[i], 1, torch.from_numpy(indices).to(device)).reshape(B*N).cpu().numpy()

            #     plt.figure(figsize=(8, 6))
            #     scatter = plt.scatter(z_t[:, 0], z_t[:, 1], c=preds, alpha=0.5)
            #     plt.legend(handles=scatter.legend_elements()[0], labels=classes[i], title=titles[i])
            #     plt.savefig(f"z_{i+1}_preds.png")
            #     plt.close()

            #     plt.figure(figsize=(8, 6))
            #     scatter = plt.scatter(z_t[:, 0], z_t[:, 1], c=gt, alpha=0.5)
            #     plt.legend(handles=scatter.legend_elements()[0], labels=classes[i], title=titles[i])
            #     plt.savefig(f"z_{i+1}_gt.png")
            #     plt.close()
            
            # B, N, _ = aux_attn.shape
            # aux_attn = aux_attn.reshape((B, N, 32, 32))
            # fig, ax = plt.subplots(ncols=N)
            # for j in range(N):                                       
            #     im = ax[j].imshow(aux_attn[0, j, :, :].detach().cpu().numpy())
            #     ax[j].grid(False)
            #     ax[j].axis('off')        
            # plt.savefig(f"attn.png")
            # plt.close()

            # plot_img = np.transpose(sample.squeeze(0).cpu().numpy(), (1, 2, 0))
            # plt.imshow(np.clip(plot_img, 0., 1.))
            # plt.axis('off')
            # plt.savefig(f"img.png")
            # plt.close()
            
            
            
            mAP = {k: v/ap_den for k, v in ap.items()}
            # logger.info(f"COUNT {COUNT}: distance thresholds: \n {threshold[0]} - {threshold[1]} - {threshold[2]} - {threshold[3]} - {threshold[4]} - {threshold[5]}")
            # logger.info(f"COUNT {COUNT}: mAP values: {mAP[threshold[0]]} - {mAP[threshold[1]]} - {mAP[threshold[2]]} - {mAP[threshold[3]]} - {mAP[threshold[4]]} - {mAP[threshold[5]]}\n")

            all_ap.append(ap_list)
            
            logger.info(f"\nCOUNT {COUNT}: distance thresholds: \n {threshold[0]}")
            logger.info(f"COUNT {COUNT}: mAP values: {mAP[threshold[0]]}\n")

            
        with open(f"{params['jobID']}_AP.pkl", "wb") as f:  # open a text file
            pickle.dump(np.asarray(all_ap), f) # serialize the list

        # get mAP values for each threshold, averaged over all scenes
        # logger.info(f"Average mAP: ")
        # for k in threshold:
        #     logger.info(f"{k}: {np.mean(all_mAP[k])} +- {np.std(all_mAP[k])}")
            

if __name__ == '__main__':
    main()