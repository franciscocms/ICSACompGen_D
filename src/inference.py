import torch
import torch.nn as nn
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
import sklearn
from sklearn.decomposition import PCA

# add project path to sys to import relative modules
import sys
sys.path.append(os.path.abspath(__file__+'/../../'))

main_dir = os.path.abspath(__file__+'/../../')

from utils.guide import LinearProbe
from models.guide import InvSlotAttentionGuide
from main import params

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
        target = torch.zeros(params['num_slots'], len(shape_vals) + len(size_vals) + len(color_vals) + 2) 
    else:
        target = torch.zeros(params['num_slots'], len(shape_vals) + len(color_vals) + 2) 

    for i in range(n):
        s = len(color_vals)
        target[i, :s] = F.one_hot(torch.tensor(color_vals[target_dict['scene_attr'][f'object_{i}']['color']]), len(color_vals))
        
        if "size" in params["latent_vars"]:
            target[i, s:s+len(size_vals)] = F.one_hot(torch.tensor(size_vals[target_dict['scene_attr'][f'object_{i}']['size']]), len(size_vals))
            s = s+len(size_vals)
        
        target[i, s:s+len(shape_vals)] = F.one_hot(torch.tensor(shape_vals[target_dict['scene_attr'][f'object_{i}']['shape']]), len(shape_vals))
        s = s+len(shape_vals)
        target[i, s] = target_dict['scene_attr'][f'object_{i}']['initLocX']
        target[i, s+1] = target_dict['scene_attr'][f'object_{i}']['initLocY']
    return target
    
class MyDataset(Dataset):
    def __init__(self, dataset_path, ood):
        self.dataset_path = dataset_path
        self.img_path = os.path.join(self.dataset_path, "images")
        
        if ood: self.target_path = os.path.join(self.dataset_path, "metadata_OOD")
        else: self.target_path = os.path.join(self.dataset_path, "metadata")
        self.nscenes = len(glob.glob(os.path.join(self.img_path, "*.png")))

        self.ood = ood
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        img = self.transform(Image.open(os.path.join(self.img_path, f"{str(idx).zfill(len(str(self.nscenes)))}.png")))        
        target_dict = json.load(open(os.path.join(self.target_path, f"{str(idx).zfill(len(str(self.nscenes)))}.json")))        
        target = process_targets(target_dict)
        return img, torch.cat((torch.argmax(target[:, :3], dim=-1).unsqueeze(-1),
                               torch.argmax(target[:, 3:6], dim=-1).unsqueeze(-1)), dim=-1)

class ProbeDataset(Dataset):
    def __init__(self, noise, yc, target, var_idx, split):
        self.noise = noise
        self.yc = yc
        self.target = target
        self.var_idx = var_idx

        logger.info(f"# of samples for {split} split: {self.noise.size(0)}")

    def __len__(self):
        return len(self.noise)
    
    def __getitem__(self, idx):
        #target = torch.argmax(self.target[idx, :, self.var_idx*3 : (self.var_idx + 1)*3], dim=-1)
        return self.noise[idx], self.yc[idx], self.target[idx, :, self.var_idx]

def hungarian_r2_score(pred, target):
    """
    pred:   [B, N] int predictions
    target: [B, N] int targets

    Returns: scalar R² averaged over batch.
    """

    B, N = pred.shape

    # ---- 1. Batched cost matrix: cost[b, i, j] = (pred[b,i] - target[b,j])^2 ----
    diff = pred.unsqueeze(2) - target.unsqueeze(1)      # [B, N, N]
    cost = (diff * diff).cpu()                          # convert once for scipy

    r2_scores = []

    # ---- 2. Hungarian algorithm per batch (cannot be parallelized) ----
    for b in range(B):
        row_ind, col_ind = linear_sum_assignment(cost[b].numpy())

        # ---- 3. Apply matching ----
        matched_pred   = pred[b, row_ind].cpu().numpy()
        matched_target = target[b, col_ind].cpu().numpy()

        # ---- 4. Compute R² using sklearn ----
        # sklearn handles edge cases (constant targets, etc.)
        r2_scores.append(sklearn.metrics.r2_score(matched_target, matched_pred))

    # ---- 5. Average over batch ----
    return torch.tensor(r2_scores) #/ B

def hungarian_ce_loss(pred_logits, target_labels):
    """
    pred_logits: [B, N, C]
    target_labels: [B, N]

    Returns scalar CE loss using Hungarian matching per batch element.
    """
    B, N, C = pred_logits.shape

    # 1. Compute log-probabilities (batched)
    log_probs = pred_logits.log_softmax(dim=-1)  # [B, N, C]

    # 2. Build cost matrices for the whole batch: cost[b,i,j] = -log_probs[b,i,target[b,j]]
    # target_labels: [B, N] -> expand to [B, N, N]
    cost = -log_probs.gather(
        dim=2,
        index=target_labels.unsqueeze(1).expand(-1, N, -1)
    )  # [B, N, N]

    # 3. Run Hungarian per batch element (cannot batch this step)
    assignments = []
    for b in range(B):
        row_ind, col_ind = linear_sum_assignment(cost[b].detach().cpu().numpy())
        assignments.append((row_ind, col_ind))

    # 4. Rearrange predictions and targets using batch matches (batched application)
    matched_logits = []
    matched_targets = []

    for b, (rows, cols) in enumerate(assignments):
        matched_logits.append(pred_logits[b, rows])    # [N, C]
        matched_targets.append(target_labels[b, cols]) # [N]

    matched_logits = torch.stack(matched_logits, dim=0)  # [B, N, C]
    matched_targets = torch.stack(matched_targets, dim=0) # [B, N]

    # 5. Compute CE over flattened batch
    loss = F.cross_entropy(
        matched_logits.reshape(B*N, C),
        matched_targets.reshape(B*N)
    )

    return loss


def main():    

    assert params["running_type"] == "eval"
    assert not params["infer_by_prediction"] 

    logger.info(device)

    threshold = [-1.]
    all_mAP = {k: [] for k in threshold}

    # if not isinstance(params["jobID"], list):
    #     params["jobID"] = [params['jobID']]

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
    
    all_ap = []

    for COUNT in range(2, 3):

        for OOD_EVAL in [False, True]:

            logger.info(f"\nOOD: {OOD_EVAL}\n")
        
            count_img_dir = os.path.join(plots_dir, str(COUNT))
            if not os.path.isdir(count_img_dir): os.mkdir(count_img_dir)

            dataset_dir = "/nas-ctm01/datasets/public/CG/test"

            dataset = MyDataset(dataset_dir, OOD_EVAL)
            dataloader = DataLoader(dataset, 64, shuffle=False, generator=torch.Generator(device='cuda'))

            z_noise = torch.tensor([])  
            targets = torch.tensor([])   
            ycs = torch.tensor([])       
            l_to_label = {i: l for i, l in enumerate(params["latent_vars"])}
            img_saved = False
            with torch.no_grad():
                for img, target in dataloader:
                    img = img.to(device)
                    target = target.to(device)
                    
                    batch_noise = guide(img, target) # [B, N, L, d]
                    z_noise = torch.cat((z_noise, batch_noise), dim=0)

                    B, N, _, d = batch_noise.shape

                    if not img_saved:
                        aux_attn = guide.attn.reshape((B, N, 32, 32))
                        fig, ax = plt.subplots(ncols=N)
                        for j in range(N):                                       
                                im = ax[j].imshow(aux_attn[0, j, :, :].detach().cpu().numpy())
                                ax[j].grid(False)
                                ax[j].axis('off')        
                        if OOD_EVAL: plt.savefig("attn_OOD.png")
                        else: plt.savefig("attn_ID.png")
                        plt.close()

                        plot_img = np.transpose(img[0].detach().cpu().numpy(), (1, 2, 0))
                        plt.imshow(np.clip(plot_img, 0., 1.))
                        plt.axis('off')
                        if OOD_EVAL: plt.savefig("img_OOD.png")
                        else: plt.savefig("img_ID.png")
                        plt.close()
                        img_saved = True
                    
                    batch_ycs = torch.tensor([])
                    for i in range(L):
                        ycs_ = guide.z_nets[l_to_label[i]].compute_yc(batch_noise.reshape(B*N, L, d)[:, i, :])
                        ycs_ = ycs_.reshape(B, N, ycs_.size(-1))
                        batch_ycs = torch.cat((batch_ycs, ycs_.unsqueeze(-2)), dim=-2)
                    
                    ycs = torch.cat((ycs, batch_ycs), dim=0)
                    targets = torch.cat((targets, target), dim=0)
            
            logger.info(z_noise.shape)
            logger.info(ycs.shape)
            logger.info(targets.shape)
            
            var_idx = {l: i for i, l in enumerate(params["latent_vars"])}
            loss_fn = nn.CrossEntropyLoss()
            train_val_test_split = [0.7, 0.2, 0.1]

            for i in range(L):

                #if i == 0: j = 1
                #elif i == 1: j = 0
                j = i
                
                logger.info(f"\nprobe for variable {params['latent_vars'][i]}\n")
                
                model = LinearProbe(z_noise.size(-1), 3).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                n_train_samples = int(train_val_test_split[0]*z_noise.size(0))
                n_val_samples = n_train_samples + int(train_val_test_split[1]*z_noise.size(0))
                
                probe_ds_train = ProbeDataset(z_noise[:n_train_samples, :, i], ycs[:n_train_samples, :, i], targets[:n_train_samples], j, split="train")
                probe_ds_val = ProbeDataset(z_noise[n_train_samples:n_val_samples, :, i], ycs[n_train_samples:n_val_samples, :, i], targets[n_train_samples:n_val_samples], j, split="validation")
                probe_ds_test = ProbeDataset(z_noise[n_val_samples:, :, i], ycs[n_val_samples:, :, i], targets[n_val_samples:], j, split="test")

                probe_dl_train = DataLoader(probe_ds_train, 256, shuffle=True, generator=torch.Generator(device='cuda'))
                probe_dl_val = DataLoader(probe_ds_val, len(probe_ds_val), shuffle=False, generator=torch.Generator(device='cuda'))
                probe_dl_test = DataLoader(probe_ds_test, len(probe_ds_test), shuffle=False, generator=torch.Generator(device='cuda'))

                training_epochs = 21
                best_loss = 1.
                for epoch in range(training_epochs):
                    epoch_train_loss = 0.
                    epoch_val_loss = 0.
                    
                    model.train()
                    for z, yc, t in probe_dl_train:
                        z = z.to(device)
                        t = t.to(device)

                        optimizer.zero_grad()
                        
                        preds = model(z) # [B, N, 3]
                        
                        # loss must be computed under a matching algorithm
                        loss = hungarian_ce_loss(preds, t)

                        loss.backward()
                        optimizer.step()
                        epoch_train_loss += loss/z.size(0)
                    
                    model.eval()
                    with torch.no_grad():
                        for z, yc, t in probe_dl_val:
                            z = z.to(device)
                            t = t.to(device)                        
                            
                            loss = hungarian_ce_loss(model(z), t)
                            epoch_val_loss += loss/z.size(0)
                    
                    if epoch_val_loss < best_loss:
                        #logger.info(f"saving new model with val loss: {epoch_val_loss}")
                        best_loss = epoch_val_loss
                        torch.save(model.state_dict(), 'pred_model.pth')
                    
                    if epoch % 10 == 0:
                        logger.info(f"epoch {epoch}/{training_epochs-1}")
                        logger.info(f"... train loss: {epoch_train_loss}")
                        logger.info(f"... val loss: {epoch_val_loss}")

                # testing
                model = LinearProbe(z_noise.size(-1), 3).to(device) 
                model.load_state_dict(torch.load('pred_model.pth')) 
                model.eval()
                with torch.no_grad():
                    for z, yc, t in probe_dl_test:
                        B, N, d = z.shape

                        z = z.to(device)
                        yc = yc.to(device)
                        t = t.to(device)
                        preds = torch.argmax(model(z), dim=-1)
                        #preds_yc = torch.argmax(yc, dim=-1)                     

                        logger.info(f"preds: {preds[:5]}")
                        #logger.info(f"preds yc: {preds_yc[:5]}")
                        logger.info(f"target: {t[:5]}")

                        r2_score = hungarian_r2_score(preds, t)
                        #r2_score_yc = hungarian_r2_score(preds_yc, t)

                        logger.info(f"r2: {r2_score[:5]}")
                        #logger.info(f"r2_yc: {r2_score_yc[:5]}")

                        
                        pca = PCA(n_components=2)
                        z_d = pca.fit_transform(z.cpu().reshape(B*N, d))
                        plt.figure(figsize=(8, 6))
                        plt.scatter(z_d[:, 0], z_d[:, 1], c=preds.cpu().numpy(), alpha=0.5)
                        if OOD_EVAL: plt.savefig(f"z_{i+1}_OOD_preds.png")
                        else: plt.savefig(f"z_{i+1}_ID_preds.png")
                        plt.close()



                # for i in range(len(r2_score)):
                #     if r2_score[i] < 1.:
                #         logger.info(f"pred: {preds[i]} - target: {t[i]} - r2: {r2_score[i]}")

                
                logger.info(f"test r2 score: {torch.mean(r2_score)}")
                #logger.info(f"test r2 score yc: {torch.mean(r2_score_yc)}")

                    
                

                    



                    

            

if __name__ == '__main__':
    main()