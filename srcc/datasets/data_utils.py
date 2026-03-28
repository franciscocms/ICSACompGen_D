import os
import glob
import json
from PIL import Image

import torch
import torch.distributions as dist
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np


import sys
sys.path.insert(0, '../..') # add parent-parent dir

from src.main import params
from src.utils.generate import norm
import logging
logger = logging.getLogger(params["running_type"])

def sample_identifiable_bounded_(parent):
  """
  Samples z_hue in (0, 1) using a Logit-Normal approach.
  Satisfies Condition 4 (heteroscedasticity) and Condition 3 (diffeomorphism).
  """
  # 1. Define Mean and Sigma in 'Logit Space'
  # We use values that, when passed through sigmoid, center around 0.5
  # mu_y will range roughly from -2.0 to 2.0
  mu_y = 4.0 * (parent - 0.5) 
  
  # Heteroscedasticity: Variance changes with scale
  # We keep sigma_y large enough to move lambda_2, 
  # but small enough that we don't always hit the sigmoid's flat ends
  sigma_y = 0.2 + 0.5 * torch.abs(torch.sin(np.pi * parent))
  
  # 2. Sample in unbounded space (Pre-latent y)
  epsilon = torch.randn_like(parent)
  y = mu_y + sigma_y * epsilon
  
  # 3. Diffeomorphic Squashing to (0, 1)
  child = torch.sigmoid(y)
  
  return child, mu_y, sigma_y**2

def sample_ood(parent, child_boundaries, threshold=4.):
  """
  Returns only combinations (scale, hue) that were NOT seen in training.
  """
  is_ood = False
  while not is_ood:
    uniform_child = dist.Uniform(child_boundaries[0], child_boundaries[1]).sample()
    clamped_child = torch.clamp(uniform_child, 1e-6, 1-1e-6)
    y_proposal = torch.log(clamped_child / (1 - clamped_child))
    
    # 3. Check against the training mechanism
    _, mu_y, sigma_y = sample_identifiable_bounded_(parent)
    
    # Distance in standard deviations
    z_score = torch.abs(y_proposal - mu_y) / sigma_y
    
    # 4. Keep only those outside the threshold (the 'Gap')
    is_ood = z_score > threshold
  
  return uniform_child

def normalize(x, min, max):
   return (x-min)/(max-min)

def sample_id_linear(parent, factor_ranges, window=0.2):
  factor_min, factor_max = parent - window, parent + window
  if factor_min < factor_ranges[0]: factor_min = factor_ranges[0]
  if factor_max > factor_ranges[1]: factor_max = factor_ranges[1]
  sample = dist.Uniform(factor_min, factor_max).sample()
  return sample
   
def sample_ood_linear(parent, factor_ranges, window=0.2):
  factor_min = parent - window
  factor_max = parent + window
  
  # Identify valid ranges outside the window
  # Range 1: [0, factor_min]
  # Range 2: [factor_max, 1]
  valid_ranges = []
  if factor_min > factor_ranges[0]:
      valid_ranges.append((factor_ranges[0], factor_min))
  if factor_max < factor_ranges[1]:
      valid_ranges.append((factor_max, factor_ranges[1]))

  # 1. Pick which range to sample from (weighted by the width of the range)
  widths = torch.tensor([r[1] - r[0] for r in valid_ranges])
  range_idx = dist.Categorical(probs=widths).sample()
  chosen_range = valid_ranges[range_idx]
  
  # 2. Sample color uniformly within the chosen OOD range
  sample = dist.Uniform(chosen_range[0], chosen_range[1]).sample()
  return sample 
   

def process_targets(target_dict):   
    n = int(target_dict['scene_attr']['N'])
    target = []
    for i in range(n):
      all_v = []
      for var in params["latent_vars"]:
        v = normalize(torch.FloatTensor([target_dict['scene_attr'][f'object_{i}'][var]]),
                          params["factor_ranges"][var][0],
                          params["factor_ranges"][var][1])    
        all_v.append(v)
      out = torch.cat(all_v, dim=-1)
      target.append(out)
    target = torch.stack(target)
    if len(target.shape) > 2: target = target.squeeze()    
    return target

class MyDataset(Dataset):
  def __init__(self, dataset_path, ood=False, inference=False):
      self.dataset_path = dataset_path
      if not ood:
        self.img_path = os.path.join(self.dataset_path, "images")
        self.target_path = os.path.join(self.dataset_path, "metadata")
      else:
        self.img_path = os.path.join(self.dataset_path, "images_OOD")
        self.target_path = os.path.join(self.dataset_path, "metadata_OOD")
         
      self.nscenes = len(glob.glob(os.path.join(self.img_path, "*.png")))
      self.transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.GaussianBlur(3, (0.5, 0.5))])

      logger.info(f"\ndataset length: {self.nscenes}\n")

      if inference:
         self.img_path

  def __len__(self):
    return self.nscenes
  
  def __getitem__(self, idx):
    img_path = os.path.join(self.img_path, f"{str(idx).zfill(len(str(self.nscenes)))}.png")
    img = self.transform(Image.open(img_path))
    
    target_path = os.path.join(self.target_path, f"{str(idx).zfill(len(str(self.nscenes)))}.json")
    target_dict = json.load(open(target_path))
    target = process_targets(target_dict)
    return img, target

class ProbeDataset(Dataset):
    def __init__(self, noise, target, var_idx, split):
        self.noise = noise
        self.target = target
        self.var_idx = var_idx

        logger.info(f"# of samples for {split} split: {self.noise.size(0)}")

    def __len__(self):
        return len(self.noise)
    
    def __getitem__(self, idx):
        return self.noise[idx], self.target[idx, :, self.var_idx]