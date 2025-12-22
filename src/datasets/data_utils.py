import os
import glob
import json
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset

import sys
sys.path.insert(0, '../..') # add parent-parent dir

from src.main import params
import logging
logger = logging.getLogger(params["running_type"])

def process_targets(target_dict):   
    n = int(target_dict['scene_attr']['N'])
    target = []
    for i in range(n):
      out = torch.cat([torch.FloatTensor([target_dict['scene_attr'][f'object_{i}']['scale']]),
                       torch.FloatTensor([target_dict['scene_attr'][f'object_{i}']['color']]),
                       torch.FloatTensor([target_dict['scene_attr'][f'object_{i}']['x_pos']]),
                       torch.FloatTensor([target_dict['scene_attr'][f'object_{i}']['y_pos']])], dim=-1)
      target.append(out)
    target = torch.stack(target)
    if len(target.shape) > 2: target = target.squeeze()
    return target

class MyDataset(Dataset):
  def __init__(self, dataset_path, ood=False):
      self.dataset_path = dataset_path
      if not ood:
        self.img_path = os.path.join(self.dataset_path, "images")
        self.target_path = os.path.join(self.dataset_path, "metadata")
      else:
        self.img_path = os.path.join(self.dataset_path, "images_OOD")
        self.target_path = os.path.join(self.dataset_path, "metadata_OOD")
         
      self.nscenes = len(glob.glob(os.path.join(self.img_path, "*.png")))
      self.transform = transforms.Compose([transforms.ToTensor()])

  def __len__(self):
      return self.nscenes
  
  def __getitem__(self, idx):
    img = self.transform(Image.open(os.path.join(self.img_path, f"{str(idx).zfill(len(str(self.nscenes)))}.png")))
    target_dict = json.load(open(os.path.join(self.target_path, f"{str(idx).zfill(len(str(self.nscenes)))}.json")))
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