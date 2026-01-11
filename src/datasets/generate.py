import torch
import torch.distributions as dist
import json 
import os

import sys
sys.path.insert(0, '../..') # add parent-parent dir

from src.utils.generate import render, norm
from src.main import params
from src.datasets.data_utils import sample_identifiable_bounded_, sample_ood

import logging
logfile_name = f"generate.log"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(logfile_name, mode='w')
logger.addHandler(fh)

def color_to_rgb(color):
  if color == "red": return (255, 0, 0)
  elif color == "green": return (0, 255, 0)
  elif color == "blue": return (0, 0, 255)
  elif color == "white": return (255, 255, 255)
  elif color == "black": return (0, 0, 0)


def check_overlap(attributes_list):
  object_1, object_2 = attributes_list

  x_cond = abs(object_1[-2]*127 - object_2[-2]*127) <= (object_1[0]/2 + object_2[0]/2) * params["factor_scaling"]["scale"]
  y_cond = abs(object_1[-1]*127 - object_2[-1]*127) <= (object_1[0]/2 + object_2[0]/2) * params["factor_scaling"]["scale"]
  if x_cond and y_cond: return True
  else: return False


def sample_adjacency_matrix(params):
  # order of latents: scale, color, x-pos, y-pos
  # specified A: scale -> color
  
  # A = torch.tensor([
  #   [0., 0., 0., 0.],
  #   [1., 0., 0., 0.],
  #   [0., 0., 0., 0.],
  #   [0., 0., 0., 0.],
  # ])

  A = torch.tensor([
    [0., 0.],
    [1., 0.],
  ])

  return A


def generate_scene(params, scene_id=0, A=None, train_set=True):

  assert A != None
  
  n_objects = params["num_slots"]

  # sample properties for each object 
  shape_obj, size_obj, color_obj, locx_obj, locy_obj = [], [], [], [], []

  scene_dict = {}
  scene_dict["scene_id"] = scene_id
  scene_dict["scene_attr"] = {
    "N": n_objects,
  }

  attributes_list = []
  dependency_type = "linear"
  all_locs = []
  for n in range(int(n_objects)):
    object_attributes = []
    for i, factor in enumerate(params["latent_vars"] + ["x_pos", "y_pos"]):
      # get ranges for the distribution for each factor, in an abstract way
      factor_min, factor_max = params["factor_ranges"][factor]
      if factor in ["x_pos", "y_pos"]:
        sampled_value = dist.Uniform(factor_min, factor_max).sample()
              
      else:
        # check in A if it is a child, and which variable is the parent
        if 1 in A[i, :]:
          parent = (A[i, :] == 1).nonzero(as_tuple=False)
          # specify the new range according to the causal dependency
          if len(parent) == 1:
            norm_parent = norm(object_attributes[parent], params["factor_ranges"][params["latent_vars"][parent]][0], params["factor_ranges"][params["latent_vars"][parent]][1])
            
            if train_set or not params["ood_eval"]:
              sampled_value, _, _ = sample_identifiable_bounded_(norm_parent)
            else:
              sampled_value = sample_ood(norm_parent, params["factor_ranges"][factor])
        else:
          # sample the value
          sampled_value = dist.Uniform(factor_min, factor_max).sample()
            
      object_attributes.append(sampled_value)
    attributes_list.append(object_attributes)
  
  while check_overlap(attributes_list):
    object_attributes = []
    for i, factor in enumerate(params["latent_vars"]):
      # get ranges for the distribution for each factor, in an abstract way
      factor_min, factor_max = params["factor_ranges"][factor]

      # check in A if it is a child, and which variable is the parent (index)
      if 1 in A[i, :]:
        parent = (A[i, :] == 1).nonzero(as_tuple=False)
        # specify the new range according to the causal dependency
        if len(parent) == 1:
          # normalize parent value
          norm_parent = norm(attributes_list[parent], params["factor_ranges"][params["latent_vars"][parent]][0], params["factor_ranges"][params["latent_vars"][parent]][1])
          factor_min, factor_max = norm_parent - 0.2, norm_parent + 0.2

          if factor_min < params["factor_ranges"][factor][0]: factor_min = params["factor_ranges"][factor][0]
          if factor_max > params["factor_ranges"][factor][1]: factor_max = params["factor_ranges"][factor][1]
      # sample the value
      sampled_value = dist.Uniform(factor_min, factor_max).sample()
      object_attributes.append(sampled_value)
    attributes_list[-1] = object_attributes

  for n in range(int(n_objects)):
    scene_dict["scene_attr"]["object_" + str(n)] = {
      "scale":  attributes_list[n][0].item(),
      "color":  attributes_list[n][1].item(),
      "x_pos":  attributes_list[n][2].item(),
      "y_pos":  attributes_list[n][3].item()
      }

  #img = pillowrender.render(shape_obj, size_obj, color_obj, locx_obj, locy_obj, BACKGROUND_COLOR)
  img = render(attributes_list)
  return img, scene_dict


def create_dirs(params, stage):
  if stage == "train":
    subdir = "train"
  elif stage == "eval":
    subdir = "eval"
  else:
    raise ValueError(f"stage {stage} is not valid!") 

  dir = os.path.join(params["dataset_path"], subdir)
  if not os.path.isdir(dir):
    os.mkdir(dir)

  img_dir = os.path.join(dir, "images")
  if not os.path.isdir(img_dir):
    os.mkdir(img_dir)
  
  metadata_dir = os.path.join(dir, "metadata")
  if not os.path.isdir(metadata_dir):
    os.mkdir(metadata_dir)
  
  if params["ood_eval"] and stage == "eval":
    img_dir = os.path.join(dir, "images_OOD")
    if not os.path.isdir(img_dir):
      os.mkdir(img_dir)
    
    metadata_dir = os.path.join(dir, "metadata_OOD")
    if not os.path.isdir(metadata_dir):
      os.mkdir(metadata_dir)

  return img_dir, metadata_dir

def generate_training_dataset(params):
    
  logger.info("starting train scenes generation...")
  
  assert not params["ood_eval"]
  assert params["running_type"] == "train"

  dataset_dir = params["dataset_path"]
  if not os.path.isdir(dataset_dir):
    os.mkdir(dataset_dir)
  
  img_dir, metadata_dir = create_dirs(params, stage="train")

  A = sample_adjacency_matrix(params)
  
  # generate scenes
  for scene_id in range(params["train_dataset_len"]): 
    img, scene_dict = generate_scene(params, scene_id, A)
    # save scenes
    with open(f"{metadata_dir}/{str(scene_id).zfill(len(str(params['train_dataset_len'])))}.json", "w") as outfile:
      json.dump(scene_dict, outfile)  
    img.save(f"{img_dir}/{str(scene_id).zfill(len(str(params['train_dataset_len'])))}.png")
  
  logger.info(f"(train set): {scene_id} scenes stored...")
  
  return A

def generate_test_dataset(params, A=None):
  
  assert A != None
  assert params["running_type"] == "eval"

  logger.info("starting test scenes generation...")

  img_dir, metadata_dir = create_dirs(params, stage="eval")
    
  # generate scenes
  for scene_id in range(params["test_dataset_len"]): 
    img, scene_dict = generate_scene(params, scene_id, A, train_set=False)
    # save scenes
    with open(f"{metadata_dir}/{str(scene_id).zfill(len(str(params['test_dataset_len'])))}.json", "w") as outfile:
      json.dump(scene_dict, outfile)  
    img.save(f"{img_dir}/{str(scene_id).zfill(len(str(params['test_dataset_len'])))}.png")
  logger.info(f"(test set): {scene_id} scenes stored...")

if __name__ == "__main__":

  if params["running_type"] == "train":
    A = generate_training_dataset(params)
  elif params["running_type"] == "eval":
    A = sample_adjacency_matrix(params)
    generate_test_dataset(params, A)
  else:
    raise ValueError(f"running type {params['running_type']} invalid!")
