# the PP model

import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
from torchvision import transforms

from utils.distributions import CategoricalVals, MyBernoulli, MyNormal, MyPoisson
from utils.generate import overflow, overlap, render
from utils.guide import to_int
from .setup import params

import multiprocessing as mp

import warnings
warnings.filterwarnings("ignore")

import logging
import time
import random

logger = logging.getLogger(params["running_type"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_transform = transforms.Compose([transforms.ToTensor()])

shape_vals = params["shape_vals"]
size_vals = params["size_vals"]
color_vals = params["color_vals"]
size_mapping_int = {'small': 10, 'medium': 15, 'large': 20}


B = params['batch_size'] if params["running_type"] == "train" else params['num_inference_samples']
M = params['max_objects']

# if params["jobID"] in [288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314]:
#   probs_0 = random.uniform(0.2, 0.8)
#   probs_1 = random.uniform(0.2, 0.8)
#   probs_2 = random.uniform(0.2, 0.8)
#   color_to_shape_probs = {
#       0: torch.tensor([0.,      probs_0,   1-probs_0]),
#       1: torch.tensor([probs_1, 0.,        1-probs_1]),
#       2: torch.tensor([probs_2, 1-probs_2, 0.])
#       }
#   #logger.info(f"\ncolor_to_shape_probs: {color_to_shape_probs}\n")
#   params["color_to_shape_probs"] = color_to_shape_probs

# if params["jobID"] >= 315:
#   probs_0 = random.uniform(0.8, 0.9)
#   probs_1 = random.uniform(0.8, 0.9)
#   probs_2 = random.uniform(0.8, 0.9)
#   color_to_shape_probs = {
#       0: torch.tensor([0.,      probs_0,   1-probs_0]),
#       1: torch.tensor([probs_1, 0.,        1-probs_1]),
#       2: torch.tensor([probs_2, 1-probs_2, 0.])
#       }
#   #logger.info(f"\ncolor_to_shape_probs: {color_to_shape_probs}\n")
#   params["color_to_shape_probs"] = color_to_shape_probs


def check_occlusion(check_dict, locx, locy, sizes):

  """
  check_dict: dictionary with locations and sizes of all sampled objects so far
  locx: x-location for the proposed object
  locy: y-location for the proposed object
  sizes: size for the proposed object
  """

  flag = False
  w, h = (128, 128)
  eps = 10
  
  for o, _ in check_dict.items():
    x_cond = abs(locx*w - check_dict[o]['locX']*w) <= (size_mapping_int[sizes]//2 + size_mapping_int[check_dict[o]['size']]//2 + eps)
    y_cond = abs(locy*h - check_dict[o]['locY']*h) <= (size_mapping_int[sizes]//2 + size_mapping_int[check_dict[o]['size']]//2 + eps)

    if x_cond and y_cond: flag = True
  
  return flag

def sample_loc(i):
    with pyro.poutine.block(): 
        locX_mu = pyro.sample(f"locX_{i}", dist.Uniform(0.15, 0.85))
        locY_mu = pyro.sample(f"locY_{i}", dist.Uniform(0.15, 0.85))
    return locX_mu, locY_mu

def sample_size(i, b=0, n=0, root=True, cause=None, conditionals=None):  
  
  if root:
    if len(size_vals) > 1:
      size_probs = torch.tensor([1/len(size_vals) for _ in range(len(size_vals))], device=device)
      with pyro.poutine.block(): 
        size = pyro.sample(f"size_{i}", dist.Categorical(probs=size_probs))
    else:
      size = torch.tensor(0)
    
  else:    
    assert cause != None
    assert conditionals != None 
    size_probs = conditionals[cause[b, n].item()]
    with pyro.poutine.block(): 
      size = pyro.sample(f"size_{i}", dist.Categorical(probs=size_probs))
       
  return size

def sample_scenes(training):

  color_to_shape_probs = {
     0: torch.tensor([0., 1/2, 1/2]),
     1: torch.tensor([1/2, 0., 1/2]),
     2: torch.tensor([1/2, 1/2, 0.])
     }  
  
  if params["n_edges"] == 2:
    if params["dag_type"] == "root":
      color_to_size_probs = {
        0: torch.tensor([0., 1/2, 1/2]),
        1: torch.tensor([1/2, 0., 1/2]),
        2: torch.tensor([1/2, 1/2, 0.])
        } 

  objects_mask = pyro.sample(f"mask", dist.Bernoulli(1.).expand([B, M])).to(torch.bool)

  with pyro.poutine.mask(mask=objects_mask): 
    
    if training:
      color = pyro.sample(f"color", dist.Categorical(probs=torch.tensor([1/len(color_vals) for _ in range(len(color_vals))])).expand([B, M]))     
      shape_probs = torch.stack([color_to_shape_probs[idx.item()] for idx in color.flatten()]).view(B, M, -1)
      shape = pyro.sample(f"shape", dist.Categorical(probs=shape_probs))

      # check if there is a 1 (a parent) in the final row (size)
      if "size" in params["latent_vars"] and 1 in params["adj_matrix"][-1, :]: 
        size_root = False
        cause = color
        conditionals = color_to_size_probs

      else: 
        size_root = True
        cause = None
        conditionals = None
    
    else:
      color = pyro.sample(f"color", dist.Categorical(probs=torch.tensor([1/len(color_vals) for _ in range(len(color_vals))])).expand([B, M]))     
      
      if params["n_edges"] == 2:  
        if torch.distributions.Bernoulli(0.5).sample():
          color_to_shape_probs_validation = {
            0: torch.tensor([1., 0., 0.]),
            1: torch.tensor([0., 1., 0.]),
            2: torch.tensor([0., 0., 1.])
            }
          color_to_size_probs_validation = {
            0: torch.ones(3)/3,
            1: torch.ones(3)/3,
            2: torch.ones(3)/3
            }
        else:
          color_to_size_probs_validation = {
            0: torch.tensor([1., 0., 0.]),
            1: torch.tensor([0., 1., 0.]),
            2: torch.tensor([0., 0., 1.])
            }
          color_to_shape_probs_validation = {
            0: torch.ones(3)/3,
            1: torch.ones(3)/3,
            2: torch.ones(3)/3
            }

      else:
        color_to_shape_probs_validation = {
          0: torch.tensor([1., 0., 0.]),
          1: torch.tensor([0., 1., 0.]),
          2: torch.tensor([0., 0., 1.])
          }
      
      shape_probs = torch.stack([color_to_shape_probs_validation[idx.item()] for idx in color.flatten()]).view(B, M, -1)
      shape = pyro.sample(f"shape", dist.Categorical(probs=shape_probs).expand([B, M]))

      # check if there is a 1 (a parent) in the final row (size)
      if "size" in params["latent_vars"] and 1 in params["adj_matrix"][-1, :]: 
        size_root = False
        cause = color
        conditionals = color_to_size_probs_validation

      else: 
        size_root = True
        cause = None
        conditionals = None

    locX_mu_, locY_mu_ = torch.ones(B, M)*0.5, torch.ones(B, M)*0.5
    size_b_ = torch.zeros(B, M)      
    i = 0

    for b in range(B):
        check_dict = {} # stores locations and sizes for all objects within a scene
        for n in range(M):
          if objects_mask[b, n]:
              # if objects have been sampled already, we need to check for overlaps 
              # while sampling new objects to add to the scene
              if n > 0:
                # sample proposed locations
                locX_mu, locY_mu = sample_loc(i) 
                size = sample_size(i, b, n, size_root, cause, conditionals)
                
                # check if proposed object violates occlusion
                # 'check_dict' holds the properties of all previously accepted objects  
                #occlusion_flag = check_occlusion(check_dict, locX_mu.item(), locY_mu.item(), size_mapping[size[b, n].item()])                    
                occlusion_flag = check_occlusion(check_dict, locX_mu.item(), locY_mu.item(), size_vals[size.item()])                    
                
                while occlusion_flag:

                    #logger.info(f"{b} - {n} - {occlusion_flag} - {i}")
                    
                    i += 1
                    # sample proposed locations
                    locX_mu, locY_mu = sample_loc(i)
                    size = sample_size(i, b, n, size_root, cause, conditionals)
                    #occlusion_flag = check_occlusion(check_dict, locX_mu.item(), locY_mu.item(), size_mapping[size[b, n].item()]) 
                    occlusion_flag = check_occlusion(check_dict, locX_mu.item(), locY_mu.item(), size_vals[size.item()])
                
                check_dict[n] = {} # init dict for object 'n'
                check_dict[n]['locX'] = locX_mu.item()
                check_dict[n]['locY'] = locY_mu.item()
                #check_dict[n]['size'] = size_mapping[size[b, n].item()]
                check_dict[n]['size'] = size_vals[size.item()]
              
              # if n == 0, then just add the sampled locations and correspondent object's size to 'check_dict'
              else:
                locX_mu, locY_mu = sample_loc(i)
                size = sample_size(i, b, n, size_root, cause, conditionals)
                
                check_dict[n] = {} # init dict for object 'n'
                check_dict[n]['locX'] = locX_mu.item()
                check_dict[n]['locY'] = locY_mu.item()
                check_dict[n]['size'] = size_vals[size.item()]
              
              locX_mu_[b, n] = locX_mu
              locY_mu_[b, n] = locY_mu
              size_b_[b, n] = size
      
    locXfn = MyNormal(locX_mu_, torch.tensor(0.01)).get_dist()
    locYfn = MyNormal(locY_mu_, torch.tensor(0.01)).get_dist()
    with pyro.poutine.mask(mask=objects_mask):
        locX = pyro.sample(f"locX", locXfn)
        locY = pyro.sample(f"locY", locYfn)
        
        if len(size_vals) > 1:
          size = pyro.sample(f"size", dist.Delta(size_b_))
        else:
          size = torch.zeros(B, M)
           
  
  scenes = []
  for b in range(B):
    objects = []
    for m in range(M):
      if objects_mask[b, m]:
        objects.append({
            "shape": shape_vals[shape[b][m].item()],
            "color": color_vals[color[b][m].item()],
            "size": size_vals[int(size[b][m].item())],
            "position": (locX[b, m].item(), locY[b, m].item())
        })

    scenes.append(objects)
  return scenes

def model(observations={"image": torch.zeros((1, 3, 128, 128))}, training=False):

  #init_time = time.time()
  
  B = params['batch_size'] if params["running_type"] == "train" else params['num_inference_samples']
  
  scenes = sample_scenes(training)
  rendered_scenes = render(scenes)
  img = torch.stack([img_transform(s) for s in rendered_scenes])
  
  #render_time = time.time() - init_time
  #logger.info(f"Batch generation duration: {render_time} - {render_time/B} per sample")

  llh_uncertainty = 0.001 if params['running_type'] == "train" else 0.05
  likelihood_fn = MyNormal(img, torch.tensor(llh_uncertainty)).get_dist()
  pyro.sample("image", likelihood_fn.to_event(3), obs=observations["image"])


# def old_model(observations={"image": torch.zeros((1, 3, 128, 128))}, show='all', save_obs=None, N=None):
                        
#   poisson_param = 5.
#   shape_vals = ["ball", "square"]
#   size_vals = ["small", "medium", "large"]
#   color_vals = ["red", "green", "blue"]

#   n_objects = pyro.sample("N", MyPoisson(torch.tensor(poisson_param), validate_args = False), obs=N)
#   if type(n_objects) == Tensor: n_objects = to_int(n_objects)
 
#   if params["infer_background"]:
#     bgR = pyro.sample("bgR", dist.Uniform(0., 1.))
#     bgG = pyro.sample("bgG", dist.Uniform(0., 1.))
#     bgB = pyro.sample("bgB", dist.Uniform(0., 1.))
#     background = [bgR, bgG, bgB]
#   else: background = None
  
#   shape_obj, size_obj, color_obj, locx_obj, locy_obj = [], [], [], [], []

#   for n in range(n_objects):
#     shape = pyro.sample(f"shape_{n}", CategoricalVals(vals=shape_vals,
#                                                      probs=torch.tensor([1/len(shape_vals) for _ in range(len(shape_vals))])))
#     shape_obj.append(shape)
#     size = pyro.sample(f"size_{n}", CategoricalVals(vals=size_vals, probs=torch.tensor([1/len(size_vals) for _ in range(len(size_vals))])))
#     size_obj.append(size)
#     color = pyro.sample(f"color_{n}", CategoricalVals(vals=color_vals,
#                                                      probs=torch.tensor([1/len(color_vals) for _ in range(len(color_vals))])))
#     color_obj.append(color)
    
#     locX, locY = pyro.sample(f"locX_{n}", dist.Uniform(0., 1.)), pyro.sample(f"locY_{n}", dist.Uniform(0., 1.))
    
#     # if n == 0:
#     #   locX, locY = pyro.sample(f"locX_{n}", dist.Uniform(0., 1.)), pyro.sample(f"locY_{n}", dist.Uniform(0., 1.))
#     # else:
#     #   locX = torch.rand(1)
#     #   locY = torch.rand(1)

#     # if n > 0:
#     #   # gather the locations and sizes of all previously sampled objects
#     #   all_locs = [(locx_obj[i], locy_obj[i]) for i in range(n)]
#     #   all_sizes = [size_obj[i] for i in range(n)]
      
#     #   occlusion_set = [occlusion((locX, locY, size), (all_locs[j][0], all_locs[j][1], all_sizes[j])) for j in range(len(all_locs))]
#     #   while any(occlusion_set):
#     #     locX, locY = torch.rand(1), torch.rand(1)
#     #     occlusion_set = [occlusion((locX, locY, size), (all_locs[j][0], all_locs[j][1], all_sizes[j])) for j in range(len(all_locs))]  
    
#     #   locX, locY = pyro.sample(f"locX_{n}", dist.Uniform(0., 1.), obs=locX), pyro.sample(f"locY_{n}", dist.Uniform(0., 1.), obs=locY)
    
#     locx_obj.append(locX)
#     locy_obj.append(locY)    

#   if show == 'all': img = img_transform(render(shape_obj, size_obj, color_obj, locx_obj, locy_obj, background)).unsqueeze(0)
#   else:
#     if isinstance(show, list):
#       show_id = int(show[0].split('_')[1])
#       prop_list = [shape_obj, size_obj, color_obj, locx_obj, locy_obj]
#       for p in prop_list:
#         for j in range(len(p)):
#           if j > show_id: p[j] = None
      
#       img = img_transform(render(shape_obj, size_obj, color_obj, locx_obj, locy_obj, background)).unsqueeze(0)
    
#     else: raise ValueError(f"show variable equal to '{show}' unknown...")

#   with pyro.plate(observations["image"].shape[0]):
#     #pyro.sample("image", MyBernoulli(img, validate_args=False).to_event(3), obs=observations["image"])
#     likelihood_fn = MyNormal(img, torch.tensor(0.1)).get_dist()
#     pyro.sample("image", likelihood_fn.to_event(3), obs=observations["image"])
