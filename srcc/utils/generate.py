import copy
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
import matplotlib

from ..main import params

import logging
logger = logging.getLogger(params["running_type"])

def hsv_to_rgb(hsv):
  rgb = tuple((matplotlib.colors.hsv_to_rgb(hsv)*255).astype(int))
  return rgb

def img_to_tensor(image):
  img_transform = transforms.Compose([transforms.ToTensor()])
  return img_transform(image).unsqueeze(0)

def norm(v, min, max):
    return (v-min)/(max-min)

def render(scene, resolution = (128, 128)):  
  background_code = (0, 0, 0)
  factor = 8
  image = Image.new('RGB', (resolution[0]*factor, resolution[1]*factor), background_code)
  draw = ImageDraw.Draw(image)

  for object in scene:
    scale, color, x, y = object
    
    scale *= params["factor_scaling"]["scale"] * factor
    #color *= params["factor_scaling"]["color"] # color remains between [0., 1.]
    x *= params["factor_scaling"]["x_pos"] * factor
    y *= params["factor_scaling"]["y_pos"] * factor
    
    draw.ellipse(
        [x - scale//2, y - scale//2, x + scale//2, y + scale//2],
        fill=hsv_to_rgb([color, 1., 1.]),
        width=factor
        )
  image = image.resize((64, 64), Image.LANCZOS)
  return image