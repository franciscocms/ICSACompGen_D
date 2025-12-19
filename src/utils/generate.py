import copy
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
import matplotlib

from ..main import params

def hsv_to_rgb(hsv):
  return (matplotlib.colors.hsv_to_rgb(hsv)*255).astype(int)

def img_to_tensor(image):
  img_transform = transforms.Compose([transforms.ToTensor()])
  return img_transform(image).unsqueeze(0)

def norm(v, min, max):
    return (v-min)/(max-min)

def render(scene, resolution = (128, 128)):  
  background_code = (0, 0, 0)
  image = Image.new('RGB', resolution, background_code)
  draw = ImageDraw.Draw(image)

  for object in scene:
    scale, color, x, y = object
    
    scale *= params["factor_scaling"]["scale"]
    #color *= params["factor_scaling"]["color"] # color remains between [0., 1.]
    x *= params["factor_scaling"]["x_pos"]
    y *= params["factor_scaling"]["y_pos"]
    
    draw.ellipse(
        [x - scale//2, y - scale//2, x + scale//2, y + scale//2],
        fill=hsv_to_rgb([color, 1., 1.])
        )
  return image