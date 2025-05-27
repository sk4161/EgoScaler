import argparse
import torch
import os
from PIL import Image
from tqdm import tqdm
import glob
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from depth_anything_v2.dpt import DepthAnythingV2

class DepthAnything:
    def __init__(self, args, device):
        self.device = device
        self.model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
        self.model.load_state_dict(torch.load(args.pretrained_resource, map_location='cpu'))
        self.model.to(device)
        self.model.eval() 
        
    @torch.no_grad()
    def get_only_depth(self, pil_image: Image.Image, final_width:int, final_height: int):
        original_width, original_height = pil_image.size
        image = np.array(pil_image)  # np array
        image = image[:, :, ::-1]    # cv array
        pred = self.model.infer_image(image)

        # Resize color image and depth to final size
        resized_pred = Image.fromarray(pred).resize((final_width, final_height), Image.NEAREST)
        z = np.array(resized_pred)
        
        return z
    
    @torch.no_grad()
    def get_depth(
        self, 
        pil_image: Image.Image, 
        final_width:int, 
        final_height: int, 
        focal_len_x: int=0,
        focal_len_y: int=0,
        principal_point: int=0
    ):
        original_width, original_height = pil_image.size
        image = np.array(pil_image) # np array
        image = image[:,:,::-1] # cv array
        pred = self.model.infer_image(image)

        # Resize color image and depth to final size
        resized_pred = Image.fromarray(pred).resize((final_width, final_height), Image.NEAREST)
        z = np.array(resized_pred)
        
        if focal_len_x > 0 and focal_len_y > 0 and principal_point > 0:
            x, y = np.meshgrid(np.arange(final_width), np.arange(final_height))
            x = (x - principal_point) / focal_len_x
            y = (y - principal_point) / focal_len_y
            points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
            colors = np.array(pil_image).reshape(-1, 3) / 255.0
        else:
            points = None
            colors = None
        
        return z, points, colors