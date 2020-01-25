import torch 
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont

class VOCDataset(data.Dataset):
    def __init__(self, data_file, image_size=(448,448), grid_size=7, transform=None):        
        self.transform = transform
        self.image_size = image_size
        self.grid_size = grid_size
        with open(data_file) as file:            
            self.image_paths = [x.strip() for x in file.readlines()]            

    def __len__(self):
        return 4#TODO: Use len(self.image_paths)

    """
    Returns (Image, [det])
    where Image datatype depends on the provided transforms else PIL image
    det is in the center normalised coordinates where x,y are normalised by the image width and height
        and width,w & height,h are normalised wrt to the image width and height.
    """
    def __getitem__(self, idx):
        img_file = self.image_paths[idx]
        label_file = img_file.replace("JPEGImages","labels").replace("jpg","txt")
        annotation_file = img_file.replace("JPEGImages","Annotations").replace("jpg","xml")
        
        img = Image.open(img_file).resize(self.image_size, Image.ANTIALIAS)

        with open(label_file) as file:
            # <object-class> <x> <y> <width> <height>            
            detections = file.readlines()
            detections = [x.split() for x in detections]
            detections = [[float(c) for c in detection] for detection in detections]

        if self.transform:            
            img = self.transform(img)
        
        detections = torch.Tensor(detections)#.unsqueeze(0)
        # print(img.size(), detections.size())
        return (img, detections)

