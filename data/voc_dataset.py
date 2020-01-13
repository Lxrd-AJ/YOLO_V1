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
        return len(self.image_paths)

    """
    Returns (Image, [det])
    where Image datatype depends on the provided transforms else PIL image
    det is in the Yolo bounding box encoding where x,y are normalised by the grid location
        and width,w & height,h are equal sqrt(w) & sqrt(h) respectively.
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

        # Convert from Pascal VOC center normalised coordinates to YOLO box encoding
        stride = self.image_size[0] // self.grid_size
        # for idx, detection in enumerate(detections):
        #     #TODO: convert the center x and y to YOLO grid coordinates                  
        #     bbox = detection[1:]
        #     print(idx)
        #     print(bbox)
        #     print(f"Stride {stride}")
        #     grid_x = (bbox[0] * self.image_size[0]) // stride
        #     grid_y = (bbox[1] * self.image_size[0]) // stride
        #     print(f"Grid x = {grid_x} and Grid y = {grid_y}")
        #     center_x = bbox[0] / grid_x
        #     center_y = bbox[1] / grid_y
        #     print(f"Center x = {center_x} and Center y = {center_y}")

        #     detection[3] = math.sqrt(bbox[2])
        #     detection[4] = math.sqrt(bbox[3])

        #     print(detection[1:])            
            
        if self.transform:
            img = self.transform(img)
        return img, detections

