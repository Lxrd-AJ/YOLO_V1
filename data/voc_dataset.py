import torch 
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class VOCDataset(data.Dataset):
    def __init__(self, data_file, image_size=(448,448), transform=None):        
        self.transform = transform
        self.image_size = image_size
        with open(data_file) as file:            
            self.image_paths = [x.strip() for x in file.readlines()]            

    def __len__(self):
        return len(self.image_paths)

    """
    Returns (Image, [det])
    where Image is a torch tensor with format CxWxH
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
            
            # for detection in detections:
            #     print(detection)
            #     show_detection(img, detection)
            #     print()
            # img.show()
            
        # with open(annotation_file) as file:
        #     print(file.read())

        if self.transform:
            img = self.transform(img)
        return img, detections

