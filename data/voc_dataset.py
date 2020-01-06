import torch 
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from PIL import Image


class VOCDataset(data.Dataset):
    def __init__(self, data_file, transform=None):        
        self.transform = transform
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
        print(img_file, label_file)
        img = Image.open(img_file)
        print(img.size)

        with open(label_file) as file:
            # <object-class> <x> <y> <width> <height>
            print(file.read())

        if self.transform:
            img = self.transform(img)
        return None

