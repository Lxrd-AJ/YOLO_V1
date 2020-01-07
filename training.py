import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms as transforms
import numpy as np
import ctypes
import os
import random
from PIL import Image
from pprint import pprint
from collections import OrderedDict
from data.voc_dataset import VOCDataset
from utilities import show_detection


transform = transforms.Compose([
    # transforms.ToTensor()
])
train_dataset = VOCDataset("./data/2012_train.txt", transform=transform)

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

if __name__ == "__main__":    
    rand_idx = random.randint(0, len(train_dataset))
    print(f"Using image at {rand_idx}")
    img, dets = train_dataset[rand_idx]

    print(dets)
    for bbox in dets:
        show_detection(img, bbox[1:], name=classes[int(bbox[0])], colour="red")
    
    img.show()

    # for i in range(len(train_dataset)):
    #     print(train_dataset[i].size)


