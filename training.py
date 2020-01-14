import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils as utils
import torchvision.transforms as transforms
import numpy as np
import ctypes
import os
import random
import math
from PIL import Image
from pprint import pprint
from collections import OrderedDict
from data.voc_dataset import VOCDataset
from utilities import show_detection, parse_config, build_class_names
from yolo_v1 import Yolo_V1


def criterion(output, target):
    pass

"""
Convert from center normalised coordinates to YOLO bounding box encoding

There are two methods of approaching it
- One method requires converting from the center noormalised coordinates back into the global image coords
    it also requires knowledge of the image width/height to calculate the stride and to convert to the global coords
    (229 - (64*3))/64 ; 229 is the center_x; 64 is the stride, 3 is the grid cell

- Another method utilises the current center normalised coordinates.
    For example, assuming a grid size of 7
    encode the normalised center x in terms of the grid cells by using floor(7 * center_x) => g_x
    the offset from the grid cell is then given by (7 * center_x) - g_x
"""
def convert_center_coords_to_YOLO(detections, grid_size=7):
    for idx, detection in enumerate(detections):        
        bbox = detection[1:]
        # Convert from Pascal VOC center normalised coordinates to YOLO box encoding
        bbox[0] = (grid_size * bbox[0]) - math.floor(grid_size * bbox[0])
        bbox[1] = (grid_size * bbox[1]) - math.floor(grid_size * bbox[1])
        detections[idx][1:] = bbox
    return detections

_GRID_SIZE_ = 7
_DEVICE_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #TODO: Add this when training
])
train_dataset = VOCDataset("./data/2012_train.txt", grid_size=_GRID_SIZE_, transform=transform)
dataloader = utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

if __name__ == "__main__":    
    imagenet_config = "./extraction_imagenet.cfg"
    blocks = parse_config(imagenet_config)
    class_names = build_class_names("./voc.names")

    model = Yolo_V1(class_names, 7, blocks)


# How to use VOCDataset
# rand_idx = random.randint(0, len(train_dataset))
#     print(f"Using image at {rand_idx}")
#     img, dets = train_dataset[rand_idx]

#     print(dets)
#     x = transforms.ToPILImage()(img)
#     for bbox in dets:
#         # print(type(img))
#         show_detection(x, bbox[1:], name=classes[int(bbox[0])], colour="red")
#     bbox_yolo_encoding = convert_center_coords_to_YOLO(dets)
#     print(bbox_yolo_encoding)
#     x.show()