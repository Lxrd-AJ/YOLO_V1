import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils as utils
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import ctypes
import os
import random
from PIL import Image
from pprint import pprint
from collections import OrderedDict
from data.voc_dataset import VOCDataset
from utilities import show_detection, parse_config, build_class_names, iou, convert_YOLO_to_center_coords, convert_center_coords_to_YOLO, gnd_truth_tensor
from yolo_v1 import Yolo_V1


def criterion(output, target):
    total_loss = 0.0
    num_grids = output.size()
    for grid_x in range(num_grids[0]):
        for grid_y in range(num_grids[1]):
            truth_bbox = target[grid_x, grid_y]
            pred_bboxs = output[grid_x, grid_y]
            
            # Find the intersection over unio between the two predicted bounding boxes at the grid location
            bbox_1, bbox_2 = pred_bboxs[0:5], pred_bboxs[5:10]
            _bbox_1 = convert_YOLO_to_center_coords(bbox_1[1:], grid_x, grid_y, _STRIDE_)
            _bbox_2 = convert_YOLO_to_center_coords(bbox_2[1:], grid_x, grid_y, _STRIDE_)
            _truth_bbox = convert_YOLO_to_center_coords(truth_bbox[1:5], grid_x, grid_y, _STRIDE_) 
            
            class_probs = pred_bboxs[10:]
            truth_probs = truth_bbox[5:]

            if truth_bbox.sum() > 0: #there is an object in this class
                
                max_bbox, min_bbox = (bbox_1, bbox_2) if iou(_bbox_1,_truth_bbox) > iou(_bbox_2, _truth_bbox) else (bbox_2, bbox_1)
                confidence = max(iou(_bbox_1, _truth_bbox),iou(_bbox_2, _truth_bbox))
                truth_bbox[0] = confidence #the ground truth data uses confidence 
                Q = torch.eye(5) * 5
                Q[0,0] = 1
                z = max_bbox - truth_bbox[0:5]
                #loss is weighted bounding box regression + weighted no object + class probabilities
                loss_gx_gy = (z.t().matmul(Q).matmul(z)) + (0.5 * min_bbox[0]) + (truth_probs - class_probs).t().matmul(truth_probs - class_probs)                            
            else:
                loss_gx_gy = 0.5 * ((bbox_1[0]*bbox_1[0]) + (bbox_2[0]*bbox_2[0])) #0.5 is the weight when there is no object
            
            total_loss += loss_gx_gy
    return total_loss


_GRID_SIZE_ = 7
_IMAGE_SIZE_ = (448,448)
_BATCH_SIZE_ = 1
_STRIDE_ = _IMAGE_SIZE_[0] / 7
_DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_NUM_EPOCHS_ = 100

# No need to resize here in transforms as the dataset class does it already
transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #TODO: Add this when training
])


dataset = { x: VOCDataset(f"./data/{x}.txt", grid_size=_GRID_SIZE_, transform=transform) 
            for x in ['train','test','val']}
dataloader = {x: utils.data.DataLoader(dataset[x], batch_size=_BATCH_SIZE_, shuffle=True, num_workers=4)
                for x in ['train','test','val']}

for x in ['train','test','val']:
    print(f"{x} dataset size => {len(dataset[x])}")

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

if __name__ == "__main__":    
    imagenet_config = "./extraction_imagenet.cfg"
    blocks = parse_config(imagenet_config)
    class_names = build_class_names("./voc.names")

    model = Yolo_V1(class_names, 7, blocks)
    # optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimiser = optim.SGD([
                {'params': model.extraction_layers.parameters(), 'lr': 1e-4},
                {'params': model.final_conv.parameters()},
                {'params': model.linear_layers.parameters()}
            ], lr=1e-2, momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.1) #for transfer learning

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(_DEVICE_)

    for epoch in range(_NUM_EPOCHS_):
        print(f"Epoch {epoch}/{_NUM_EPOCHS_}")
        epoch_loss = 0.0
        model.train()

        for idx, data in enumerate(dataloader['train'],0):
            images, detections = data            
            
            optimiser.zero_grad()
            
            predictions = model(images)

            batch_loss = 0.0
            iteration_loss = 0.0
            num_batch = detections.size(0)

            # Having 3 foor-loops might be slow, this could be improved later as premature optimisation is the root of all evils
            for batch_idx in range(num_batch):
                pred_detections = predictions[batch_idx].transpose(0,2) #convert the dimension from 30x7x7 to 7x7x30                
                target_detections = convert_center_coords_to_YOLO(detections[batch_idx], _GRID_SIZE_)
                target_tensor = gnd_truth_tensor(target_detections)                

                loss = criterion(pred_detections, target_tensor)
                batch_loss += loss
                
            iteration_loss = batch_loss / num_batch
            epoch_loss += iteration_loss
            print(f"Iteration {idx}/{len(dataloader['train'])//_BATCH_SIZE_}: Loss = {iteration_loss}")
            
            iteration_loss.backward()
            optimiser.step()

            exit(0)
        epoch_loss = epoch_loss / len(dataloader['train'])
        print(f"Average Epoch loss is {epoch_loss}")        

# See https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb for inspiration

