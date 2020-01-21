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
import math
from PIL import Image
from pprint import pprint
from collections import OrderedDict
from data.voc_dataset import VOCDataset
from utilities import show_detection, parse_config, build_class_names, iou, convert_YOLO_to_center_coords
from yolo_v1 import Yolo_V1


def criterion(output, target):
    return 0

"""
# TODO: Move this to the utilities class
uses the detections in the YOLO format to construct the a tensor in the grid coordinates to make it easier for loss calculations
"""
def gnd_truth_tensor(detections, grid_size=7, num_classes=20):    
    x = torch.zeros([grid_size, grid_size, num_classes+5], dtype=torch.float32)
    for i in range(detections.size(0)):
        grid_x, grid_y = int(detections[i,5]), int(detections[i,6])
        cls_idx = int(detections[i,0])
        bbox = detections[i,1:5]
        x[grid_x,grid_y,1:5] = bbox
        x[grid_x, grid_y, cls_idx] = 1        
    return x

"""
TODO: Move this to the utilities class
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
    res = []    
    for idx, detection in enumerate(detections):        
        bbox = detection[1:]
        gx = math.floor(grid_size * bbox[0]) # grid x location of the cell
        gy = math.floor(grid_size * bbox[1]) # grid y location of the cell
        # Convert from Pascal VOC center normalised coordinates to YOLO box encoding
        bbox[0] = (grid_size * bbox[0]) - gx
        bbox[1] = (grid_size * bbox[1]) - gy
        bbox[2] = math.sqrt(bbox[2])
        bbox[3] = math.sqrt(bbox[3])
        
        # Adding the grid cell locations gx, gy to the detection to make loss calculations easier
        grid_cells = torch.Tensor([gx,gy])
        detection[1:] = bbox
        res.append(torch.cat([detection, grid_cells]).unsqueeze(0))    
    return torch.cat(res, dim=0) #detections

_GRID_SIZE_ = 7
_IMAGE_SIZE_ = (448,448)
_STRIDE_ = _IMAGE_SIZE_[0] / 7
_DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_NUM_EPOCHS_ = 100

transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #TODO: Add this when training
])
train_dataset = VOCDataset("./data/2012_train.txt", grid_size=_GRID_SIZE_, transform=transform)
dataloader = utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

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

        for idx, data in enumerate(dataloader,0):
            images, detections = data            
            
            optimiser.zero_grad()
            
            predictions = model(images)
            # predictions = model.transform_predict(predictions)

            batch_loss = 0.0
            num_batch = detections.size(0)
            # Having 3 foor-loops might be slow, this could be improved later as premature optimisation is the root of all evils
            for batch_idx in range(num_batch):
                pred_detections = predictions[batch_idx].transpose(0,2) #convert the dimension from 30x7x7 to 7x7x30                
                target_detections = convert_center_coords_to_YOLO(detections[batch_idx], _GRID_SIZE_)
                target_tensor = gnd_truth_tensor(target_detections)                

                num_grids = pred_detections.size()
                
                loss = 0.0
                for grid_x in range(num_grids[0]):
                    for grid_y in range(num_grids[1]):
                        truth_bbox = target_tensor[grid_x, grid_y]
                        pred_bboxs = pred_detections[grid_x, grid_y]
                        
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
                        
                        loss += loss_gx_gy
                        # print(f"Cell ({grid_x},{grid_y}) loss = {loss_gx_gy}")
                # print(f"Grid Loss {loss}")
            batch_loss += loss / num_batch
            epoch_loss += batch_loss
            print(f"Iteration {idx}: Loss = {batch_loss}")

            batch_loss.backward()
            optimiser.step()
        epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch loss is {epoch_loss}")
        exit(0)

# See https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb for inspiration





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
