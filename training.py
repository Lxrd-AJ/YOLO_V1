import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils as utils
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import time
import ctypes
import os
import random
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image
from pprint import pprint
from collections import OrderedDict
from data.voc_dataset import VOCDataset
from utilities import draw_detection, parse_config, build_class_names, iou, convert_center_coords_to_YOLO, gnd_truth_tensor, imshow, im2PIL, draw_detections
from yolo_v1 import Yolo_V1
# from torchviz import make_dot
# from graphviz import Source
from loss import criterion
from transforms import RandomBlur, RandomHorizontalFlip, RandomVerticalFlip



def batch_collate_fn(batch):    
    images = [item[0].unsqueeze(0) for item in batch]    
    
    detections = []
    for item in batch:
        det = item[1]
        image_detections = torch.zeros(1, _GRID_SIZE_, _GRID_SIZE_, 5)
        for cell in det:            
            gx = math.floor(_GRID_SIZE_ * cell[1])
            gy = math.floor(_GRID_SIZE_ * cell[2])
            image_detections[0,gx,gy,0:4] = cell[1:]
            image_detections[0,gx,gy,4] = cell[0]        
        detections.append(image_detections)

    images = torch.cat(images,0)
    # For every item in the batch
    # There is a tensor of size _GRID_SIZE_ x _GRID_SIZE_ x 5 
    # where each grid cell contains <x> <y> <w> <h> <class>, 
    # where <x> <y> <w> <h> are normalised relative to the image size and width
    detections = torch.cat(detections,0)    
    return (images, detections)


def evaluate(model, dataloader):
    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for idx, data in enumerate(dataloader, 0):
            X, Y = data #transform(data[0]), data[1]
            X = X.to(_DEVICE_)
            Y = Y.to(_DEVICE_)
            res = model(X)
            batch_loss = criterion(res, Y)            
            eval_loss += batch_loss
    eval_loss = eval_loss / len(dataloader)
    return eval_loss



_GRID_SIZE_ = 7
_IMAGE_SIZE_ = (448,448)
_BATCH_SIZE_ = 16
_STRIDE_ = _IMAGE_SIZE_[0] / 7
_DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_NUM_EPOCHS_ = 150


# No need to resize here in transforms as the dataset class does it already
image_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),        
    RandomBlur(probability=0.2)
])
#Image detection pair transforms
pair_transform = transforms.Compose([
    RandomHorizontalFlip(probability=0.5),
    RandomVerticalFlip(probability=0.3)
])

normalise_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

erase_transform = transforms.RandomErasing(p=0.1, scale=(0.02, 0.12), ratio=(0.1, 1.1))


dataset = { 
    'train': VOCDataset(f"./data/train.txt", image_size=_IMAGE_SIZE_, grid_size=_GRID_SIZE_,transform=[image_transform, normalise_transform, erase_transform], pair_transform=pair_transform),
    'val': VOCDataset(f"./data/val.txt", transform=[normalise_transform])
}

dataloader = {x: utils.data.DataLoader(dataset[x], batch_size=_BATCH_SIZE_, shuffle=True, num_workers=4, collate_fn=batch_collate_fn)
                for x in ['train','val']}

for x in ['train','val']:
    print(f"{x} dataset size => {len(dataset[x])}")

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    class_names = build_class_names("./voc.names")

    model = Yolo_V1(class_names, _GRID_SIZE_, _IMAGE_SIZE_)
    model.init_weights()
    # optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimiser = optim.SGD([
                {'params': model.extraction_layers.parameters(), 'lr': 1e-3}, #1e-3
                {'params': model.final_conv.parameters(), 'lr': 1e-2}, 
                {'params': model.linear_layers.parameters()}
            ], lr=1e-1, momentum=0.9)
    
    # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=15, gamma=0.1) #for transfer learning
    exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(_DEVICE_)

    train_since = time.time()
    avg_train_loss = []
    avg_val_loss = []
    for epoch in range(_NUM_EPOCHS_):
        lr = [group['lr'] for group in optimiser.param_groups]
        print(f"Epoch {epoch+1}/{_NUM_EPOCHS_}\t Learning Rate = {lr}")
        print("-----" * 15)
        epoch_loss = 0.0
        epoch_since = time.time()
        model.train()

        for idx, data in enumerate(dataloader['train'],0):
            with torch.set_grad_enabled(True):
                images, detections = data
                
                images = images.to(_DEVICE_)
                detections = detections.to(_DEVICE_)
                
                optimiser.zero_grad()
                
                predictions = model(images)

                batch_loss = criterion(predictions, detections)                
                epoch_loss += batch_loss.item()
                
                if idx % 100 == 0:
                    print(f"\tIteration {idx+1}/{len(dataloader['train'])}: Loss = {batch_loss.item()}")
                
                    # m_arch = make_dot(batch_loss, params=dict(model.named_parameters()))
                    # Source(m_arch).render("./model_arch")

                batch_loss.backward()
                optimiser.step()

        epoch_loss = epoch_loss / len(dataloader['train'])
        epoch_elapsed = time.time() - epoch_since
        print(f"\tAverage Train Epoch loss is {epoch_loss:.2f} [{epoch_elapsed//60:.0f}m {epoch_elapsed%60:.0f}s]")
        avg_train_loss.append(epoch_loss)
        

        #evaluate on the validation dataset
        val_loss = evaluate(model, dataloader['val'])
        avg_val_loss.append(val_loss)
        print(f"\tAverage Val Loss is {val_loss:.2f}")

        exp_lr_scheduler.step(val_loss)

        # Save the model parameters https://pytorch.org/tutorials/beginner/saving_loading_models.html
        if epoch % 10 == 0:
            torch.save(model.state_dict(), "./yolo_v1_model.pth")
            # torch.save(optimiser.state_dict(), "./optimiser_yolo.pth")

        #Make some plots baby! 
        plt.plot(avg_train_loss,'r',label='Train',marker='o')
        plt.plot(avg_val_loss,'b',label='Val',marker='x')
        plt.xticks(np.arange(0,_NUM_EPOCHS_,10))
        
        plt.title(f"Train & Val loss using {len(dataset['train'])} images")
        plt.grid(True)
        plt.savefig(f"./{len(dataset['train'])}_elems_train_val_loss.png")
    
    # plt.legend()
    plt.savefig(f"./{len(dataset['train'])}_elems_train_val_loss.png")

    #Evaluate on the validation dataset
    train_elapsed = time.time() - train_since
    print(f"Total training time is [{train_elapsed//60:.0f}m {train_elapsed%60:.0f}s]")
