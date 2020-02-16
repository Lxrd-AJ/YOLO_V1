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
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image
from pprint import pprint
from collections import OrderedDict
from data.voc_dataset import VOCDataset
from utilities import draw_detection, parse_config, build_class_names, iou, convert_YOLO_to_center_coords, convert_center_coords_to_YOLO, gnd_truth_tensor, imshow, im2PIL, draw_detections
from yolo_v1 import Yolo_V1
from torchviz import make_dot
from graphviz import Source
from loss import criterion



def batch_collate_fn(batch):    
    images = [item[0].unsqueeze(0) for item in batch]
    detections = [item[1] for item in batch]  
    images = torch.cat(images,0)
    return (images, detections)


def evaluate(model, dataloader):
    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for idx, data in enumerate(dataloader, 0):
            X, Y = data #transform(data[0]), data[1]
            X = X.to(_DEVICE_)
            res = model(X)
            batch_loss = 0.0
            for batch_idx in range(X.size(0)):
                pred_detections = res[batch_idx].transpose(0,2) #convert the dimension from 30x7x7 to 7x7x30 for use in `criterion`               
                target_detections = convert_center_coords_to_YOLO(Y[batch_idx], _GRID_SIZE_)
                target_tensor = gnd_truth_tensor(target_detections)
                loss = criterion(pred_detections, target_tensor, _STRIDE_)
                batch_loss += loss
            batch_loss = batch_loss / X.size(0)
            eval_loss += batch_loss
    eval_loss = eval_loss / len(dataloader)
    return eval_loss



_GRID_SIZE_ = 7
_IMAGE_SIZE_ = (448,448)
_BATCH_SIZE_ = 8
_STRIDE_ = _IMAGE_SIZE_[0] / 7
_DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_NUM_EPOCHS_ = 25#150 #Maybe try using len(dataset) / 10

# No need to resize here in transforms as the dataset class does it already
transform = transforms.Compose([
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


dataset = { x: VOCDataset(f"./data/{x}.txt", image_size=_IMAGE_SIZE_, grid_size=_GRID_SIZE_, transform=transform)
            for x in ['train','test','val']}
dataloader = {x: utils.data.DataLoader(dataset[x], batch_size=_BATCH_SIZE_, shuffle=True, num_workers=4, collate_fn=batch_collate_fn)
                for x in ['train','test','val']}

for x in ['train','test','val']:
    print(f"{x} dataset size => {len(dataset[x])}")

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

if __name__ == "__main__":    
    imagenet_config = "./extraction_imagenet.cfg"
    blocks = parse_config(imagenet_config)
    class_names = build_class_names("./voc.names")

    model = Yolo_V1(class_names, _GRID_SIZE_, _IMAGE_SIZE_)
    model.init_weights()
    # optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimiser = optim.SGD([
                {'params': model.extraction_layers.parameters(), 'lr': 1e-4}, #1e-3
                {'params': model.final_conv.parameters(), 'lr': 1e-2}, 
                {'params': model.linear_layers.parameters()}
            ], lr=1e-1, momentum=0.9)
    
    #The learning rate scheduler will be added later post model debugging
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=5, gamma=0.1) #for transfer learning

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(_DEVICE_)

    # Save the model before training starts
    # torch.save(model.state_dict(), "./yolo_v1_model.pth")

    train_since = time.time()
    avg_train_loss = []
    avg_test_loss = []
    for epoch in range(_NUM_EPOCHS_):
        print(f"Epoch {epoch+1}/{_NUM_EPOCHS_}\t Learning Rate = {exp_lr_scheduler.get_lr()}")
        print("-----" * 15)
        epoch_loss = 0.0
        epoch_since = time.time()
        model.train()

        for idx, data in enumerate(dataloader['train'],0):
            images, detections = data
            images = images.to(_DEVICE_)            
            
            optimiser.zero_grad()
            
            #+++++++++++++++++++++++++++++++++++
            #Passing junk data
            # images = torch.randn(2, 3, 448, 448)
            # detections = torch.rand(_BATCH_SIZE_, 1, 5)
            # detections[:,:,0] *= 20
            
            # x = im2PIL(images[0])            
            # x = draw_detections(x, detections[0], class_names)            
            #+++++++++++++++++++++++++++++++++++
            
            predictions = model(images)

            batch_loss = 0.0
            iteration_loss = 0.0
            num_batch = len(detections)

            with torch.set_grad_enabled(True):
                # Having 3 foor-loops might be slow, this could be improved later as premature optimisation is the root of all evils
                for batch_idx in range(num_batch):
                    pred_detections = predictions[batch_idx].transpose(0,2) #convert the dimension from 30x7x7 to 7x7x30                
                    target_detections = convert_center_coords_to_YOLO(detections[batch_idx], _GRID_SIZE_)
                    target_tensor = gnd_truth_tensor(target_detections)                

                    loss = criterion(pred_detections, target_tensor, _STRIDE_)
                    
                    batch_loss += loss                    
                    
                iteration_loss = batch_loss / num_batch            
                epoch_loss += iteration_loss.item()
                if True: #TODO: Remove #idx % 1000 == 0:
                    print(f"\tIteration {idx+1}/{len(dataloader['train'])}: Loss = {iteration_loss.item()}")
                
                    # m_arch = make_dot(iteration_loss, params=dict(model.named_parameters()))
                    # Source(m_arch).render("./model_arch")

                iteration_loss.backward()
                optimiser.step()

        epoch_loss = epoch_loss / len(dataloader['train'])
        epoch_elapsed = time.time() - epoch_since
        print(f"\tAverage Train Epoch loss is {epoch_loss:.2f} [{epoch_elapsed//60:.0f}m {epoch_elapsed%60:.0f}s]")
        avg_train_loss.append(epoch_loss)

        exp_lr_scheduler.step()

        #evaluate on the test dataset
        test_loss = evaluate(model, dataloader['test'])
        avg_test_loss.append(test_loss)
        print(f"\tAverage Test Loss is {test_loss:.2f}")

        # Save the model parameters https://pytorch.org/tutorials/beginner/saving_loading_models.html
        if True: #TODO: Remove after debugging #epoch % 10 == 0:
            torch.save(model.state_dict(), "./yolo_v1_model.pth")
            torch.save(optimiser.state_dict(), "./optimiser_yolo.pth")

        #Make some plots baby! 
        plt.plot(avg_train_loss,'r',label='Train',marker='o')
        plt.plot(avg_test_loss,'b',label='Test',marker='o')
        plt.xticks(np.arange(0,_NUM_EPOCHS_))
        
        plt.title(f"Train & Test loss using {len(dataset['train'])} images")
        plt.grid(True)
        plt.savefig(f"./{len(dataset['train'])}_elems_train_val_loss.png")
    
    plt.legend()
    plt.savefig(f"./{len(dataset['train'])}_elems_train_val_loss.png")

    #Evaluate on the validation dataset
    val_loss = evaluate(model, dataloader['val'])
    train_elapsed = time.time() - train_since
    print(f"Validation loss is {val_loss:.2f}")
    print(f"Total training time is [{train_elapsed//60:.0f}m {train_elapsed%60:.0f}s]")

    



"""
- Try adding batch norm and setting bias to false (conv2d)
- investigate why the model is giving 0 confidence to the detections
"""