import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import random
# import cv2 as cv
from torch.autograd import Variable
from utilities import build_class_names, draw_detection, confidence_threshold, max_box
from PIL import Image, ImageOps
from yolo_v1 import Yolo_V1
from data.voc_dataset import VOCDataset

transform = transforms.Compose([    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

_IMAGE_SIZE_ = (448, 448)
_GRID_SIZE_ = 7
_STRIDE_ = _IMAGE_SIZE_[0] / _GRID_SIZE_
class_names = build_class_names("./voc.names")

dataset = VOCDataset(f"./data/val.txt", image_size=_IMAGE_SIZE_, grid_size=_GRID_SIZE_)

if __name__ == "__main__":
    model = Yolo_V1(class_names, 7)
    model.load_state_dict(torch.load('./yolo_v1_model.pth'))
    model.eval()
    
    # print(model)

    """
    - Show ground truth detections
    - Rearrange predicted detections
    - Perform Non-Maximum Suppression
    - Show Predicted detections after NMS
    """
    with torch.no_grad():
        # Show ground truth detections
        X, dets = dataset[random.randint(0, len(dataset))]
        X__ = X.copy()
        for det in dets:
            draw_detection(X, det[1:], class_names[int(det[0])])        
        # X.show()

        # Rearrange predicted detections
        X_ = transform(X).unsqueeze(0)
        predictions = model(X_)
        sz = predictions.size()    
        predictions = predictions.view(sz[0], sz[1], -1) # change from 1x30x7x7 to 1x30x49
        predictions = predictions.transpose(1,2).contiguous() #change from 1x30x49 to 1x49x30
        print(predictions.size())
        detection_results = {}
        for batch_idx in range(sz[0]): #Operate over the predictions in a batch
            pred = predictions[batch_idx]
            bboxes = pred[:,:10] #All the bboxes for every grid cell

            # convert predictions from YOLO coordinates to center coordinates
            rng = np.arange(_GRID_SIZE_) # the range of possible grid coords
            cols, rows = np.meshgrid(rng, rng)
            #create a grid with each cell containing the (x,y) location multiplied by stride  
            rows = torch.FloatTensor(rows).view(-1,1)
            cols = torch.FloatTensor(cols).view(-1,1)
            grid = torch.cat((rows,cols),1) * _STRIDE_
            #convert the boxes to center coordinates (NOT NORMALISED BY THE IMAGE WIDTH)
            bboxes[:,1:3] = (bboxes[:,1:3] * _STRIDE_).round() + grid #1st box's center x,y
            bboxes[:,3:5] = (bboxes[:,3:5].pow(2) * _IMAGE_SIZE_[0]).round() #1st box's w & h
            bboxes[:,6:8] = (bboxes[:,6:8] * _STRIDE_).round() + grid #2nd box's center x,y
            bboxes[:,8:10] = (bboxes[:,8:10].pow(2) * _IMAGE_SIZE_[0]).round() #2nd box's w & h
            
            bboxes = max_box(bboxes[:,:5], bboxes[:,5:])
            
            #Get the predicted class at each grid cell
            class_probs = pred[:,10:] #this will be of size 49x20 for a grid size of 7x7 and 20 classes
            pred_class, class_idx = class_probs.max(1) #1 is along the rows i.e for each grid cell

            #Join the predicted classes `class_idx` with the bounding boxes
            bboxes = torch.cat((bboxes, class_idx.unsqueeze(1).float()), 1)

            #confidence threshold the bounding boxes by their class confidence scores            
            bboxes = confidence_threshold(bboxes, 0.5)
            print(bboxes)

            #TEST: Show the predictions
            for bbox in bboxes:                
                draw_detection(X__,bbox[1:5] / _IMAGE_SIZE_[0], class_names[int(bbox[5])], "red")
            # X__.show()
            # X.show()
        # print(predictions.size())





    # Load the test image
    # x = torch.randn(3,3,448,448) # x = x.unsqueeze(0)
    
    # x = Image.open("./grab_drive_3_1.png")    
    # x = x.resize((448,448)) #TODO: Research how to maintain the aspect ratio
    # # x.show()
    # x = np.array(x)    
    # x_np = np.copy(x)
    # x = x[:,:,:3].transpose((2,0,1))
    # x = torch.from_numpy(x).float()#.div(255).unsqueeze(0)   
    # x = ((x - x.mean()) / x.std()).unsqueeze(0)    

    # x = Variable(x) 
    # assert not torch.isnan(x).any()
    
    # # print("Input size", x.size())
    
    # with torch.no_grad():
    #     res = model(x)        
    # print("Prediction size of", res.size())
    # res = model.transform_predict(res)
    # class_names = convert_cls_idx_name(model.class_names, res[0][:,5].numpy())
    # res = res[0].numpy()
    # #convert the width and height to bottom-right coords
    # res[:,3] = res[:,1] + res[:,3]
    # res[:,4] = res[:,2] + res[:,4]
    # for i in range(res.shape[0]):
    #     box = res[i,:]        
    #     print(box[1:-1])
    # res = torch.flatten(res).view(x.size()[0],-1)
    # print("Flattened prediction",res.size())
    # print("Num elements in prediction",res.numel())
