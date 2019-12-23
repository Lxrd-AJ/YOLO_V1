import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import cv2 as cv
from torch.autograd import Variable
from models.utilities import *
from PIL import Image, ImageOps
from models.yolo_v1 import Yolo_V1

if __name__ == "__main__":
    imagenet_config = "./models/extraction_imagenet.cfg"
    blocks = parse_config(imagenet_config)
    class_names = build_class_names("./models/voc.names")

    model = Yolo_V1(class_names, 7, blocks)
    # model.load_extraction_weights("./models/yolov1.weights") #extraction.conv.weights    
    model.eval()
    
    # Load the test image
    # x = torch.randn(3,3,448,448) # x = x.unsqueeze(0)
    
    x = Image.open("./grab_drive_3_1.png")    
    x = x.resize((448,448)) #TODO: Research how to maintain the aspect ratio
    # x.show()
    x = np.array(x)    
    x_np = np.copy(x)
    x = x[:,:,:3].transpose((2,0,1))
    x = torch.from_numpy(x).float()#.div(255).unsqueeze(0)   
    x = ((x - x.mean()) / x.std()).unsqueeze(0)    

    x = Variable(x) 
    assert not torch.isnan(x).any()
    
    # print("Input size", x.size())
    
    with torch.no_grad():
        res = model(x)        
    print("Prediction size of", res.size())
    res = model.transform_predict(res)
    class_names = convert_cls_idx_name(model.class_names, res[0][:,5].numpy())
    res = res[0].numpy()
    #convert the width and height to bottom-right coords
    res[:,3] = res[:,1] + res[:,3]
    res[:,4] = res[:,2] + res[:,4]
    for i in range(res.shape[0]):
        box = res[i,:]        
        print(box[1:-1])
    # res = torch.flatten(res).view(x.size()[0],-1)
    # print("Flattened prediction",res.size())
    # print("Num elements in prediction",res.numel())
