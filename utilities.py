import numpy as np 
import torch
from PIL import Image, ImageDraw, ImageFont


"""
converts the output of a YOLO model to a center coordinate where x,y is the center of the box and w,h represent the width
and height of the image, this is not noormalised by the image width and height.
- bbox is in the format [x y w h]

NB: Calling this function on the ground truth bounding box encoded in the YOLO format would result in
a bounding box containing the x,y coordinates of the grid cell with no width and height
"""
def convert_YOLO_to_center_coords(bbox, grid_x, grid_y, stride, grid_size=7):
    img_size = stride * grid_size
    gx = (stride * (grid_x)) + (stride * bbox[0])
    gy = (stride * grid_y) + (stride * bbox[1])
    w = (bbox[2] * bbox[2]) * img_size
    h = (bbox[3] * bbox[3]) * img_size
    return torch.Tensor([gx, gy, w, h])
    

"""
- bbox cls <x> <y> <width> <height> is in normalised center coordinates where 
    x,y is the center of the box relative to the width and height of the image (grid cell)
        where x is ((x_max + x_min)/2)/width
    width and height is normalised relative to the width and height of `image`
"""
def show_detection(image, bbox, name, colour="white"):
    width, height = image.size
    
    box_width = int(bbox[2] * width)
    box_height = int(bbox[3] * height)
    
    center_x = int(bbox[0] * width)
    center_y = int(bbox[1] * height)    

    top_left = (center_x - box_width/2, center_y - box_height/2)
    bottom_right = (center_x + box_width/2, center_y + box_height/2)

    draw = ImageDraw.Draw(image)
    draw.rectangle((top_left, bottom_right), width=2, outline=colour)
    draw.text(top_left, name, font=ImageFont.truetype("Helvetica",15), fill=(250,150,118,255))


def parse_config(cfg_file):
    with open(cfg_file) as file:
        lines = file.read().split('\n')
        lines = [x for x in lines if len(x) > 0]
        lines = [x for x in lines if x[0] != '#'] #remove comments
        lines = [x.rstrip().lstrip() for x in lines]
        
        block = {}
        blocks = []

        for line in lines:
            if line[0] == '[':
                if len(block) != 0:
                    blocks.append(block)
                    block = {}
                block["type"] = line[1:-1].rstrip()                
            else:
                key, value = line.split("=")                
                block[key.rstrip()] = value.lstrip()
        blocks.append(block) 
        return blocks

def build_class_names(class_file):        
    fp = open(class_file,"r")
    names = fp.read().split("\n")
    names_dict = {idx:e for idx, e in enumerate(names)}    
    return names_dict

def max_box(b1, b2):
    assert b1.size() == b2.size()
    A = torch.zeros(b1.size()).float()
    for i in range(b1.size(0)):
        #0 is the index of the probability scores
        A[i,:] = b1[i,:] if b1[i,0] > b2[i,0] else b2[i,:]
    return A

def confidence_threshold(A, conf_thresh):
    #0 is the index of the confidence value
    conf_mask = (A[:,0] > conf_thresh).float().unsqueeze(1)
    conf_mask = conf_mask * A
    conf_mask = torch.nonzero(conf_mask[:,0]).squeeze()
    return A[conf_mask,:]    



"""
Non-vectorised form of iou. It expects a, b to be in the center coordinates form where
    - x,y is the center of the box
    - w,h is the width and height of the box
And a & b are in the form [x, y, w, h]
"""
def iou(a,b):
    a_x_min, a_y_min = a[0] - a[2]/2.0, a[1] - a[3]/2.0
    a_x_max, a_y_max = a[0] + a[2]/2.0, a[1] + a[3]/2.0
    b_x_min, b_y_min = b[0] - b[2]/2.0, b[1] - b[3]/2.0
    b_x_max, b_y_max = b[0] + b[2]/2.0, b[1] + b[3]/2.0
    a_area, b_area = a[2] * a[3], b[2] * b[3]
    
    inter_width = max(0,min(a_x_max, b_x_max) - max(a_x_min, b_x_min))
    inter_height = max(0, min(a_y_max, b_y_max) - max(a_y_min, b_y_min))
    inter_area = inter_width * inter_height
    union_area = (a_area + b_area) - inter_area

    return inter_area / union_area


"""
Vectorised form of the intersection over union aka Jacquard index
"""
def _iou(a,b):    
    a_x_min, a_y_min = a[:,1], a[:,2]
    a_x_max, a_y_max = (a[:,3] + a_x_min), (a[:,4] + a_y_min)
    b_x_min, b_y_min = b[:,1], b[:,2]
    b_x_max, b_y_max = (b[:,3] + b_x_min), (b[:,4] + b_y_min)
    area_a = a[:,3] * a[:,4]
    area_b = b[:,3] * b[:,4]
    zero = torch.zeros(a_x_min.size()).float()    

    inter_width = torch.max(zero, torch.min(a_x_max, b_x_max) - torch.max(a_x_min,b_x_min))
    inter_height = torch.max(zero, torch.min(a_y_max, b_y_max) - torch.max(a_y_min,b_y_min))
    inter_area = inter_width * inter_height
    union_area = (area_a + area_b) - inter_area
    
    jac_index = inter_area / union_area    
    return jac_index
    
        

def convert_center_coords_to_noorm(bboxes):
    (rows,cols) = (7,7)    
    stride = 64
    assert (rows * cols) == bboxes.size(0)
    #generate the strides for each grid position  
    grid_size = 7  
    grid = np.arange(grid_size)
    row,col = np.meshgrid(grid,grid)
    row = torch.FloatTensor(row).view(-1,1)
    col = torch.FloatTensor(col).view(-1,1)
    grid = torch.cat((row,col),1) * stride
    # center coordinates
    bboxes[:,1:3] = (bboxes[:,1:3] * stride).round()
    bboxes[:,1:3] = bboxes[:,1:3] + grid
    bboxes[:,3:] = (bboxes[:,3:].pow(2) * 448).round()
    # Convert x,y to top left coords and leave the width and height as they are
    bboxes[:,1] -= bboxes[:,3]/2
    bboxes[:,2] -= bboxes[:,4]/2
    
    return bboxes


def convert_cls_idx_name(name_mapping, arr):
    return [name_mapping[x] for x in arr]