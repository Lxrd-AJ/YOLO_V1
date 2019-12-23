import numpy as np 
import torch

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

def iou(a,b):    
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