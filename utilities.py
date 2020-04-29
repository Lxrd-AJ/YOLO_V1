import numpy as np 
import matplotlib.pyplot as plt
import torch
import math
from PIL import Image, ImageDraw, ImageFont


_DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.waitforbuttonpress()

def im2PIL(inp):
    """Converts Tensor to PIL Image"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = np.uint8(inp * 255)    
    inp = Image.fromarray(inp)
    return inp


"""
Non-Maximum Suppression
- bboxes: Expected to be of size 'num_boxes x 6'
"""
def nms(bboxes, threshold):    
    # Get the bounding boxes for each class
    # print(bboxes)
    classes = bboxes[:,5].int().tolist()
    classes = list(set(classes))
    results = []
    for cls in classes:
        cls_mask = (bboxes[:,5] == float(cls)).unsqueeze(1) * bboxes        
        #use the class confidence, as it is not possible for this to be zero due to confidence thresholding
        cls_mask = torch.nonzero(cls_mask[:,0]).squeeze()
        class_preds = bboxes[cls_mask,:]
        
        if len(list(class_preds.size())) > 1: #if more then 1 bbox belongs to this class
            #`torch.sort` fails on single dimensional tensors hence this if block
            sorted_conf, sort_indices = torch.sort(class_preds[:,4], descending=True)
            class_preds = class_preds[sort_indices]
            
            for idx in range(class_preds.size(0)-1):
                cur_box = class_preds[idx]
                other_boxes = class_preds[idx+1:]
                repeated_cur_box = cur_box.repeat(other_boxes.size(0),1)

                ious = _iou(repeated_cur_box, other_boxes)
                if (ious < threshold).all():                    
                    results.append(cur_box)

                #if on the last box then add it as no other box before it will have a 
                # high iou with this box, if it did, it will be not have been added to
                # results
                if other_boxes.size(0) == 1: 
                    results.append(other_boxes[0])                
        else:
            results.append(class_preds)

    return results #FIXME: Ideally this should be a tensor not a python list

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
        
        # Adding the grid cell locations gx, gy to the detection for use in `gnd_truth_tensor` function
        grid_cells = torch.Tensor([gx,gy])
        detection[1:] = bbox
        res.append(torch.cat([detection, grid_cells]).unsqueeze(0))
    res = torch.cat(res, dim=0)
    return res #detections


"""
converts the output of a YOLO model to a center coordinate where x,y is the center of the box and w,h represent the square root of width
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
uses the detections in the YOLO format to construct the a tensor in the grid coordinates e.g 7x7x30 to make it easier for loss calculations.
Used only on the ground truth detection matrices
#TODO: Rewrite this function
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


def draw_detections(image, detections, class_names):    
    for det in detections:
        draw_detection(image, det[1:], class_names[int(det[0])])
    return image

"""
- bbox: [<x> <y> <width> <height>] is in normalised center coordinates where 
    x,y is the center of the box relative to the width and height of the image (grid cell)
        where x is ((x_max + x_min)/2)/width
    width and height is normalised relative to the width and height of `image`
- image: should be a PIL image
"""
def draw_detection(image, bbox, name, colour="white"):
    width, height = image.size
    
    box_width = int(bbox[2] * width)
    box_height = int(bbox[3] * height)
    
    center_x = int(bbox[0] * width)
    center_y = int(bbox[1] * height)    

    top_left = (center_x - box_width/2, center_y - box_height/2)
    bottom_right = (center_x + box_width/2, center_y + box_height/2)

    draw = ImageDraw.Draw(image)
    draw.rectangle((top_left, bottom_right), width=2, outline=colour)
    draw.text(top_left, name.upper(), fill=(250,180,148,255)) #, font=ImageFont.truetype("Helvetica",15)


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

"""
Returns the bounding box with the highest confidence score
- Each bounding box `b1` is in the format [x, y, w, h, conf]
"""
def max_box(b1, b2):
    assert b1.size() == b2.size()
    A = torch.zeros(b1.size()).float()
    for i in range(b1.size(0)):
        #4 is the index of the probability scores
        A[i,:] = b1[i,:] if b1[i,4] > b2[i,4] else b2[i,:]
    return A

"""
For a grid size of (7,7), It expects `A` to be of size 49x6
Assumes the class confidence score is the first element
"""
def confidence_threshold(A, conf_thresh):
    #4 is the index of the confidence value
    conf_mask = (A[:,4] > conf_thresh).float().unsqueeze(1)    
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
Vectorised form of the intersection over union aka Jacquard index.
Expects the bounding boxes in the center coordinates.
"""
def _iou(a,b):    
    a_x_min, a_y_min = a[:,0] - a[:,2]/2.0, a[:,1] - a[:,3]/2.0
    a_x_max, a_y_max = (a[:,0] + a[:,2]/2.0), (a[:,1] + a[:,3]/2.0)
    b_x_min, b_y_min = b[:,0] - b[:,2]/2.0, b[:,1] - b[:,3]/2.0
    b_x_max, b_y_max = b[:,0] + b[:,2]/2.0, b[:,1] + b[:,3]/2.0
    area_a = a[:,2] * a[:,3]
    area_b = b[:,2] * b[:,3]
    zero = torch.zeros(a_x_min.size()).float().to(_DEVICE_)    

    inter_width = torch.max(zero, torch.min(a_x_max, b_x_max) - torch.max(a_x_min,b_x_min))
    inter_height = torch.max(zero, torch.min(a_y_max, b_y_max) - torch.max(a_y_min,b_y_min))
    inter_area = inter_width * inter_height
    union_area = (area_a + area_b) - inter_area
    
    jac_index = inter_area / union_area    
    return jac_index
    
        

# def convert_center_coords_to_noorm(bboxes):
#     (rows,cols) = (7,7)    
#     stride = 64
#     assert (rows * cols) == bboxes.size(0)
#     #generate the strides for each grid position  
#     grid_size = 7  
#     grid = np.arange(grid_size)
#     row,col = np.meshgrid(grid,grid)
#     row = torch.FloatTensor(row).view(-1,1)
#     col = torch.FloatTensor(col).view(-1,1)
#     grid = torch.cat((row,col),1) * stride
#     # center coordinates
#     bboxes[:,1:3] = (bboxes[:,1:3] * stride).round()
#     bboxes[:,1:3] = bboxes[:,1:3] + grid
#     bboxes[:,3:] = (bboxes[:,3:].pow(2) * 448).round()
#     # Convert x,y to top left coords and leave the width and height as they are
#     bboxes[:,1] -= bboxes[:,3]/2
#     bboxes[:,2] -= bboxes[:,4]/2
    
#     return bboxes


def convert_cls_idx_name(name_mapping, arr):
    return [name_mapping[x] for x in arr]