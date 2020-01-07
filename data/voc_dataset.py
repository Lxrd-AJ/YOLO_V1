import torch 
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageDraw

"""
- bbox cls <x> <y> <width> <height> is in normalised center coordinates where 
    x,y is the center of the box relative to the width and height of the image (grid cell)
        where x is ((x_max + x_min)/2)/width
    width and height is normalised relative to the width and height of `image`
"""
def show_detection(image, bbox):
    width, height = image.size
    # get the scales
    dw = 1/width
    dh = 1/height
    
    box_width = int(bbox[3] * width)
    box_height = int(bbox[4] * height)

    print(box_width, box_height)
    center_x = int(bbox[1] * width)
    center_y = int(bbox[2] * height)
    print(center_x, center_y)

    top_left = (center_x - box_width/2, center_y - box_height/2)
    bottom_right = (center_x + box_width/2, center_y + box_height/2)

    draw = ImageDraw.Draw(image)
    draw.rectangle((top_left, bottom_right), width=3)


class VOCDataset(data.Dataset):
    def __init__(self, data_file, transform=None):        
        self.transform = transform
        with open(data_file) as file:            
            self.image_paths = [x.strip() for x in file.readlines()]            

    def __len__(self):
        return len(self.image_paths)

    """
    Returns (Image, [det])
    where Image is a torch tensor with format CxWxH
    """
    def __getitem__(self, idx):
        img_file = self.image_paths[idx]
        label_file = img_file.replace("JPEGImages","labels").replace("jpg","txt")
        annotation_file = img_file.replace("JPEGImages","Annotations").replace("jpg","xml")
        print(img_file, label_file)
        img = Image.open(img_file)
        print(img.size)

        with open(label_file) as file:
            # <object-class> <x> <y> <width> <height>            
            detections = file.readlines()
            detections = [x.split() for x in detections]
            detections = [[float(c) for c in detection] for detection in detections]
            
            for detection in detections:
                print(detection)
                show_detection(img, detection)
                print()
            img.show()
            
        with open(annotation_file) as file:
            print(file.read())

        if self.transform:
            img = self.transform(img)
        return img, None

