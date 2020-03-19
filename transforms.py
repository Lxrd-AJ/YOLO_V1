import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import PIL
# import cv2
import numpy as np
from PIL import ImageOps, Image, ImageFilter
from data.voc_dataset import VOCDataset
from utilities import build_class_names, draw_detection, im2PIL

#TODO: Random Scaling
#TODO: Random Translation



class RandomScale(object):
    """
    TODO: RandomScale needs to be fixed
    """
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, items):
        if random.random() < self.p:
            img, det = items
            scale = 0.8 #random.uniform(0.8, 1.2)
            width, height = img.size
            # img = img.resize((int(width*scale), height))
            # img = ImageOps.fit(img, (int(width*scale),height), Image.ANTIALIAS)
            
            # croppedImage = img.crop((10,10,300,300))
            # print(det)
            # print(img.size)
            n_width, n_height = int(width * scale), int(height * scale)
            croppedImage = TF.center_crop(img, (n_width,n_height))
            print(croppedImage.size)
            delta_w = 1 - (n_width/width)
            delta_h = 1 - (n_height/height)
            for idx, bbox in enumerate(det):
                orig_x, orig_y, orig_w, orig_h = det[idx,1:] * 448
                print(orig_x, orig_y, orig_w, orig_h)
                det[idx,1] = (scale * orig_x) / n_width
                det[idx,2] = (scale * orig_h) / n_height
                
            croppedImage = croppedImage.resize((width,height))
            
            return (croppedImage, det)
        else:
            return items


class RandomHorizontalFlip(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, items):
        if random.random() < self.p:
            img, det = items
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            for idx,bbox in enumerate(det):
                det[idx,1] = 1 - bbox[1]
            return (img, det)
        else:
            return items

class RandomBlur(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, image):
        if random.random() < self.p:
            return image.filter(ImageFilter.GaussianBlur(radius=2))
        return image

class RandomVerticalFlip(object):
    def __init__(self, probability=0.5):
        self.p = probability
        self.t = transforms.RandomVerticalFlip(p=1)

    def __call__(self, items):
        if random.random() < self.p:
            img, det = items
            img = self.t(img)
            for idx,bbox in enumerate(det):
                det[idx,2] = 1 - bbox[2]
            return (img, det)
        else:
            return items

if __name__ == "__main__":
    #Test the transforms out
    class_names = build_class_names("./voc.names")
    #Image only transforms
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.12), ratio=(0.1, 1.1)),
    ])

    dataset = VOCDataset(f"./data/train.txt", transform=[image_transform, normalise_transform], pair_transform=pair_transform)

    image, detections = dataset[random.randint(0, len(dataset))]
    # image, detections = dataset[100]
    image = im2PIL(image)
    true_image = image.copy()
    for bbox in detections:
        c = int(bbox[0])        
        draw_detection(true_image,bbox[1:], class_names[c], "white")
    true_image.show()

    #Apply the transform on the image
    # image, detections = pair_transform((image,detections))
    # image = image_transform(image)
    # image = im2PIL(normalise_transform(image))
    
    # for bbox in detections:  
    #     print(bbox)     
    #     c = int(bbox[0])        
    #     draw_detection(image,bbox[1:], class_names[c], "white")

    # image.show()
