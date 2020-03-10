import random
import torchvision.transforms as transforms
from data.voc_dataset import VOCDataset
from utilities import build_class_names, draw_detection

class RandomFlip(object):
    """
    Flips the given image and bouding boxes
    """

    def __init__(self, probability=0.5):
        pass



if __name__ == "__main__":
    #Test the transforms out
    class_names = build_class_names("./voc.names")
    #Image only transforms
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    #Image detection pair transforms
    # pair_transform = transforms.Compose([])

    dataset = VOCDataset(f"./data/train.txt", transform=transform)

    image, detections = dataset[random.randint(0, len(dataset))]
    for bbox in detections:  
        print(bbox)     
        c = int(bbox[0])        
        draw_detection(image,bbox[1:], class_names[c], "white")
    image.show()

    #Apply the transform on the image
    image = transform(image)
    image.show()
