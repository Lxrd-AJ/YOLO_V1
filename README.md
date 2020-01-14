# YOLO_V1
You Only Look Once version 1

# To do
- [ ] Data Processing
    - [x] Download the VOC dataset
    - [x] Custom Dataset class
        - [x] Code to show bounding boxes given image and detections
        - [x] Code to resize an image and its detections to a given size
    - [x] Data Transformations
        - [x] Resize images to correct size
            - [x] Need to modify the detection labels
        - [x] Convert from Pascal VOC center normalised coordinates to YOLO box encoding        
        - [x] Random exposure and saturatation
- [x] Network Design
    - [x] Add dropout after first fully connected
- [ ] Training 
    - [ ] Loss function
- [ ] Modify the network to use a resnet101 (https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.resnet101) as the extraction layer in yolo_v1.py(L27) OR a SqueezeNet model (lesser resource intensive) https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.squeezenet1_1
    - [ ] Need to apply `self.final_conv` in my implementation to the output of `self.layer4` layer in PyTorch to support transfer learning OR `self.classifier` in SqueezeNet. Need to go from 512 output channels of layer4 to the 256 input channels final_conv requires
    - [x] Add the new Normalise transform to adjust the mean and std of the image
            `normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            `
- [ ] Inference
    - [ ] Non-maximum suppression
- [ ] Model Metrics evaluation
    - https://github.com/rafaelpadilla/Object-Detection-Metrics
    - https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
    - [ ] mAP (Mean average precision)
    - [ ] Precision - Recall curve
- [ ] Open Images Challenge 
    - https://storage.googleapis.com/openimages/web/challenge2019.html
    - [ ] Custom dataset class
- [ ] Datat Augmentation
    - [ ] Random scaling
    - [ ] Random translation