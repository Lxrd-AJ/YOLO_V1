# YOLO_V1
You Only Look Once version 1

# To do
- [ ] Data Processing
    - [x] Download the VOC dataset
    - [x] Custom Dataset class
        - [x] Code to show bounding boxes given image and detections
        - [x] Code to resize an image and its detections to a given size
    - [ ] Data Transformations
        - [x] Resize images to correct size
            - [x] Need to modify the detection labels
        - [ ] Random scaling
        - [ ] Random translation
        - [ ] Random exposure and saturatation
- [ ] Network Design
    - [x] Add dropout after first fully connected
- [ ] Training 
    - [ ] Loss function
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