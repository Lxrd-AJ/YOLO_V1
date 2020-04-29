# YOLO_V1
You Only Look Once version 1

* Dataset length in voc_dataset.py
* Batch size & L158 & L181 in training.py

# TODO:
- [ ] Try changing self.final_conv in YOLOv1.py to use a 3x3 filter instead of a 1x1
    - [ ] Fix NaNs occuring during training
- [x] Check the batch_collate_fn and ensure it is logically correct in training.py
    - [ ] Check the `box` function in loss.py and sure the bounding boxes are in the correct coordinate
- [ ] Add pretrained model to digital ocean space dir       
- [ ] Enable file logging
- [ ] Remove torch.autograd.set_detect_anomaly(True) from training.py