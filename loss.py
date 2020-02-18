from utilities import convert_YOLO_to_center_coords, iou, im2PIL, draw_detections, build_class_names
import torch
import torch.nn as nn
import torch.nn.functional as F 


def box(output, target):
    """
    Receives an output prediction of size S*S*(B*5+C) 
    and `target` ground truth of size S*S*5 and returns the bounding box to use
    for each grid cell.

    - return bbox:  Tensor of size SxSx(5+C) where each bounding box is in 
                    the format <x> <y> <w> <h> <confidence>
    """
    #TODO: Continue here `similar to predict_one_box`
    pass

"""
- Does not support batching, it operates on a single target-output pair
"""
def criterion(output, target, stride):
    """
    Computes the loss (YOLO) between the output and the target tensor
    - It assumes the both the output and target are encoded in the YOLO format
        where <x> and <y> are normalised to the grid cell
        and <w> and <h> is the square root of the width of the object / width of image
    - The output is of size NxSxSx(Bx5+C) where B is the no. of bounding boxes
    - The target is of size NxSxSx5 in format <x> <y> <w> <h> <class>
    #TODO: Finish refactoring loss function
    """
    total_loss = 0.0
    num_grids = output.size()
    #TODO: Reduce this to a single for-loop using np.meshgrid
    for grid_x in range(num_grids[0]):
        for grid_y in range(num_grids[1]):
            truth_bbox = target[grid_x, grid_y]
            pred_bboxs = output[grid_x, grid_y].cpu()
            
            # Find the intersection over unio between the two predicted bounding boxes at the grid location
            bbox_1, bbox_2 = pred_bboxs[0:5], pred_bboxs[5:10]
            _bbox_1 = convert_YOLO_to_center_coords(bbox_1[1:], grid_x, grid_y, stride)
            _bbox_2 = convert_YOLO_to_center_coords(bbox_2[1:], grid_x, grid_y, stride)
            _truth_bbox = convert_YOLO_to_center_coords(truth_bbox[1:5], grid_x, grid_y, stride) 
            
            class_probs = pred_bboxs[10:]
            truth_probs = truth_bbox[5:]            

            if truth_bbox.sum() > 0: #there is an object in this class
                max_bbox, min_bbox = (bbox_1, bbox_2) if iou(_bbox_1,_truth_bbox) > iou(_bbox_2, _truth_bbox) else (bbox_2, bbox_1)
                confidence = max(iou(_bbox_1, _truth_bbox),iou(_bbox_2, _truth_bbox))
                truth_bbox[0] = confidence #the ground truth data uses confidence
                Q = torch.eye(5) * 5
                Q[0,0] = 1
                z = max_bbox - truth_bbox[0:5]
                #loss is weighted bounding box regression + weighted no object + class probabilities                
                loss_gx_gy = (z.t().matmul(Q).matmul(z)) \
                                + (0.5 * min_bbox[0]) \
                                + (truth_probs - class_probs).t().matmul(truth_probs - class_probs)                
            else:
                #0.5 is the weight when there is no object
                #0.5 is multiplied by the class confidences for each bounding box in the current grid
                loss_gx_gy = 0.5 * ((bbox_1[0]*bbox_1[0]) + (bbox_2[0]*bbox_2[0])) 
            
            total_loss += loss_gx_gy
    return total_loss


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 5, 1, 0 ),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.linear = nn.Sequential(
            nn.Linear(512, 1024, True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1470),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        x = x.view(-1, 7, 7, 30)        
        return x

if __name__ == "__main__":
    # Test the loss function
    # class_names = build_class_names("./voc.names")
    # images = torch.randn(1, 3, 448, 448)
    # detections = torch.rand(1, 1, 5)
    # detections[:,:,0] *= 20
    
    # x = im2PIL(images[0])            
    # x = draw_detections(x, detections[0], class_names)
    # x.show()
    """
    This test assumes that I can build a simple model that can overfit to 
    a random batch of images over at least 20 epochs
    """
    net = TestNet()
    optimiser = torch.optim.Adam(net.parameters(), lr=0.001)

    X = torch.randn(2, 3, 448, 448)    
    Y = torch.rand(2,7,7,5)
    Y[:,:,:,:4] = torch.clamp(Y[:,:,:,:4], 0,1)
    Y[:,:,:,4] = (Y[:,:,:,4] * 20).floor()
    

    for i in range(25):
        optimiser.zero_grad()

        Y_ = net(X)        
        loss = criterion(Y_, Y)
        print(f"Test Epoch {i}: Loss = {loss.item()}")

        loss.backward()
        optimiser.step()

    print(f"Finished testing model and Loss function")
