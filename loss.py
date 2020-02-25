from utilities import convert_YOLO_to_center_coords, iou, _iou, im2PIL, draw_detections, build_class_names
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np



def normalised_to_global(A, width=448, height=448):
    """
    Converts a bounding box in the center normalised coordinates to the center global 
    coordinates.
    A is in the format N*5 where N is the number of bounding boxes and each bounding
    box is in the format <x> <y> <w> <h> <class>

    - returns   A in the same size as input
    """
    A[:,0] = A[:,0] * width
    A[:,1] = A[:,1] * height
    A[:,2] = A[:,2] * width
    A[:,3] = A[:,3] * height 
    return A

def cell_to_global(A, im_size=448, stride=64, B=2, S=7):
    """
    Receives a tensor in the format N*(B*5) that represents each cell's bounding box 
    prediction where each cell is in the format <x> <y> <w> <h> <conf> encoded 
    in the YOLO format where it normalised relative to the grid cell and N is the
    total number of grids i.e S*S and B is the number of bounding boxes
    It returns the grid cell coordinates with respect to the global image.

    - return B:     Tensor of size N*(B*5), same size as input where the <x> <y> 
                    <w> <h> <conf> in each cell is wrt to the global image. This is
                    still in the center normalised coordinate.
    """
    
    rng = np.arange(S) # the range of possible grid coords
    cols, rows = np.meshgrid(rng, rng)
    
    #create a grid with each cell containing the (x,y) location multiplied by stride  
    rows = torch.FloatTensor(rows).view(-1,1)
    cols = torch.FloatTensor(cols).view(-1,1)
    grid = torch.cat((rows,cols),1) * stride
    
    bboxes = torch.split(A, A.size(1)//B, 1) #split the N*10 bboxes into two N*5 sets

    res = []
    for v in bboxes: # v would be of size N*5
        #convert the <x> <y> and <w> <h> cell coordinates to global image coordinates
        v[:,:2] = (v[:,:2] * stride).round() + grid
        v[:,2:4] = (v[:,2:4].pow(2) * im_size).round()        
        res.append(v)
    res = torch.cat(res,1)    
    return res


def box(output, target, size=448, B=2):
    """
    Returns the box to use for loss calculation. This is either the box with the 
    highest confidence or the box with the highest intersection over union
    with the target
    Receives an output prediction of size S*S*(B*5+C) where each cell is in
    the format <x> <y> <w> <h> <conf> | <x> <y> <w> <h> <conf> | <cls.......probs>
    assuming number of bounding boxes is 2
    and `target` ground truth of size S*S*5 where target is in the format
        <x> <y> <w> <h> <cls>
    and returns the bounding box to use
    for each grid cell.

    - return bbox:  Tensor of size SxSx(5+C) where each bounding box is in 
                    the format <x> <y> <w> <h> <confidence> <cls probs>
                    The coordinate, width and height are in global image coordinates
    """
    
    #Reshape the output tensor into (S*S)*(B*5+C) to make it easier to work with
    sz = output.size()
    output = output.view(sz[0] * sz[1], -1) #e.g 49*30
    pred_bboxes = output[:,:B*5] #slice out only the bounding boxes e.g 49*10
    pred_classes = output[:,B*5:] #slice out the pred classes  e.g 49x10
    target = target.view(sz[0] * sz[1], -1) #e.g 49*5

    pred_bboxes = cell_to_global(pred_bboxes) #e.g 49*10
    target = normalised_to_global(target) #e.g 49*5

    num_classes = output.size(1) - (B*5)
    
    R = torch.zeros(output.size(0),5+num_classes) #result to return    
    for i in range(output.size(0)): #loop over each cell coordinate
        # `bboxes` will be a tuple of size B (e.g 2), where each elem is 1*5
        bboxes = torch.split(pred_bboxes[i,:], pred_bboxes.size(1)//B)        
        bboxes = torch.stack(bboxes)        

        """
        In the case where there is a ground truth tensor at the current grid cell,
        the predicted bounding box with the highest intersection over union to the
        ground truth is chosen.
        If there is no ground truth prediction at the current cell, just pick the
        bounding box with the highest confidence
        """

        #case 1: There is a ground truth prediction at this cell i
        if target[i].sum() > 0:#select the box with the highest intersection over union
            repeated_target = target[i].repeat(bboxes.size(0),1).detach()
            jac_idx = _iou(bboxes.clone().detach(), repeated_target)
            
            max_iou_idx = torch.argmax(jac_idx)
            R[i,:5] = bboxes[max_iou_idx,:]
        else: #select the box with the highest confidence
            highest_conf_idx = torch.argmax(bboxes[:,4])
            R[i,:5] = bboxes[highest_conf_idx,:]

        #Add the predicted class confidence to the results
        R[i,5:] = pred_classes[i]
        
    return R.view(sz[0], sz[1], -1)


def criterion(output, target, lambda_coord = 5, lambda_noobj=0.5): #, stride
    """
    Computes the average loss (YOLO) between the output and the target batch tensor
    - It assumes the both the output and target are encoded in the YOLO format
        where <x> and <y> are normalised to the grid cell
        and <w> and <h> is the square root of the width of the object / width of image
    - The output is of size NxSxSx(Bx5+C) where B is the no. of bounding boxes encoded 
        in the YOLO format
    - The target is of size NxSxSx5 in format <x> <y> <w> <h> <class> not encoded in 
        the YOLO format but normalised wrt the image
    """

    batch_loss = torch.tensor(0).float()
    for idx, out_tensor in enumerate(output):
        best_boxes = box(out_tensor, target[idx]) #e.g 7x7x(5+20)
        sz = best_boxes.size()
        P = best_boxes.view(sz[0] * sz[1], -1) #e.g 49x25
        G = target[idx].view(sz[0] * sz[1], -1) #e.g 49x5

        image_loss = torch.tensor(0).float()

        for i in range(P.size(0)): #loop over each cell coordinate
            if G[i].sum() > 0: #there is a ground truth prediction at this cell
                pred_cls = P[i,5:]
                true_cls = torch.zeros(pred_cls.size())
                true_cls[int(G[i,4])] = 1                

                # grid cell regression loss
                grid_loss = lambda_coord * torch.pow(P[i,0:2] - G[i,0:2], 2).sum() \
                    + lambda_coord * torch.pow(torch.sqrt(P[i,2:4]) - torch.sqrt(G[i,2:4]),2).sum() \
                    + torch.pow(P[i,4] - 1,2) \
                    + torch.pow(pred_cls - true_cls, 2).sum() # class probability loss
            else:
                grid_loss = lambda_noobj * torch.pow(P[i,4] - 0,2) #confidence should be zero
            
            image_loss += grid_loss
        print(f"Image {i}th loss = {image_loss}")
        batch_loss += image_loss

    print(f"Batch loss = {batch_loss}")
    print(f"Avg batch loss = {batch_loss/output.size(0)}")

    exit(0)

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

    """
    Ground truth prediction format
    - It is not in global image coordinates
    - It is in the center normalised form where x,y,w,h have been divided by the image
        width and height (not encoded in the YOLO format)
    """
    Y = torch.rand(2,7,7,5)
    Y[:,:,:,:4] = torch.clamp(Y[:,:,:,:4], 0,1) #This is not encoded in the YOLO format
    Y[:,:,:,4] = (Y[:,:,:,4] * 20).floor()
    #zero out some cells to represent no ground truth data at those locations
    mask = torch.empty(7,7).uniform_(0,1)
    mask = torch.bernoulli(mask).bool()
    Y[:,mask,:] = 0
    

    for i in range(25):
        optimiser.zero_grad()

        Y_ = net(X)        
        loss = criterion(Y_, Y)
        print(f"Test Epoch {i}: Loss = {loss.item()}")

        loss.backward()
        optimiser.step()

    print(f"Finished testing model and Loss function")
