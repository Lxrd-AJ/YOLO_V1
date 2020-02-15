from utilities import convert_YOLO_to_center_coords, iou, im2PIL, draw_detections, build_class_names
import torch

"""
- Does not support batching, it operates on a single target-output pair
"""
def criterion(output, target, stride):
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
                loss_gx_gy = (z.t().matmul(Q).matmul(z)) + (0.5 * min_bbox[0]) + (truth_probs - class_probs).t().matmul(truth_probs - class_probs)                            
            else:
                #0.5 is the weight when there is no object
                #0.5 is multiplied by the class confidences for each bounding box in the current grid
                loss_gx_gy = 0.5 * ((bbox_1[0]*bbox_1[0]) + (bbox_2[0]*bbox_2[0])) 
            
            total_loss += loss_gx_gy
    return total_loss


if __name__ == "__main__":
    # Test the loss function
    class_names = build_class_names("./voc.names")
    images = torch.randn(1, 3, 448, 448)
    detections = torch.rand(1, 1, 5)
    detections[:,:,0] *= 20
    
    x = im2PIL(images[0])            
    x = draw_detections(x, detections[0], class_names)
    x.show()
