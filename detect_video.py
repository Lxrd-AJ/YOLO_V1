import argparse
import torch
import torchvision.transforms as transforms
import time
import cv2
from PIL import Image
from yolo_v1 import YOLOv1
from utilities import build_class_names, predict, draw_detection

parser = argparse.ArgumentParser(description="Detect objects in video")
parser.add_argument('--video', dest='video_path', help='The path to the Video file')
parser.add_argument('--model', dest='model_path', help='Pretrained YOLOv1 model weights')
parser.add_argument('--output', dest='output_path', help='The file name of the processed video')

_IMAGE_SIZE_ = (448,448)
_GRID_SIZE_ = 7
_MODEL_PATH_ = "./model_checkpoints/25_epochs_yolo_v1.pth"

if __name__ == "__main__":
    args = parser.parse_args()

    class_names = build_class_names("./voc.names")
    model = YOLOv1(class_names, _GRID_SIZE_)
    model_path = args.model_path if args.model_path is not None else _MODEL_PATH_
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=torch_device))
    model.eval()

    transform = transforms.Compose([    
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    start_time = time.time()
    print(f"Using pretrained weights: {model_path}")

    if args.video_path:
        print(f"-> Processing objects in '{args.video_path}'")
        capture = cv2.VideoCapture(args.video_path)

        output_path = args.output_path if args.output_path is not None else f"processed_{args.video_path}"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        output_capture = cv2.VideoWriter(output_path, fourcc, 30, (448,448))

        while capture.isOpened():
            ret, frame = capture.read()
            
            if ret:
                print(type(frame))
                print(frame.shape)

                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        
        capture.release()
        output_capture.release()
        cv2.destroyAllWindows()
