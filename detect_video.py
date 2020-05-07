import argparse
import torch
import torchvision.transforms as transforms
import time
import cv2
import numpy as np
from PIL import Image
from yolo_v1 import YOLOv1
from utilities import build_class_names, predict, draw_detection

parser = argparse.ArgumentParser(description="Detect objects in video")
parser.add_argument('--video', dest='video_path', help='The path to the Video file')
parser.add_argument('--model', dest='model_path', help='Pretrained YOLOv1 model weights')
parser.add_argument('--output', dest='output_path', help='The file name of the processed video')

_IMAGE_SIZE_ = (448,448)
_GRID_SIZE_ = 7
_MODEL_PATH_ = "./model_checkpoints/yolo_epoch_20_model.pth"

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
        capture = cv2.VideoCapture(args.video_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_counter = 0
        print(f"-> Processing {total_frames} frames in '{args.video_path}'")

        output_path = args.output_path if args.output_path is not None else f"./processed_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_capture = cv2.VideoWriter(output_path, fourcc, 30, (448,448))

        while capture.isOpened():
            ret, frame = capture.read() #NB: `frame` is a numpy array
            frame_counter += 1
            
            if ret:
                percent = (frame_counter/total_frames) * 100
                print(f"- [{percent:.0f}%] Processing frame {frame_counter}")
                image_PIL = Image.fromarray(frame).resize(_IMAGE_SIZE_, Image.ANTIALIAS)
                image = transform(image_PIL).unsqueeze(0)

                predictions = predict(model, image, 0.6)[0]
                for bbox in predictions:
                    #TODO: Fix bug: multi detections are not showing
                    try:
                        pred_class = class_names[int(bbox[5])]
                        print(f"\t-> Predicted {pred_class} with bounding box: {bbox}")
                        draw_detection(image_PIL, bbox[:4]/_IMAGE_SIZE_[0], pred_class)
                    except:
                        print(bbox)

                frame = np.array(image_PIL)
                cv2.imshow(args.video_path, frame)
                output_capture.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        
        capture.release()
        output_capture.release()
        cv2.destroyAllWindows()
