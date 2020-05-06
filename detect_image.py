import argparse
import torch
import torchvision.transforms as transforms
import time
from PIL import Image
from yolo_v1 import YOLOv1
from utilities import build_class_names, predict, draw_detection

parser = argparse.ArgumentParser(description="Detect objects in images")
parser.add_argument('--image', dest='image_path', help='The path to the image')
parser.add_argument('--model', dest='model_path', help='Pretrained YOLOv1 model weights')

_IMAGE_SIZE_ = (448,448)
_GRID_SIZE_ = 7
_MODEL_PATH_ = "./model_checkpoints/25_epochs_yolo_v1.pth"

if __name__ == "__main__":
    """
    TODO:
    - [x] Detect from image path
    - [ ] Detect using a folder of images
    """
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

    if args.image_path:
        print(f"-> Detecting objects in '{args.image_path}'")
        with torch.no_grad():
            image = Image.open(args.image_path).convert('RGB').resize(_IMAGE_SIZE_, Image.ANTIALIAS)
            image_ = transform(image).unsqueeze(0)
            image.show()

            predictions = predict(model, image_)[0]
            for bbox in predictions:
                print(bbox)
                pred_class = int(bbox[5])
                draw_detection(image, bbox[:4]/_IMAGE_SIZE_[0], class_names[pred_class])
            image.show()
        elapsed = time.time() - start_time
        print(f"Total time taken {elapsed//60:.0f}m {elapsed%60:.0f}s")



