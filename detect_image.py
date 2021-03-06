import argparse
import torch
import torchvision.transforms as transforms
import time
from PIL import Image
from yolo_v1 import YOLOv1
from utilities import build_class_names, predict, draw_detection, im2PIL
from transforms import RandomBlur, RandomHorizontalFlip, RandomVerticalFlip

parser = argparse.ArgumentParser(description="Detect objects in images")
parser.add_argument('--image', dest='image_path', help='The path to the image')
parser.add_argument('--model', dest='model_path', help='Pretrained YOLOv1 model weights')

_IMAGE_SIZE_ = (448,448)
_GRID_SIZE_ = 7
_MODEL_PATH_ = "./model_checkpoints/yolo_v1_model_80_epoch.pth"

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

    rhf = RandomVerticalFlip(probability=1.0)

    if args.image_path:
        print(f"-> Detecting objects in '{args.image_path}'")
        with torch.no_grad():
            image = Image.open(args.image_path).convert('RGB')
            old_size = image.size
            image = image.resize(_IMAGE_SIZE_, Image.ANTIALIAS)
            image_ = transform(image).unsqueeze(0)
            # image.show()
            flipped_image, _ = rhf((image,[]))
            # flipped_image.show()

            batch_idx = 0
            start_time = time.time()
            predictions = predict(model, image_, 0.6) #[B,N,1,6] or [B,1,1,6]
            elapsed = time.time() - start_time
            predictions = predictions[batch_idx] #[N,1,6]

            # for bbox in predictions:
            for idx in range(0, predictions.size(0)):
                bbox = predictions[idx,:][0]
                print(bbox)
                pred_class = int(bbox[5])
                draw_detection(image, bbox[:4]/_IMAGE_SIZE_[0], class_names[pred_class])
            image.resize(old_size, Image.ANTIALIAS)
            image.show()
        
        print(f"Total time taken {elapsed//60:.0f}m {elapsed%60:.0f}s")



