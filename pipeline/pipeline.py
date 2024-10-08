import configparser
import argparse
import logging
import os
import shutil
from pathlib import Path

from ultralytics import YOLO
from PIL import Image

from apply_yolo import process_image
from cropper import crop_image
from image_service import ImageService
# from apply_dino import Dino, make_classification_eval_transform

class Pipeline:
    def __init__(self, name, input_dataset_path, output_path, reid_params, yolo_path, prediction_params, allowed_extensions):
        self.name = name
        self.input_dataset_path = input_dataset_path
        self.output_path = output_path
        logging.info(f"Loading yolo model from: {Path(yolo_path)}")
        self.yolo = YOLO(yolo_path)
        self.yolo.to('cuda')
        self.prediction_params = prediction_params
        # self.reid_model = Dino(model_path=reid_params["model_path"], 'cuda' if torch.cuda.is_available() else 'cpu')
        self.image_service = ImageService(input_dataset_path, output_path, allowed_extensions["allowed_img_extensions"], allowed_extensions["allowed_video_extensions"])

    def run(self):
        logging.info("Detecting faces using Yolo.")
        for img in self.image_service:
            bboxes, images = self.detect_face(img)
            paths = self.crop_to_face(bboxes, images, img)
            # for path in paths:
            #     img = Image.open(path).convert("RGB")
            #     img = make_classification_eval_transform()(img)
            #     s = self.reid_model(img.unsqueeze(0))
            #     print("PREDICTIOn", s)
            #     # apply dino
        l = map(lambda x: 
                    self.crop_to_face(x, self.detect_face(x)), self.image_service)
        logging.info("Done so far.")

    def detect_face(self, img):
        label_path = os.path.join(self.output_path, "labels", img[1])
        os.makedirs(label_path, exist_ok=True)
        results = process_image(img[0], self.yolo, self.prediction_params["confidence"], self.prediction_params["vid_stride"])
        image_bboxes = []
        for key, images in results.items():
            print("aaw", key, len(images))
            print(type(images))
            for idx, bbox in enumerate(images['bboxes']):
                label = img[0].stem + f"_id_{key}" if len(images['result']) == 1 else img[0].stem + f"_id_{key}" + f"_{idx}"
                local_label_path = os.path.join(label_path, label + ".txt")
                with open(local_label_path, 'w') as f:
                    line = f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}"
                    f.write("%s\n" % line)
                image_bboxes.append({'label': label, 'bboxes':bbox})
        output = []
        for key, result in results.items():
            for res in result['result']:
                output.append(res)
        return image_bboxes, output

    def crop_to_face(self, image_bboxes, images, img):
        cropped_path = os.path.join(self.output_path, "cropped", img[1])
        os.makedirs(cropped_path, exist_ok=True)
        cropped_paths = []
        for idx, (bboxes, image) in enumerate(zip(image_bboxes, images)):
            if len(bboxes) == 0:
                logging.warning(f"No bboxes found for {img[0].stem}")
                continue
            local_cropped_path = os.path.join(cropped_path, bboxes['label'] + ".png")
            #local_cropped_path = os.path.join(cropped_path, img[0].stem + ".png") if len(images) == 1 else os.path.join(cropped_path, img[0].stem + f"_{idx}.png")
            cropped_image = crop_image(image, bboxes['bboxes'])
            cropped_image.save(local_cropped_path)
            cropped_paths.append(local_cropped_path)
        return cropped_paths

    def save_bboxes(self):
        raise NotImplementedError

    def calculate_embeddings(self):
        raise NotImplementedError


    def fiftyone(self):
        raise NotImplementedError

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", required=True, help="Name of the dataset")
    ap.add_argument("-d", "--dataset-path", required=True, help="Path to the image folder")
    ap.add_argument("-o", "--output-path", required=True, help="Path to the new folder")
    args = vars(ap.parse_args())

    config = configparser.ConfigParser()
    config.read('config.ini')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Pipeline started")
    pipeline = Pipeline(
        name=args['name'],
        input_dataset_path=args['dataset_path'],
        prediction_params=config["PREDICTION"],
        reid_params = config["REID"],
        output_path = args["output_path"], 
        yolo_path=config['DEFAULT']['yolo_path'],
        allowed_extensions={'allowed_img_extensions': config['DEFAULT']['allowed_img_extensions'], 'allowed_video_extensions': config['DEFAULT']['allowed_video_extensions']}
    )
    pipeline.run()
