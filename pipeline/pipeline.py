import configparser
import argparse
import logging
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

from apply_yolo import process_image
from cropper import crop_image
from image_service import ImageService

class Pipeline:
    def __init__(self, name, input_dataset_path, output_path, yolo_path, prediction_params, allowed_extensions):
        self.name = name
        self.input_dataset_path = input_dataset_path
        self.output_path = output_path
        logging.info(f"Loading yolo model from: {Path(yolo_path)}")
        self.yolo = YOLO(yolo_path)
        self.prediction_params = prediction_params
        self.image_service = ImageService(input_dataset_path, output_path, allowed_extensions["allowed_img_extensions"], allowed_extensions["allowed_video_extensions"])

    def run(self):
        logging.info("Detecting faces using Yolo.")
        for img in self.image_service:
            labels, images = self.detect_face(img)
            self.crop_to_face(labels, images, img)
        l = map(lambda x: 
                    self.crop_to_face(x, self.detect_face(x)), self.image_service)
        logging.info("Done so far.")

    def detect_face(self, img):
        label_path = os.path.join(self.output_path, "labels", img[1])
        os.makedirs(label_path, exist_ok=True)
        images = process_image(img[0], self.yolo, self.prediction_params["confidence"], self.prediction_params["vid_stride"])
        image_bboxes= []
        for idx, image in enumerate(images):
            local_label_path = os.path.join(label_path, img[0].stem + ".txt") if len(images) == 1 else os.path.join(label_path, img[0].stem + f"_{idx}.txt")
            bboxes = image['bboxes']
            with open(local_label_path, 'w') as f:
                for item in bboxes:
                    line = f"{item[0]} {item[1]} {item[2]} {item[3]} {item[4]}"
                    f.write("%s\n" % line)
            image_bboxes.append(bboxes)
        return image_bboxes, list(map(lambda x: x['img'], images))

    def crop_to_face(self, image_bboxes, images, img):
        cropped_path = os.path.join(self.output_path, "cropped", img[1])
        os.makedirs(cropped_path, exist_ok=True)
        cropped_paths = []
        for idx, (bboxes, image) in enumerate(zip(image_bboxes, images)):
            if len(bboxes) == 0:
                logging.warning(f"No bboxes found for {img[0].stem}")
                continue
            local_cropped_path = os.path.join(cropped_path, img[0].stem + ".png") if len(images) == 1 else os.path.join(cropped_path, img[0].stem + f"_{idx}.png")
            cropped_image = crop_image(image, bboxes)
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
        output_path = args["output_path"], 
        yolo_path=config['DEFAULT']['yolo_path'], 
        allowed_extensions={'allowed_img_extensions': config['DEFAULT']['allowed_img_extensions'], 'allowed_video_extensions': config['DEFAULT']['allowed_video_extensions']}
    )
    pipeline.run()
