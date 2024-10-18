import configparser
import argparse
import logging
import os
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO
from PIL import Image

from apply_yolo import process_image, calculate_single_bbox
from cropper import crop_image
from image_service import ImageService
from PytorchWildlife.models import detection as pw_detection


# from apply_dino import Dino, make_classification_eval_transform


class Pipeline:
    def __init__(
        self,
        name,
        input_dataset_path,
        output_path,
        reid_params,
        yolo_path,
        prediction_params,
        allowed_extensions,
        save_output_videos=False,
    ):
        self.name = name
        self.input_dataset_path = input_dataset_path
        self.output_path = output_path
        self.save_output_videos = save_output_videos
        self.allowed_extensions = allowed_extensions
        logging.info(f"Loading yolo model from: {Path(yolo_path)}")
        self.yolo = YOLO(yolo_path)
        self.detection_model = pw_detection.MegaDetectorV5(
            device="cuda", pretrained=True
        )  # Model weights are automatically downloaded.
        self.yolo.to("cuda")
        self.prediction_params = prediction_params
        # self.reid_model = Dino(model_path=reid_params["model_path"], 'cuda' if torch.cuda.is_available() else 'cpu')
        self.image_service = ImageService(
            input_dataset_path,
            output_path,
            allowed_extensions["allowed_img_extensions"],
            allowed_extensions["allowed_video_extensions"],
            self.save_output_videos,
        )

    def run(self):
        logging.info("Detecting faces using Yolo.")
        for img in self.image_service:
            bboxes, images = self.detect_face(img)
            print("dawda", images)
            print("img", img)
            paths = self.crop_to_face(bboxes, images, img)
            # for path in paths:
            #     img = Image.open(path).convert("RGB")9
            #     img = make_classification_eval_transform()(img)
            #     s = self.reid_model(img.unsqueeze(0))
            #     print("PREDICTIOn", s)
            #     # apply dino
        logging.info("Done so far.")

    def detect_face(self, img):
        label_path = os.path.join(self.output_path, "labels", img[1])
        output_video = None
        os.makedirs(label_path, exist_ok=True)
        detected_tracks = process_image(
            img[0],
            self.yolo,
            self.detection_model,
            self.prediction_params["confidence"],
            self.prediction_params["vid_stride"],
        )

        image_bboxes = []
        for key, images in detected_tracks.items():
            for idx, bbox in enumerate(images["bboxes"]):
                label = (
                    img[0].stem + f"_id_{key}"
                    if len(images["result"]) == 1
                    else img[0].stem + f"_id_{key}" + f"_{idx}"
                )
                local_label_path = os.path.join(label_path, label + ".txt")
                with open(local_label_path, "w") as f:
                    single_box = calculate_single_bbox(bbox, images["width"], images["height"])
                    line = f"0 {single_box[0]} {single_box[1]} {single_box[2]} {single_box[3]}"# {bbox[4]}"
                    f.write("%s\n" % line)
                image_bboxes.append({"label": label, "bboxes": bbox})

        output = []
        for key, result in detected_tracks.items():
            for res in result["result"]:
                output.append(res)

        # collect all the boxes per frame and save them as a video
        bboxes_per_frame = {}
        for key, images in detected_tracks.items():
            for idx, res in enumerate(images["result"]):
                frame_id = res["frame_id"]
                if not frame_id in bboxes_per_frame:
                    bboxes_per_frame[frame_id] = {"img": res["img"], "bboxes": []}
                bboxes_per_frame[frame_id]["bboxes"].append(res["xyxy"])

        # save the video with the boxes
        input_path = Path(img[0])
        if (
            self.save_output_videos
            and input_path.suffix.lower()[1:]
            in self.allowed_extensions["allowed_video_extensions"]
        ):
            video_output_path = os.path.join(self.output_path, "video_results", img[1])
            os.makedirs(video_output_path, exist_ok=True)
            cap = cv2.VideoCapture(str(input_path))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            scale_factor = 0.5  # Adjust the scale factor as needed (e.g., 0.5 for half size)
            new_width = int(frame_width * scale_factor)
            new_height = int(frame_height * scale_factor)
            output_video = cv2.VideoWriter(
                os.path.join(video_output_path, img[0].stem + "_output.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (frame_width, frame_height)
            )

            for frame_id, frame in bboxes_per_frame.items():
                print(bboxes_per_frame)
                for bboxes in frame["bboxes"]:
                    if not isinstance(bboxes, list):
                        bboxes = [bboxes]
                    for bbox in bboxes:
                        
                        #draw_img = 
                        #print(bboxes)
                        x1, y1, x2, y2 = bbox
                        #cv2.rectangle(draw_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.rectangle(frame["img"], (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        output_video.write(frame["img"])

            output_video.release()
            cap.release()

        return image_bboxes, output

    def crop_to_face(self, image_bboxes, images, img):
        cropped_path = os.path.join(self.output_path, "cropped", img[1])
        os.makedirs(cropped_path, exist_ok=True)
        cropped_paths = []
        for idx, (bboxes, image) in enumerate(zip(image_bboxes, images)):
            if len(bboxes) == 0:
                logging.warning(f"No bboxes found for {img[0].stem}")
                continue
            local_cropped_path = os.path.join(cropped_path, bboxes["label"] + ".png")
            # local_cropped_path = os.path.join(cropped_path, img[0].stem + ".png") if len(images) == 1 else os.path.join(cropped_path, img[0].stem + f"_{idx}.png")
            print(bboxes)
            print(image)
            cropped_image = Image.fromarray(image['img']).crop(bboxes["bboxes"])
            cropped_image.save(local_cropped_path)
            cropped_paths.append(local_cropped_path)
            #crop_image(image, bboxes["bboxes"])
        return cropped_paths

    def save_bboxes(self):
        raise NotImplementedError

    def calculate_embeddings(self):
        raise NotImplementedError

    def fiftyone(self):
        raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", required=True, help="Name of the dataset")
    ap.add_argument(
        "-d", "--dataset-path", required=True, help="Path to the image folder"
    )
    ap.add_argument("-o", "--output-path", required=True, help="Path to the new folder")
    args = vars(ap.parse_args())

    config = configparser.ConfigParser()
    config.read("config.ini")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Pipeline started")
    pipeline = Pipeline(
        name=args["name"],
        input_dataset_path=args["dataset_path"],
        prediction_params=config["PREDICTION"],
        reid_params=config["REID"],
        output_path=args["output_path"],
        yolo_path=config["DEFAULT"]["yolo_path"],
        allowed_extensions={
            "allowed_img_extensions": config["DEFAULT"]["allowed_img_extensions"],
            "allowed_video_extensions": config["DEFAULT"]["allowed_video_extensions"],
        },
        save_output_videos=True,
    )
    pipeline.run()
