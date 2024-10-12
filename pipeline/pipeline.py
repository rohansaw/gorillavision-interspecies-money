import configparser
import argparse
import logging
import os
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO
from PIL import Image

from apply_yolo import process_image
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
                    line = f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}"
                    f.write("%s\n" % line)
                image_bboxes.append({"label": label, "bboxes": bbox})

        output = []
        for key, result in detected_tracks.items():
            for res in result["result"]:
                output.append(res)

        # collect all the boxes per frame and save them as a video
        bboxes_per_frame = {}
        for key, images in detected_tracks.items():
            print(images)
            for idx, res in enumerate(images["result"]):
                frame_id = res["frame_id"]
                if not frame_id in bboxes_per_frame:
                    bboxes_per_frame[frame_id] = {"img": res["img"], "bboxes": []}
                bboxes_per_frame[frame_id]["bboxes"].append(res["xyxy"])

        # save the video with the boxes

        input_path = Path(img[0])
        if (
            self.save_output_videos
            and input_path.suffix.lower()
            in self.allowed_extensions["allowed_video_extensions"]
        ):
            video_output_path = os.path.join(self.output_path, "video_results", img[1])
            os.makedirs(video_output_path, exist_ok=True)
            cap = cv2.VideoCapture(str(input_path))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            output_video = cv2.VideoWriter(
                os.path.join(video_output_path, img[1] + "_output.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (frame_width, frame_height),
            )

            for frame_id, frame in bboxes_per_frame.items():
                for bboxes in frame["bboxes"]:
                    for bbox in bboxes:
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame["img"], (x1, y1), (x2, y2), (0, 255, 0), 2)
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
            cropped_image = crop_image(image, bboxes["bboxes"])
            cropped_image.save(local_cropped_path)
            cropped_paths.append(local_cropped_path)
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
    )
    pipeline.run()
