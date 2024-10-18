import cv2
import logging
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.data import transforms as pw_trans
import supervision as sv

from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt


import numpy as np
from cropper import only_crop_image
import supervision as sv

fixed_yolo_input_size = (416, 416)

def get_bbox_for_original_frame(body_bbox, yolo_box, original_cropped_width, original_cropped_height, yolo_input_size):
    x,y,_,_ = body_bbox
    scale_x = original_cropped_width / yolo_input_size[0]
    scale_y = original_cropped_height / yolo_input_size[1]
    x_min, y_min, x_max, y_max = yolo_box
    original_x_min = x_min * scale_x
    original_y_min = y_min * scale_y
    original_x_max = x_max * scale_x
    original_y_max = y_max * scale_y
    orig_x_min = original_x_min + x
    orig_y_min = original_y_min + y
    orig_x_max = original_x_max + x
    orig_y_max = original_y_max + y
    return (orig_x_min, orig_y_min, orig_x_max, orig_y_max)

def calculate_single_bbox(bbox, width, height):
    #box = bbox["xyxy"]#_face_in_body_crop"]  # Bounding box coordinates [x1, y1, x2, y2]
    x_center = (bbox[0] + bbox[2]) / 2 / width
    y_center = (bbox[1] + bbox[3]) / 2 / height
    box_width = (bbox[2] - bbox[0]) / width
    box_height = (bbox[3] - bbox[1]) / height
    return [x_center, y_center, box_width, box_height]

def process_image(image_path, model, detection_model, confidence, vid_stride):
    tracks = {}
    cap = cv2.VideoCapture(str(image_path))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Processing video {image_path} with {length} frames.")
    i = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if i % int(vid_stride) != 0:
            i += 1
            continue
        i += 1
        logging.info(f"Processing frame {i}")

        if success:
            # Convert frame to PIL Image for MegaDetector
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if img.mode == "RGBA":
                img = img.convert("RGB")
            np_img = np.array(img.convert("RGB"))
            transform = pw_trans.MegaDetector_v5_Transform(
                target_size=detection_model.IMAGE_SIZE, stride=detection_model.STRIDE
            )
            transformed_image = transform(np_img)
            F.to_pil_image(transformed_image).save('alfa.png')
            detection_result = detection_model.single_image_detection(
                transformed_image, np_img.shape
            )
            detections = detection_result["detections"]

            # Extract gorilla bounding boxes from MegaDetector detections
            gorilla_boxes = []
            for idx, class_id in enumerate(detections.class_id):
                if class_id == 0:  # Assuming class_id 0 is for gorillas/animals
                    gorilla_boxes.append(detections.xyxy[idx])

            if len(gorilla_boxes) == 0:
                logging.info("No gorillas detected by MegaDetector in this frame.")
                continue

            # Crop the frame based on the bounding boxes and run YOLOv8 on each crop
            if len(gorilla_boxes) == 0:
                continue
            for box in gorilla_boxes:
                # cropped_img = only_crop_image(img, *box)
                Image.fromarray(np_img).save('merc.png')
                cropped_img = sv.crop_image(image=np_img, xyxy=box)
                Image.fromarray(cropped_img).save('./audi.png')
                # Run YOLOv8 tracking on the cropped frame
                cropped_frame = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                original_cropped_width, original_cropped_height = Image.fromarray(cropped_img).size
                cropped_frame = cv2.resize(cropped_frame, fixed_yolo_input_size)
                results = model.track(
                    cropped_frame,
                    persist=True,
                    conf=float(confidence),
                    vid_stride=int(vid_stride),
                    device=0,
                    iou=0.2,
                )
                nothing_in_track = False
                if results[0].boxes.is_track is False:
                    nothing_in_track = True
                    results = model.predict(
                        cropped_frame,
                        conf=float(confidence),
                        vid_stride=int(vid_stride),
                    )
                    gorilla_detected_in_predict = len(results[0].boxes.cls) > 0
                    print(
                        "Nothing in track, predicting with predict function",
                        gorilla_detected_in_predict,
                    )

                for result in results:
                    if result.boxes.id is not None:
                        for idx, id in enumerate(result.boxes.id):
                            id = (
                                "no_track"
                                if nothing_in_track and gorilla_detected_in_predict
                                else str(int(id.item()))
                            )
                            if id not in tracks:
                                tracks[id] = {}
                                tracks[id]["result"] = []

                            xyxy_face_in_body_crop = result.boxes.xyxy[idx].tolist()
                            xyxy_body_crop_in_full_frame = box
                            xyxy_face_in_full_frame = np.add(
                                xyxy_face_in_body_crop, xyxy_body_crop_in_full_frame
                            )
                            bbox_in_orig_image = get_bbox_for_original_frame(body_bbox=box, yolo_box=xyxy_face_in_body_crop, original_cropped_height=original_cropped_height, original_cropped_width=original_cropped_width, yolo_input_size=fixed_yolo_input_size)
                            tracks[id]["result"].append(
                                {
                                    "xyxy": bbox_in_orig_image,
                                    "xyxy_face_in_body_crop": xyxy_face_in_body_crop,
                                    "img": np_img,
                                    "transformed_img": transformed_image,
                                    "body_img": cropped_frame,
                                    "img_cropped": result.orig_img,
                                    "cls": result.boxes.cls[idx].item(),
                                    "frame_id": i,
                                }
                            )
        else:
            break

    for key in tracks.keys():
        tracks[key]["bboxes"] = []  # calculate_bbox(tracks[result]['result'])
        tracks[key]["width"] = Image.fromarray(np_img).size[0]
        tracks[key]["height"] = Image.fromarray(np_img).size[1]
        for result in tracks[key]["result"]:
            #print("res", result["img_cropped"])
            #Image.fromarray(result["img_cropped"]).save("fiat.png")
            tracks[key]["bboxes"].append(
                result["xyxy"]
                # calculate_single_bbox(
                #     result,
                #     Image.fromarray(result["img_cropped"], "RGB").size[0],
                #     Image.fromarray(result["img_cropped"], "RGB").size[1],
                # )
            )
    cap.release()
    return tracks
