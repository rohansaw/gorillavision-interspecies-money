import cv2
import logging
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife import utils as pw_utils
from PIL import Image
import torchvision.transforms as T
import numpy as np
from cropper import only_crop_image
import supervision as sv
fixed_size = (416, 416)
def calculate_single_bbox(bbox, width, height):
    # try:
    #     logging.info(f"Detected {bbox.cls} with confidence {bbox.conf:.2f}")
    # except:
    #     print("Error")
    # print(bbox)
    box = bbox['xyxy']  # Bounding box coordinates [x1, y1, x2, y2]
    x_center = (box[0] + box[2]) / 2 / width
    y_center = (box[1] + box[3]) / 2 / height
    box_width = (box[2] - box[0]) / width
    box_height = (box[3] - box[1]) / height
    return [int(bbox['cls']), x_center, y_center, box_width, box_height]
def process_image(image_path, model, detection_model, confidence, vid_stride):
    tracks = {}
    cap = cv2.VideoCapture(str(image_path))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Processing video {image_path} with {length} frames.")
    i = 0

    # Initialize MegaDetector model
    

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
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            np_img = np.array(img.convert("RGB"))
            transform = pw_trans.MegaDetector_v5_Transform(target_size=detection_model.IMAGE_SIZE,
                                               stride=detection_model.STRIDE)
            detection_result = detection_model.single_image_detection(transform(np_img), np_img.shape)
            detections = detection_result['detections']

            # Extract gorilla bounding boxes from MegaDetector detections
            gorilla_boxes = []
            for idx, class_id in enumerate(detections.class_id):
                if class_id == 0:  # Assuming class_id 0 is for gorillas/animals
                    gorilla_boxes.append(detections.xyxy[idx])

            if len(gorilla_boxes) == 0:
                logging.info("No gorillas detected by MegaDetector in this frame.")
                continue

            # Crop the frame based on the bounding boxes and run YOLOv8 on each crop
            for box in gorilla_boxes:
                # cropped_img = only_crop_image(img, *box)
                cropped_img = sv.crop_image(
                    image=np_img, xyxy=box
                    )

                # Run YOLOv8 tracking on the cropped frame
                cropped_frame = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                #if res is False:
                   # print("FALSLEL")
                cropped_frame = cv2.resize(cropped_frame, (416, 416))
                res = cv2.imwrite('./bmw.png', cropped_frame)

                results = model.track(cropped_frame, persist=True, conf=float(confidence), vid_stride=int(vid_stride),device=0, iou=0.2)
                nothing_in_track = False
                if results[0].boxes.is_track is False:
                    nothing_in_track = True
                    results = model.predict(cropped_frame, conf=float(confidence), vid_stride=int(vid_stride))
                    gorilla_detected_in_predict = len(results[0].boxes.cls) > 0
                    print("Nothing in track, predicting with predict function", gorilla_detected_in_predict)

                for result in results:
                    if result.boxes.id is not None:
                        for idx, id in enumerate(result.boxes.id):
                            id = "no_track" if nothing_in_track and gorilla_detected_in_predict else str(int(id.item()))
                            print("JJJJ", id == "no_track")
                            if id not in tracks:
                                tracks[id] = {}
                                tracks[id]['result'] = []
                            tracks[id]['result'].append({'xyxy': result.boxes.xyxy[idx].tolist(), 'img': result.orig_img, 'cls': result.boxes.cls[idx].item()})
        else:
            break
    for key in tracks.keys():
        tracks[key]['bboxes'] = [] #calculate_bbox(tracks[result]['result'])
        for result in tracks[key]['result']:
            tracks[key]['bboxes'].append(calculate_single_bbox(result, Image.fromarray(result['img'], "RGB").size[0], Image.fromarray(result['img'], "RGB").size[1]))
    cap.release()
    return tracks